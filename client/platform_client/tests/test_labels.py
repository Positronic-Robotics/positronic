"""The label of an episode, its views, and the labelers, against a recorded MolmoSpaces sweep and a RoboLab sweep.

The MolmoSpaces fixture holds 20 episodes of one sweep of `pi05_droid_jointpos`, in the `<block>/<episode>` layout that
positronic records: each `static.json`, the id from each `meta.json`, and the time column of each `robot_state.q`
signal. The labeler reads each end from that signal. The RoboLab fixture holds six episodes of one sweep of a pi05
policy: each `static.json`, the id from each `meta.json`, and the last 15 samples of each `subtask` signal.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval_vocabulary.episode import (
    BENCHMARK_TASK,
    ENDED_BY,
    ENDED_BY_OPERATOR,
    ENDED_BY_POLICY,
    ENV_ENDS_ON_SUCCESS,
    LEGACY_TIME_COLUMN,
    META_FILE,
    POLICY_END,
    SIGNAL_FILE_SUFFIX,
    STATIC_FILE,
    SUBTASK_SCORE_INDEX,
    SUBTASK_SIGNAL,
    SUCCESS,
    TASK,
    TERMINATED,
    TIMESTAMP_COLUMN_PREFIX,
    UID,
    VALUE_COLUMN,
    PolicyEnd,
)
from eval_vocabulary.outcome import OUTCOME
from eval_vocabulary.outcome import Outcome as CountedOutcome
from eval_vocabulary.timeline import EMITTED_WALL, EMITTED_WORLD
from platform_client import labels
from platform_client.labels import (
    MOLMO_LABELER,
    PUBLIC_LABELERS,
    ROBOLAB_LABELER,
    Attempt,
    Fail,
    ItemName,
    ItemsDelivered,
    Label,
    LabelerRef,
    Outcome,
    Safety,
    Stage,
    StageItems,
    StageReached,
    SubtaskAchieved,
    Success,
    label,
    molmo_label,
    read_static,
    recorded_task_and_outcome,
    robolab_label,
)
from platform_client.slug import slug_of
from pydantic import ValidationError

SWEEP = sorted(p for p in (Path(__file__).parent / 'fixtures' / 'molmo_sweep').glob('*/*') if p.is_dir())

CUP = 'pick up the cup.'
UID_OF_EPISODE = 'b9887e3fe4e44d0b6b872132588b034b'

# Three cubes go into a bowl, and the second trip carries two cubes at once.
CUBES = Label(
    uid=UID_OF_EPISODE,
    task='cubes-to-bowl',
    annotator=LabelerRef('rig-console'),
    seconds_to_end=30.0,
    total_items=3,
    events=(
        StageReached(seconds=4.0, stage=Stage.reaching),
        StageReached(seconds=9.5, stage=Stage.at_target),
        ItemsDelivered(seconds=10.0, delivered=1),
        StageReached(seconds=12.0, stage=Stage.reaching),
        StageReached(seconds=20.0, stage=Stage.at_target, items=StageItems(count=2)),
        ItemsDelivered(seconds=21.0, delivered=3),
        Success(seconds=21.0),
    ),
)


# --- the API words


def test_the_outcome_numbers_are_fixed_and_the_wire_carries_the_member_name():
    assert {outcome.name: outcome.value for outcome in Outcome if outcome} == {
        'success': 1,
        'fail': 2,
        'out_of_time': 3,
        'safety': 4,
    }
    assert slug_of(Outcome.out_of_time) == 'out_of_time'


def test_the_stage_numbers_are_ten_apart_from_the_lowest_stage_to_the_highest():
    stages = [stage for stage in Stage if stage]
    assert stages == sorted(stages)
    assert [stage.value for stage in stages] == [10, 20, 30, 40, 50]
    assert StageReached(seconds=1.0, stage=Stage.at_target).model_dump(mode='json')['stage'] == 'at_target'


def test_a_stage_mark_refuses_the_unset_stage():
    with pytest.raises(ValidationError):
        StageReached(seconds=1.0, stage=Stage.INVALID)


def test_a_label_reads_back_from_its_wire_form():
    assert Label.model_validate_json(CUBES.model_dump_json()) == CUBES


@pytest.mark.parametrize('value', ['', ' ', 'two words'])
def test_a_value_that_could_never_name_a_labeler_is_refused(value: str):
    with pytest.raises(ValueError):
        LabelerRef(value)


# --- stage items


@pytest.mark.parametrize(
    'names', [{}, {ItemName('spoon'): 3}, {ItemName('spoon'): 2, ItemName('fork'): 1}], ids=['none', 'one', 'two']
)
def test_stage_items_take_names_whose_counts_add_up_to_the_count(names: dict[ItemName, int]):
    assert StageItems(count=3, names=names).names == names


@pytest.mark.parametrize('names', [{'spoon': 2}, {'spoon': 3, 'fork': 1}], ids=['short', 'over'])
def test_stage_items_refuse_names_whose_counts_do_not_add_up_to_the_count(names: dict[str, int]):
    with pytest.raises(ValidationError, match='add up to'):
        StageItems.model_validate({'count': 3, 'names': names})


# --- the label's shape


def _label(*events: object, seconds_to_end: float = 240.0, **fields: object) -> Label:
    return Label.model_validate({
        'uid': UID_OF_EPISODE,
        'task': CUP,
        'seconds_to_end': seconds_to_end,
        'events': events,
        **fields,
    })


def test_a_label_refuses_events_out_of_time_order():
    with pytest.raises(ValidationError, match='ascending order'):
        _label(Success(seconds=5.0), StageReached(seconds=4.0, stage=Stage.reaching))


def test_a_label_refuses_an_event_after_the_episode_end():
    with pytest.raises(ValidationError, match='after the episode end'):
        _label(Fail(seconds=31.0), seconds_to_end=30.0)


@pytest.mark.parametrize(
    'change', [{'unknown': 1}, {'events': [{'kind': 'teleported', 'seconds': 1.0}]}, {'uid': ''}, {'total_items': 0}]
)
def test_a_label_refuses_a_field_or_an_event_it_does_not_know(change: dict[str, object]):
    with pytest.raises(ValidationError):
        Label.model_validate({**CUBES.model_dump(mode='json'), **change})


# Each count that a label carries, the total that bounds it, and the label fields that hold the count.
COUNTS_AND_TOTALS = [
    pytest.param(
        'items delivered',
        'total_items',
        lambda count: {'events': (ItemsDelivered(seconds=1.0, delivered=count),)},
        id='delivered',
    ),
    pytest.param(
        'items at a stage mark',
        'total_items',
        lambda count: {'events': (StageReached(seconds=1.0, stage=Stage.contact, items=StageItems(count=count)),)},
        id='stage-items',
    ),
    pytest.param('items lost', 'total_items', lambda count: {'lost_items': count}, id='lost'),
    pytest.param(
        'subtasks achieved',
        'total_subtasks',
        lambda count: {'events': (SubtaskAchieved(seconds=1.0, achieved=count),)},
        id='achieved',
    ),
]
_OTHER_TOTAL = {'total_items': 'total_subtasks', 'total_subtasks': 'total_items'}


@pytest.mark.parametrize(('what', 'total', 'counted'), COUNTS_AND_TOTALS)
def test_a_label_refuses_a_count_above_its_total(what: str, total: str, counted: Callable[[int], dict[str, Any]]):
    with pytest.raises(ValidationError, match=f'3 {what} is more than the total of 2'):
        _label(**counted(3), **{total: 2})


@pytest.mark.parametrize(('what', 'total', 'counted'), COUNTS_AND_TOTALS)
def test_a_label_takes_a_count_up_to_its_total_and_holds_it_to_no_other_total(
    what: str, total: str, counted: Callable[[int], dict[str, Any]]
):
    _label(**counted(2), **{total: 2})
    _label(**counted(5))
    _label(**counted(5), **{_OTHER_TOTAL[total]: 2})


def test_a_fall_in_the_delivered_count_does_not_hide_an_earlier_count_above_the_total():
    with pytest.raises(ValidationError, match='more than the total of 2'):
        _label(ItemsDelivered(seconds=1.0, delivered=3), ItemsDelivered(seconds=2.0, delivered=2), total_items=2)


def test_a_label_refuses_more_items_delivered_and_lost_than_its_total():
    with pytest.raises(ValidationError, match='3 items delivered and 1 items lost are more than the total of 3'):
        _label(ItemsDelivered(seconds=1.0, delivered=3), total_items=3, lost_items=1)


def test_a_label_bounds_only_its_last_delivered_count_with_its_lost_items():
    _label(ItemsDelivered(seconds=1.0, delivered=2), total_items=3, lost_items=1)
    _label(
        ItemsDelivered(seconds=1.0, delivered=3), ItemsDelivered(seconds=2.0, delivered=2), total_items=3, lost_items=1
    )
    _label(ItemsDelivered(seconds=1.0, delivered=3), lost_items=1)


def test_a_label_refuses_a_time_that_is_not_finite():
    with pytest.raises(ValidationError):
        _label(Success(seconds=1.0), seconds_to_end=float('inf'))


@pytest.mark.parametrize('seconds', [-1.0, float('nan')])
def test_a_label_cannot_stand_at_a_time_that_is_not_from_the_trial_start(seconds: float):
    with pytest.raises(ValueError, match='not a time from the trial start'):
        CUBES.until(seconds)


def test_until_names_fields_that_the_label_has():
    assert {'events', 'seconds_to_end', 'lost_items'} <= set(Label.model_fields)


# --- the views


def test_the_views_of_the_example():
    assert CUBES.outcome is Outcome.success
    assert (CUBES.seconds_to_outcome, CUBES.seconds_to_success) == (21.0, 21.0)
    assert CUBES.stage_seconds == {Stage.reaching: 4.0, Stage.at_target: 9.5}
    assert CUBES.furthest_stage is Stage.at_target
    assert CUBES.placements == (10.0, 21.0, 21.0)
    assert (CUBES.delivered, CUBES.subtask_score) == (3, None)
    assert CUBES.attempts == (
        Attempt(0, 0.0, 10.0, {Stage.reaching: 4.0, Stage.at_target: 9.5}),
        Attempt(1, 10.0, 21.0, {Stage.reaching: 12.0, Stage.at_target: 20.0}),
        Attempt(3, 21.0, 30.0, {}),
    )


def test_the_example_until_a_time_holds_the_events_up_to_it_and_ends_there():
    before = CUBES.until(15.0)
    assert (before.outcome, before.delivered, before.seconds_to_end) == (Outcome.out_of_time, 1, 15.0)
    assert before.events == CUBES.events[:4]
    assert CUBES.until(60.0) == CUBES


def test_a_label_until_a_time_before_its_end_has_lost_no_item():
    assert _label(seconds_to_end=100.0, total_items=3, lost_items=1).until(50.0).lost_items == 0


def test_a_label_until_its_end_or_later_keeps_its_lost_items():
    lossy = _label(seconds_to_end=100.0, total_items=3, lost_items=1)
    assert lossy.until(100.0).lost_items == 1
    assert lossy.until(150.0) == lossy


def test_a_label_with_no_end_event_ran_out_of_time():
    ran_out = _label(StageReached(seconds=3.0, stage=Stage.contact))
    assert (ran_out.outcome, ran_out.seconds_to_outcome, ran_out.seconds_to_success) == (
        Outcome.out_of_time,
        None,
        None,
    )
    assert ran_out.furthest_stage is Stage.contact


def test_the_earliest_end_event_decides_the_outcome():
    pressed = _label(Fail(seconds=100.0, reason='dropped it'), Success(seconds=120.0))
    assert (pressed.outcome, pressed.seconds_to_outcome, pressed.seconds_to_success) == (Outcome.fail, 100.0, None)
    assert _label(Success(seconds=50.0), Safety(seconds=60.0)).outcome is Outcome.success


@pytest.mark.parametrize(
    ('ends', 'outcome'),
    [((Safety, Fail, Success), Outcome.safety), ((Success, Fail), Outcome.fail), ((Success, Safety), Outcome.safety)],
)
def test_end_events_at_one_time_decide_safety_then_fail_then_success(
    ends: tuple[type[Success | Fail | Safety], ...], outcome: Outcome
):
    assert _label(*(end(seconds=10.0) for end in ends)).outcome is outcome


def test_a_press_keeps_its_press_time_under_a_time_limit():
    pressed = _label(StageReached(seconds=20.0, stage=Stage.control), Fail(seconds=100.0))
    assert pressed.until(99.0).outcome is Outcome.out_of_time
    assert pressed.until(100.0).outcome is Outcome.fail
    assert pressed.until(99.0).furthest_stage is Stage.control


def test_the_views_read_events_after_the_end_event_that_decides():
    late = _label(Safety(seconds=10.0), StageReached(seconds=12.0, stage=Stage.at_target))
    assert (late.outcome, late.furthest_stage) == (Outcome.safety, Stage.at_target)


def test_a_fall_in_the_items_count_takes_back_the_latest_items():
    recounted = _label(
        ItemsDelivered(seconds=1.0, delivered=2),
        ItemsDelivered(seconds=2.0, delivered=1),
        ItemsDelivered(seconds=3.0, delivered=2),
    )
    assert (recounted.placements, recounted.delivered) == ((1.0, 3.0), 2)


def test_without_an_items_count_the_attempts_split_at_each_change_of_the_subtasks_count():
    graded = _label(
        StageReached(seconds=1.0, stage=Stage.reaching),
        SubtaskAchieved(seconds=2.0, achieved=1, score=0.25),
        StageReached(seconds=3.0, stage=Stage.reaching),
        SubtaskAchieved(seconds=4.0, achieved=1, score=0.25),
        SubtaskAchieved(seconds=5.0, achieved=2, score=0.5),
        seconds_to_end=6.0,
        total_subtasks=4,
    )
    assert graded.subtask_score == 0.5
    assert graded.attempts == (
        Attempt(0, 0.0, 2.0, {Stage.reaching: 1.0}),
        Attempt(1, 2.0, 5.0, {Stage.reaching: 3.0}),
        Attempt(2, 5.0, 6.0, {}),
    )


def test_a_label_with_no_count_is_one_attempt():
    assert _label(StageReached(seconds=1.0, stage=Stage.contact), seconds_to_end=2.0).attempts == (
        Attempt(0, 0.0, 2.0, {Stage.contact: 1.0}),
    )


# --- the MolmoSpaces labeler on the recorded sweep


def test_each_episode_of_the_sweep_has_a_label_at_the_end_that_its_signal_records():
    given = [molmo_label(episode) for episode in SWEEP]
    labelled = [given_label for given_label in given if given_label is not None]
    assert len(labelled) == len(SWEEP)
    assert len({given_label.uid for given_label in labelled}) == len(SWEEP)
    assert {given_label.annotator for given_label in labelled} == {MOLMO_LABELER}
    assert all(given_label.events[0].seconds == given_label.seconds_to_end for given_label in labelled)

    first, fourth = labelled[0], labelled[3]
    assert (first.uid, first.outcome, first.seconds_to_end) == (
        'bb9887e3fe4e44d0b6b872132588b034',
        Outcome.fail,
        pytest.approx(30.03),
    )
    assert (fourth.uid, fourth.task, fourth.outcome, fourth.seconds_to_end) == (
        '4e11e705a42342ab8f8b8556d33298e8',
        'pick up the tissue.',
        Outcome.success,
        pytest.approx(4.686),
    )


def _episodes(
    tmp_path: Path, *statics: dict[str, object] | str | bytes | None, uid: str = UID_OF_EPISODE
) -> list[Path]:
    """One episode directory per given `static.json`, each with a meta that records `uid`.

    A mapping is written as JSON, a string or bytes as they are, and None leaves the episode without statics.
    """
    made = []
    for index, static in enumerate(statics):
        episode = tmp_path / f'{index:012d}'
        episode.mkdir()
        (episode / META_FILE).write_text(json.dumps({UID: uid}))
        if isinstance(static, bytes):
            (episode / STATIC_FILE).write_bytes(static)
        elif static is not None:
            (episode / STATIC_FILE).write_text(static if isinstance(static, str) else json.dumps(static))
        made.append(episode)
    return made


def _write_signal(episode: Path, name: str, times: list[int], column: str = LEGACY_TIME_COLUMN) -> None:
    pq.write_table(pa.table({column: times}), episode / f'{name}{SIGNAL_FILE_SUFFIX}')


def _write_timed_signal(episode: Path, name: str, world: list[int]) -> None:
    """A signal in the format with named timelines, slower than real time: ten wall seconds to each world second."""
    columns = {
        f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WALL}': [10 * t for t in world],
        f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WORLD}': world,
    }
    pq.write_table(pa.table(columns), episode / f'{name}{SIGNAL_FILE_SUFFIX}')


def test_an_untimed_verdict_sits_at_the_end_that_the_signals_record(tmp_path: Path):
    succeeded, failed = _episodes(tmp_path, {TASK: CUP, SUCCESS: True, TERMINATED: True}, {TASK: CUP, SUCCESS: False})
    _write_timed_signal(succeeded, 'grip', [1_000_000_000, 13_500_000_000])
    _write_timed_signal(failed, 'grip', [0, 3_000_000_000])
    assert molmo_label(succeeded) == _label(Success(seconds=12.5), seconds_to_end=12.5, annotator=MOLMO_LABELER)
    assert molmo_label(failed) == _label(Fail(seconds=3.0), seconds_to_end=3.0, annotator=MOLMO_LABELER)


def test_a_trial_that_ran_out_of_time_has_a_label_with_no_end_event(tmp_path: Path):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True, TERMINATED: False})
    _write_timed_signal(episode, 'grip', [0, 60_000_000_000])
    ran_out = molmo_label(episode)
    assert ran_out == _label(seconds_to_end=60.0, annotator=MOLMO_LABELER)
    assert ran_out is not None and ran_out.outcome is Outcome.out_of_time


def test_a_trial_that_an_operator_ended_has_the_label_of_its_verdict_and_none_without_one(tmp_path: Path):
    ended = {TASK: CUP, ENDED_BY: ENDED_BY_OPERATOR, TERMINATED: True}
    judged, unjudged, discarded = _episodes(
        tmp_path, {**ended, OUTCOME: CountedOutcome.FAIL}, ended, {**ended, OUTCOME: CountedOutcome.DISCARDED}
    )
    for episode in (judged, unjudged, discarded):
        _write_timed_signal(episode, 'grip', [0, 1_000_000_000])
    assert molmo_label(judged) == _label(Fail(seconds=1.0), seconds_to_end=1.0, annotator=MOLMO_LABELER)
    assert molmo_label(unjudged) is None
    assert molmo_label(discarded) is None


def test_a_trial_that_the_policy_gave_up_fails_at_its_end(tmp_path: Path):
    (episode,) = _episodes(
        tmp_path, {TASK: CUP, ENDED_BY: ENDED_BY_POLICY, POLICY_END: PolicyEnd.GIVE_UP, TERMINATED: True}
    )
    _write_timed_signal(episode, 'grip', [0, 2_000_000_000])
    assert molmo_label(episode) == _label(Fail(seconds=2.0), seconds_to_end=2.0, annotator=MOLMO_LABELER)


def test_the_end_is_the_span_of_all_signals_on_the_world_clock(tmp_path: Path):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True})
    _write_timed_signal(episode, 'grip', [2_000_000_000, 3_000_000_000])
    _write_timed_signal(episode, 'robot_state.q', [1_000_000_000, 4_500_000_000])
    assert molmo_label(episode) == _label(Success(seconds=3.5), seconds_to_end=3.5, annotator=MOLMO_LABELER)


def test_an_older_recording_takes_its_end_from_the_first_and_last_samples_of_its_signals(tmp_path: Path):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True})
    _write_signal(episode, 'grip', [2_000_000_000, 3_000_000_000])
    _write_signal(episode, 'robot_state.q', [1_000_000_000, 4_500_000_000])
    pq.write_table(pa.table({'ts_ns': [0, 9_000_000_000]}), episode / f'image.frames{SIGNAL_FILE_SUFFIX}')
    assert molmo_label(episode) == _label(Success(seconds=3.5), seconds_to_end=3.5, annotator=MOLMO_LABELER)


def test_a_recording_with_no_world_time_has_no_label_and_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True})
    _write_signal(episode, 'grip', [0, 1_000_000_000], column=f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WALL}')
    with caplog.at_level(logging.ERROR, logger=labels.__name__):
        assert molmo_label(episode) is None
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert EMITTED_WORLD in caplog.records[0].getMessage()


def test_a_signal_that_does_not_parse_leaves_the_episode_without_a_label(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True})
    (episode / f'grip{SIGNAL_FILE_SUFFIX}').write_bytes(b'not parquet')
    with caplog.at_level(logging.ERROR, logger=labels.__name__):
        assert molmo_label(episode) is None
    assert 'does not parse' in caplog.records[0].getMessage()


def test_a_recording_whose_meta_holds_no_id_has_no_label_and_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    (episode,) = _episodes(tmp_path, {TASK: CUP, SUCCESS: True})
    _write_timed_signal(episode, 'grip', [0, 1_000_000_000])
    (episode / META_FILE).write_text(json.dumps({'created_ts_ns': 1}))
    with caplog.at_level(logging.ERROR, logger=labels.__name__):
        assert molmo_label(episode) is None
    assert UID in caplog.records[0].getMessage()


def test_an_episode_with_no_verdict_has_no_label_and_reads_no_meta(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    episodes = _episodes(tmp_path, {TASK: CUP}, None, {TASK: CUP, TERMINATED: True})
    for episode in episodes:
        (episode / META_FILE).unlink()
    with caplog.at_level(logging.DEBUG, logger=labels.__name__):
        assert [molmo_label(episode) for episode in episodes] == [None, None, None]
    assert caplog.records == []


@pytest.mark.parametrize('static', ['{ truncated upload', '[true, "pick up the cup."]', b'{"task": "\xff"}'])
def test_statics_that_are_not_a_json_object_have_no_label_and_are_not_fatal(tmp_path: Path, static: str | bytes):
    (episode,) = _episodes(tmp_path, static)
    assert read_static(episode) is None
    assert molmo_label(episode) is None


# --- the RoboLab labeler


ROBOLAB_SWEEP = sorted(p for p in (Path(__file__).parent / 'fixtures' / 'robolab_sweep').glob('*/*') if p.is_dir())

# Read from the RoboLab recording: each episode's task and the score of its last step. The signal is float32.
ROBOLAB_LAST_STEP_SCORES = [
    ('FruitsOnPlateTask', pytest.approx(1 / 7)),
    ('GrabABagelTask', 0.0),
    ('PickUpBluePitcherTask', 0.5),
    ('PutBowlOnShelfTopTask', 0.0),
    ('ToolsPickingAllHammersTask', 0.0),
    ('ToyInBinTask', 0.0),
]


def _last_step_score(episode: Path) -> float:
    signal = pq.read_table(episode / f'{SUBTASK_SIGNAL}{SIGNAL_FILE_SUFFIX}', columns=[VALUE_COLUMN])
    return signal.column(VALUE_COLUMN)[-1].as_py()[SUBTASK_SCORE_INDEX]


def test_no_episode_of_the_robolab_sweep_completes_a_subtask_so_no_label_has_a_subtask_event():
    given = [robolab_label(episode) for episode in ROBOLAB_SWEEP]
    labelled = [given_label for given_label in given if given_label is not None]
    assert len(labelled) == len(ROBOLAB_SWEEP)
    assert [
        (given_label.task, _last_step_score(episode))
        for given_label, episode in zip(labelled, ROBOLAB_SWEEP, strict=True)
    ] == ROBOLAB_LAST_STEP_SCORES
    assert [given_label.events for given_label in labelled] == [
        (Fail(seconds=given_label.seconds_to_end),) for given_label in labelled
    ]
    assert {(given_label.annotator, given_label.outcome, given_label.total_subtasks) for given_label in labelled} == {
        (ROBOLAB_LABELER, Outcome.fail, 1)
    }

    first = labelled[0]
    assert (first.uid, first.seconds_to_end) == ('306d7c4674d34152b5741ebcfcfc15ff', pytest.approx(0.933, abs=1e-3))


BANANA = 'BananaInBowlTask'
# A task of two subtasks, whose score also grades the progress within a subtask.
CUBES_IN_BIN = 'CubesAndBlocksInBinTask'


def _write_subtask(episode: Path, samples: list[tuple[float, list[float]]]) -> None:
    """A `subtask` signal: the world time of each sample in seconds, and its `[status, completed, total, score]`."""
    columns = {
        f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WORLD}': pa.array([int(t * 1e9) for t, _ in samples], type=pa.int64()),
        VALUE_COLUMN: pa.array([progress for _, progress in samples], type=pa.list_(pa.float32())),
    }
    pq.write_table(pa.table(columns), episode / f'{SUBTASK_SIGNAL}{SIGNAL_FILE_SUFFIX}')


def test_a_robolab_label_has_an_event_at_each_change_of_the_subtasks_count_and_none_for_the_score_alone(tmp_path: Path):
    (episode,) = _episodes(
        tmp_path,
        {
            TASK: 'put all the cubes and blocks in the bin',
            BENCHMARK_TASK: CUBES_IN_BIN,
            SUCCESS: False,
            TERMINATED: True,
        },
    )
    _write_subtask(
        episode,
        [
            (0.0, [0.0, 0.0, 0.0, 0.0]),  # the frame of the reset
            (1.0, [0.0, 0.0, 2.0, 0.0]),
            (2.0, [0.0, 0.0, 2.0, 0.25]),
            (3.0, [0.0, 1.0, 2.0, 0.5]),
            (4.0, [0.0, 1.0, 2.0, 0.75]),
            (5.0, [0.0, 1.0, 2.0, 0.75]),
        ],
    )
    assert robolab_label(episode) == _label(
        SubtaskAchieved(seconds=3.0, achieved=1, score=0.5),
        Fail(seconds=5.0),
        task=CUBES_IN_BIN,
        seconds_to_end=5.0,
        total_subtasks=2,
        annotator=ROBOLAB_LABELER,
    )


@pytest.mark.parametrize(
    ('verdict', 'outcome'),
    [({SUCCESS: True, TERMINATED: True}, Outcome.success), ({SUCCESS: False, TERMINATED: False}, Outcome.out_of_time)],
)
def test_a_robolab_trial_ends_as_a_molmo_trial_ends(tmp_path: Path, verdict: dict[str, object], outcome: Outcome):
    (episode,) = _episodes(tmp_path, {BENCHMARK_TASK: CUBES_IN_BIN, **verdict})
    _write_subtask(episode, [(0.0, [0.0, 0.0, 2.0, 0.0]), (1.0, [0.0, 1.0, 2.0, 0.5])])
    given = robolab_label(episode)
    assert given is not None
    assert (given.outcome, given.subtask_score, given.seconds_to_end) == (outcome, 0.5, 1.0)


def test_a_robolab_trial_with_no_subtask_sample_has_no_label_and_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    missing, empty = _episodes(tmp_path, *[{BENCHMARK_TASK: BANANA, SUCCESS: True, TERMINATED: True}] * 2)
    for episode in (missing, empty):
        _write_timed_signal(episode, 'grip', [0, 1_000_000_000])
    _write_subtask(empty, [])
    with caplog.at_level(logging.DEBUG, logger=labels.__name__):
        assert [robolab_label(missing), robolab_label(empty)] == [None, None]
    assert caplog.records == []


def test_a_subtask_signal_of_reset_frames_only_states_no_total(tmp_path: Path):
    (episode,) = _episodes(tmp_path, {BENCHMARK_TASK: BANANA, SUCCESS: False, TERMINATED: True})
    _write_subtask(episode, [(0.0, [0.0, 0.0, 0.0, 0.0])])
    given = robolab_label(episode)
    assert given is not None
    assert (given.total_subtasks, given.subtask_score) == (None, None)


# --- the recorded verdict


TIMED_OUT = {TASK: CUP, TERMINATED: False}
COMPLETED = {TASK: CUP, ENDED_BY: ENDED_BY_POLICY, POLICY_END: PolicyEnd.COMPLETE, TERMINATED: True}


@pytest.mark.parametrize(
    ('static', 'recorded'),
    [
        ({TASK: CUP, SUCCESS: True}, (CUP, Outcome.success)),
        ({TASK: CUP, SUCCESS: False, TERMINATED: True}, (CUP, Outcome.fail)),
        (TIMED_OUT, (CUP, Outcome.out_of_time)),
        ({**TIMED_OUT, SUCCESS: True}, (CUP, Outcome.out_of_time)),
        ({**COMPLETED, ENV_ENDS_ON_SUCCESS: True}, (CUP, Outcome.fail)),
        ({**COMPLETED, OUTCOME: CountedOutcome.SUCCESS}, (CUP, Outcome.success)),
    ],
)
def test_the_recorded_verdict_gives_the_outcome(static: dict[str, object], recorded: tuple[str, Outcome]):
    assert recorded_task_and_outcome(static, TASK) == recorded


@pytest.mark.parametrize(
    ('counted', 'outcome'),
    [
        (CountedOutcome.SUCCESS, Outcome.success),
        (CountedOutcome.FAIL, Outcome.fail),
        (CountedOutcome.OUT_OF_TIME, Outcome.out_of_time),
        (CountedOutcome.SAFETY, Outcome.safety),
        (CountedOutcome.SYSTEM, Outcome.fail),
        (CountedOutcome.UNSCORED, None),
        (CountedOutcome.DISCARDED, None),
    ],
)
def test_each_counted_outcome_gives_its_label_outcome_or_none(counted: CountedOutcome, outcome: Outcome | None):
    assert recorded_task_and_outcome({TASK: CUP, OUTCOME: counted}, TASK) == (
        None if outcome is None else (CUP, outcome)
    )


def test_a_complete_that_nobody_confirmed_has_no_outcome():
    assert recorded_task_and_outcome(COMPLETED, TASK) is None


def test_a_success_of_the_wrong_type_beside_a_timeout_is_logged_and_the_trial_still_ran_out_of_time(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.DEBUG, logger=labels.__name__):
        assert recorded_task_and_outcome({**TIMED_OUT, SUCCESS: 'yes'}, TASK) == (CUP, Outcome.out_of_time)
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert SUCCESS in caplog.records[0].getMessage()


@pytest.mark.parametrize(
    ('static', 'misrecorded'),
    [
        ({SUCCESS: 1, TASK: CUP}, SUCCESS),
        ({SUCCESS: True, TASK: 3}, TASK),
        ({SUCCESS: None, TASK: CUP}, SUCCESS),
        ({SUCCESS: True, TASK: None}, TASK),
    ],
)
def test_a_success_or_task_of_the_wrong_type_is_no_verdict_and_is_logged(
    static: dict[str, object], misrecorded: str, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger=labels.__name__):
        assert recorded_task_and_outcome(static, TASK) is None
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert misrecorded in caplog.records[0].getMessage()


@pytest.mark.parametrize('static', [{TASK: CUP}, {SUCCESS: True}, {TASK: CUP, TERMINATED: True}, {TERMINATED: False}])
def test_a_missing_success_or_task_is_no_verdict_and_logs_nothing(
    static: dict[str, object], caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.DEBUG, logger=labels.__name__):
        assert recorded_task_and_outcome(static, TASK) is None
    assert caplog.records == []


@pytest.mark.parametrize(
    ('static', 'expected'), [({TASK: CUP, SUCCESS: True}, (CUP, Outcome.success)), ({TASK: CUP}, None)]
)
def test_a_wrong_typed_timeout_marker_is_logged_and_read_as_absent(
    static: dict[str, object], expected: tuple[str, Outcome] | None, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger=labels.__name__):
        assert recorded_task_and_outcome({**static, TERMINATED: 'false'}, TASK) == expected
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert TERMINATED in caplog.records[0].getMessage()


# --- precedence


def _never(episode: Path) -> Label | None:
    return None


def _withheld(episode: Path) -> Label | None:
    return _label(Success(seconds=1.0), seconds_to_end=1.0)


def test_label_takes_the_first_labeler_that_labels_the_episode():
    first = LabelerRef('first')
    table = {**PUBLIC_LABELERS, first: _never, LabelerRef('withheld'): _withheld}
    assert label(SWEEP[3], (first, MOLMO_LABELER, LabelerRef('withheld')), table) == (
        MOLMO_LABELER,
        molmo_label(SWEEP[3]),
    )
    assert label(SWEEP[3], (LabelerRef('withheld'), MOLMO_LABELER), table) == (
        LabelerRef('withheld'),
        _withheld(SWEEP[3]),
    )
    assert label(SWEEP[3], (first,), table) is None


def test_label_refuses_a_labeler_that_the_table_does_not_hold():
    with pytest.raises(LookupError, match="no labeler 'held_out'; the labelers are molmo, robolab$"):
        label(SWEEP[0], (MOLMO_LABELER, LabelerRef('held_out')))
