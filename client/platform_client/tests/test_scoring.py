"""The scorers and the tally, against a recorded MolmoSpaces sweep.

The fixture holds the `static.json` of each episode of one 20-episode sweep of `pi05_droid_jointpos`, in
the `<block>/<episode>` layout that positronic records. Five of the twenty episodes succeeded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pytest
from eval_vocabulary import outcome as verdicts
from eval_vocabulary.episode import STATIC_FILE, SUCCESS, TASK, TERMINATED
from platform_client.evals import MOLMO_SCORER, PUBLIC_EVALS, EvalRef, ScorerRef, public_eval
from platform_client.responses import Scores as PublishedScores
from platform_client.scoring import (
    PUBLIC_SCORERS,
    Outcome,
    Scores,
    molmo_outcome,
    read_static,
    recorded_task_and_success,
    score,
)

SWEEP = sorted(p for p in (Path(__file__).parent / 'fixtures' / 'molmo_sweep').glob('*/*') if p.is_dir())

# Read from the recording: ten tasks over twenty trials, split irregularly. Four tasks ran three times,
# two ran twice, and four ran once.
PER_TASK = {
    'pick up the bottle.': (2, 0),
    'pick up the cup.': (1, 1),
    'pick up the kettle.': (2, 0),
    'pick up the ladle.': (3, 1),
    'pick up the mug.': (1, 0),
    'pick up the pot.': (3, 0),
    'pick up the remote.': (1, 0),
    'pick up the spatula.': (1, 1),
    'pick up the spoon.': (3, 1),
    'pick up the tissue.': (3, 1),
}

# Written into a synthetic episode and read back out of its `Outcome`, so the assertion turns on the two
# being the same string.
CUP = 'pick up the cup.'
KETTLE = 'pick up the kettle.'

# What the harness records for a trial that ran out of time, on a sim and on a rig alike.
TIMED_OUT = {TASK: CUP, TERMINATED: False}
# A rig's console adds its operator's verdict and item count to the same record.
RIG_TIMED_OUT = {
    **TIMED_OUT,
    verdicts.OUTCOME: verdicts.Outcome.OUT_OF_TIME,
    verdicts.SUCCESSFUL_ITEMS: 0,
    verdicts.TOTAL_ITEMS: 1,
}


def score_molmo(episodes: list[Path]) -> Scores:
    return score(MOLMO_SCORER, episodes)


def _episodes(tmp_path: Path, *statics: dict[str, object] | str | bytes | None) -> list[Path]:
    """One episode directory per given `static.json`: a mapping is written as JSON, a string or bytes as they
    are, and None leaves the episode without one."""
    made = []
    for index, static in enumerate(statics):
        episode = tmp_path / f'{index:012d}'
        episode.mkdir()
        if isinstance(static, bytes):
            (episode / STATIC_FILE).write_bytes(static)
        elif static is not None:
            (episode / STATIC_FILE).write_text(static if isinstance(static, str) else json.dumps(static))
        made.append(episode)
    return made


def test_the_sweep_scores_five_of_twenty_in_this_record():
    assert score_molmo(SWEEP).model_dump(mode='json') == {'primary': 0.25, 'episodes': 20, 'unscored': 0}


def test_each_episode_of_the_sweep_gives_the_task_and_success_it_recorded():
    outcomes = [molmo_outcome(episode) for episode in SWEEP]
    scored = [outcome for outcome in outcomes if outcome is not None]
    trials = Counter(outcome.task for outcome in scored)
    successes = Counter(outcome.task for outcome in scored if outcome.succeeded)
    assert len(scored) == len(outcomes)
    assert {task: (trials[task], successes[task]) for task in trials} == PER_TASK


def test_an_episode_that_recorded_no_outcome_is_unscored_rather_than_failed(tmp_path: Path):
    episodes = _episodes(tmp_path, {TASK: CUP, SUCCESS: True}, {TASK: CUP}, None)
    assert [molmo_outcome(episode) for episode in episodes] == [Outcome(CUP, True), None, None]
    scores = score_molmo(episodes)
    assert (scores.episodes, scores.unscored) == (3, 2)
    assert scores.primary == pytest.approx(1.0)


def test_a_sweep_that_recorded_nothing_scores_zero_rather_than_dividing_by_it(tmp_path: Path):
    scores = score_molmo(_episodes(tmp_path, None))
    assert (scores.episodes, scores.unscored, scores.primary) == (1, 1, 0.0)


def test_a_failed_episode_is_scored_as_a_trial_rather_than_as_nothing(tmp_path: Path):
    (episode,) = _episodes(tmp_path, {TASK: KETTLE, SUCCESS: False})
    assert molmo_outcome(episode) == Outcome(KETTLE, False)
    scores = score_molmo([episode])
    assert (scores.episodes, scores.unscored, scores.primary) == (1, 0, 0.0)


def test_a_sim_and_a_rig_trial_that_ran_out_of_time_both_count_as_a_failed_trial(tmp_path: Path):
    sim, rig = _episodes(tmp_path, TIMED_OUT, RIG_TIMED_OUT)
    assert molmo_outcome(sim) == molmo_outcome(rig) == Outcome(CUP, False)
    scores = score_molmo([sim, rig])
    assert (scores.episodes, scores.unscored, scores.primary) == (2, 0, 0.0)


def test_a_success_recorded_beside_a_timeout_does_not_count():
    assert recorded_task_and_success({**TIMED_OUT, SUCCESS: True}, TASK) == (CUP, False)


@pytest.mark.parametrize('static', ['{ truncated upload', '[true, "pick up the cup."]', b'{"task": "\xff"}'])
def test_statics_that_are_not_a_json_object_are_unscored_rather_than_fatal(tmp_path: Path, static: str | bytes):
    (episode,) = _episodes(tmp_path, static)
    assert read_static(episode) is None
    assert score_molmo([episode]).unscored == 1


@pytest.mark.parametrize(
    ('static', 'misrecorded'),
    [
        ({SUCCESS: 1, TASK: CUP}, SUCCESS),
        ({SUCCESS: True, TASK: 3}, TASK),
        ({SUCCESS: None, TASK: CUP}, SUCCESS),
        ({SUCCESS: True, TASK: None}, TASK),
    ],
)
def test_a_success_or_task_of_the_wrong_type_is_no_outcome_and_is_logged(
    static: dict[str, object], misrecorded: str, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger='platform_client.scoring'):
        assert recorded_task_and_success(static, TASK) is None
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert misrecorded in caplog.records[0].getMessage()


@pytest.mark.parametrize('static', [{TASK: CUP}, {SUCCESS: True}, {TASK: CUP, TERMINATED: True}, {TERMINATED: False}])
def test_a_missing_success_or_task_is_no_outcome_and_logs_nothing(
    static: dict[str, object], caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.DEBUG, logger='platform_client.scoring'):
        assert recorded_task_and_success(static, TASK) is None
    assert caplog.records == []


@pytest.mark.parametrize(('static', 'expected'), [({TASK: CUP, SUCCESS: True}, (CUP, True)), ({TASK: CUP}, None)])
def test_a_wrong_typed_timeout_marker_is_logged_and_read_as_absent(
    static: dict[str, object], expected: tuple[str, bool] | None, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger='platform_client.scoring'):
        assert recorded_task_and_success({**static, TERMINATED: 'false'}, TASK) == expected
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert TERMINATED in caplog.records[0].getMessage()


def _graded(episode: Path) -> Outcome | None:
    """A caller's scorer: a success scores 1.0, and a failure scores the `grade` its statics record."""
    static = read_static(episode)
    recorded = recorded_task_and_success(static, TASK) if static is not None else None
    if static is None or recorded is None:
        return None
    task, succeeded = recorded
    if succeeded:
        return Outcome(task, True, 1.0)
    grade = static.get('grade')
    return Outcome(task, False, grade) if isinstance(grade, float) else None


def test_a_caller_scores_with_its_own_scorer_beside_the_public_ones(tmp_path: Path):
    graded = ScorerRef('graded')
    episodes = _episodes(
        tmp_path, {TASK: CUP, SUCCESS: True}, {TASK: CUP, SUCCESS: False, 'grade': 0.5}, {TASK: KETTLE, SUCCESS: False}
    )

    scores = score(graded, episodes, {**PUBLIC_SCORERS, graded: _graded})

    assert [_graded(episode) for episode in episodes] == [Outcome(CUP, True, 1.0), Outcome(CUP, False, 0.5), None]
    assert (scores.episodes, scores.unscored) == (3, 1)
    assert scores.primary == pytest.approx(0.75)


def test_primary_takes_the_success_of_an_ungraded_episode_beside_a_graded_one(tmp_path: Path):
    def grades_only_the_cup(episode: Path) -> Outcome | None:
        outcome = PUBLIC_SCORERS[MOLMO_SCORER](episode)
        if outcome is None or outcome.task != CUP:
            return outcome
        return Outcome(outcome.task, outcome.succeeded, 0.25)

    episodes = _episodes(tmp_path, {TASK: CUP, SUCCESS: False}, {TASK: KETTLE, SUCCESS: True})
    scores = score(ScorerRef('mixed'), episodes, {ScorerRef('mixed'): grades_only_the_cup})

    assert [grades_only_the_cup(episode) for episode in episodes] == [Outcome(CUP, False, 0.25), Outcome(KETTLE, True)]
    assert scores.primary == pytest.approx((0.25 + 1.0) / 2)


def test_a_scorer_the_table_does_not_hold_is_refused():
    with pytest.raises(LookupError, match="no scorer 'held_out'; the scorers are molmo"):
        score(ScorerRef('held_out'), SWEEP)


@pytest.mark.parametrize('name', PUBLIC_EVALS)
def test_every_public_eval_names_a_public_scorer(name: EvalRef):
    assert public_eval(name).scorer in PUBLIC_SCORERS


def test_the_published_scores_are_a_subset_of_the_record():
    record = score_molmo(SWEEP).model_dump(mode='json')
    assert set(PublishedScores.model_fields) < set(Scores.model_fields)
    published = PublishedScores.model_validate(record).model_dump(mode='json')
    assert published == {name: record[name] for name in PublishedScores.model_fields}
