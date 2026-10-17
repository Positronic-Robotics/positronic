"""Server presets: derived episode values and table cells."""

import pytest
from eval_vocabulary.outcome import OUTCOME, Outcome, counted_outcome
from eval_vocabulary.progress import LADDER, STAGE_LABELS, STATE_SIGNAL, Stage

from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.cfg import server as cfg_server
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.eval import keys as eval_keys
from positronic.policy import PolicyEnd
from positronic.server.positronic_server import filter_spelling
from positronic.server.rollouts import NO_RATE, NO_STAGE, StageCell


@pytest.mark.parametrize(
    'timelines',
    [
        (HARNESS_WORLD,),
        (RECEIVED_WORLD,),
        (RECORDED_TIME,),
        (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME),
        (RECEIVED_WORLD, RECORDED_TIME),
    ],
)
def test_finetuning_rates_use_the_recording_clock(timelines):
    episode = EpisodeContainer({
        keys.GRIP: DummySignal(
            [[0] * len(timelines), [60_000_000_000 * (i + 1) for i in range(len(timelines))]],
            [0, 1],
            timelines=timelines,
        ),
        keys.TASK: 'pick',
        'units': 2,
    })
    assert cfg_server.uph(episode) == pytest.approx(120)
    row = cfg_server.finetune_group_by_task().group_fn([episode, episode])
    assert row['duration'] == pytest.approx(2 / 60)
    assert row['uph'] == pytest.approx(120)
    assert row['count'] == 2


def _with_stages(*codes):
    """An episode carrying the rungs a rollout recorded, as `progress.state` writes them."""
    return EpisodeContainer({STATE_SIGNAL: [(code, 0) for code in codes]})


def _row(outcome, stage=NO_STAGE, model='pi05', statics=None):
    """One episode as the model table reads it: ``outcome`` in ``eval.outcome``, ``statics`` beside it, and the three
    fields the preset derived onto it."""
    recorded = dict(statics or {})
    if outcome is not None:
        recorded[OUTCOME] = outcome
    return EpisodeContainer({
        **recorded,
        cfg_server.DERIVED_MODEL: model,
        cfg_server.DERIVED_OUTCOME: counted_outcome(recorded),
        cfg_server.DERIVED_STAGE: stage,
    })


def test_the_stage_cell_sorts_by_the_ladder_and_not_by_the_label():
    cells = [cfg_server.highest_rollout_stage(_with_stages(stage.value)) for stage in LADDER]
    ranks = [cell.rank for cell in cells]
    labels = [cell.label for cell in cells]

    assert ranks == sorted(ranks)
    assert labels != sorted(labels)


def test_the_stage_cell_shows_the_operators_word_for_the_rung():
    top = Stage.AT_TARGET

    assert cfg_server.highest_rollout_stage(_with_stages(top)) == StageCell(LADDER.index(top), STAGE_LABELS[top])


def test_the_highest_rung_reached_is_the_one_shown():
    """A rollout records every rung it passes, and the cell shows the highest."""
    cell = cfg_server.highest_rollout_stage(_with_stages('at-target', 'reaching', 'contact'))

    assert cell.label == STAGE_LABELS[Stage.AT_TARGET]


def test_an_episode_that_recorded_no_progress_still_sorts_against_the_ladder():
    """A string cell among numeric ones is not an ordering: the page compares whatever each cell holds,
    so a no-progress episode could sit above a higher rung from one end of the sort."""
    assert cfg_server.highest_rollout_stage(EpisodeContainer({})) == NO_STAGE
    assert NO_STAGE.rank < min(LADDER.index(stage) for stage in LADDER)
    assert isinstance(NO_STAGE.rank, int)


def test_a_rung_the_ladder_does_not_carry_is_not_a_stage():
    """A recording made against a longer ladder carries codes that this ladder does not."""
    assert cfg_server.highest_rollout_stage(_with_stages('teleporting')) == NO_STAGE


def test_every_column_of_the_episode_table_is_a_key_the_preset_derives():
    """A column keyed by a name that the preset does not derive shows empty and raises nothing."""
    derived = {
        cfg_server.DERIVED_MODEL,
        cfg_server.DERIVED_OUTCOME,
        cfg_server.DERIVED_STAGE,
        cfg_server.DERIVED_ITEMS,
        cfg_server.DERIVED_STARTED,
    }
    built_in = {'__index__', '__duration__'}
    recorded = {'task'}

    columns = set(cfg_server.rollouts_episodes_table.instantiate())

    assert columns - built_in - recorded <= derived


def test_the_model_group_table_groups_on_the_key_the_preset_derives():
    group = cfg_server.rollouts_by_model.instantiate()

    assert group.group_keys == cfg_server.DERIVED_MODEL
    assert set(group.format_table) >= {cfg_server.DERIVED_MODEL}


def test_a_stage_cell_would_spell_itself_into_a_filter_dropdown():
    """`filter_spelling` is `str(value)`, which spells a `StageCell` as its repr — so the Stage
    column offers no filter."""
    top = Stage.AT_TARGET
    cell = cfg_server.highest_rollout_stage(_with_stages(top))

    assert filter_spelling(cell) == f'StageCell(rank={LADDER.index(top)}, label={STAGE_LABELS[top]!r})'
    assert cfg_server.rollouts_episodes_table.instantiate()[cfg_server.DERIVED_STAGE].filter is False


def test_the_model_table_counts_a_target_reached_off_the_stage_cell():
    group = cfg_server.rollouts_by_model.instantiate()
    top = Stage.AT_TARGET
    at_target = _row(Outcome.SUCCESS, StageCell(LADDER.index(top), STAGE_LABELS[top]))
    short = _row(Outcome.OUT_OF_TIME, NO_STAGE)

    row = group.group_fn([at_target, short])

    assert row['at_target'] == 1
    assert row['successes'] == 1


def test_a_rung_above_the_target_still_counts_as_reaching_it():
    group = cfg_server.rollouts_by_model.instantiate()
    beyond = LADDER.index(Stage.AT_TARGET) + 1

    row = group.group_fn([_row(Outcome.SUCCESS, StageCell(beyond, 'put it down again'))])

    assert row['at_target'] == 1


def test_a_rung_below_the_target_does_not_count_as_reaching_it():
    group = cfg_server.rollouts_by_model.instantiate()
    below = Stage.CONTROL

    row = group.group_fn([_row(Outcome.FAIL, StageCell(LADDER.index(below), STAGE_LABELS[below]))])

    assert row['at_target'] == 0


def test_the_eval_table_shows_every_fact_that_the_count_reads_and_an_absent_fact_as_blank():
    """The plain table cannot derive the counted outcome, so it shows each fact that `counted_outcome` reads."""
    table = cfg_server.eval_table.instantiate()
    facts = (
        eval_keys.SUCCESS,
        OUTCOME,
        eval_keys.ENDED_BY,
        eval_keys.POLICY_END,
        eval_keys.ENV_ENDS_ON_SUCCESS,
        eval_keys.TERMINATED,
    )

    assert set(facts) <= set(table)
    assert [table[key].default for key in facts] == [None] * len(facts)


def test_an_episode_the_operator_discarded_is_listed_and_left_out_of_the_rate():
    """A discarded attempt measured nothing, so counting it as a failure understates the endpoint."""
    group = cfg_server.rollouts_by_model.instantiate()

    row = group.group_fn([_row(Outcome.SUCCESS), _row(Outcome.DISCARDED), _row(Outcome.UNSCORED)])

    assert row['count'] == 3
    assert row['scored'] == 1
    assert row['successes'] == 1
    assert row['success_rate'].rate == 100


def test_the_outcome_column_and_the_rate_both_count_a_policy_end_with_no_verdict_by_the_rule():
    group = cfg_server.rollouts_by_model.instantiate()
    ended = {eval_keys.ENDED_BY: eval_keys.ENDED_BY_POLICY, eval_keys.TERMINATED: True}
    gave_up = _row(None, statics={**ended, eval_keys.POLICY_END: PolicyEnd.GIVE_UP})
    completed = _row(None, statics={**ended, eval_keys.POLICY_END: PolicyEnd.COMPLETE})

    row = group.group_fn([_row(Outcome.SUCCESS), gave_up, completed])

    assert (gave_up[cfg_server.DERIVED_OUTCOME], completed[cfg_server.DERIVED_OUTCOME]) == (
        Outcome.FAIL,
        Outcome.UNSCORED,
    )
    assert (row['count'], row['scored'], row['successes']) == (3, 2, 1)
    assert row['success_rate'].rate == 50


def test_a_model_nobody_scored_has_no_rate_rather_than_a_zero():
    """Zero would read as an endpoint that failed every attempt, which is the opposite of unscored."""
    group = cfg_server.rollouts_by_model.instantiate()

    row = group.group_fn([_row(Outcome.UNSCORED), _row(Outcome.DISCARDED)])

    assert row['success_rate'] == NO_RATE
    # The cell carries its own text, so the column formats nothing and never falls back to a string.
    assert group.format_table['success_rate'].format is None
    assert group.format_table['success_rate'].default is None
