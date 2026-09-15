"""The rollouts preset: what it derives, and what the page does with each cell it writes."""

from eval_vocabulary.outcome import OUTCOME, Outcome
from eval_vocabulary.progress import LADDER, STAGE_LABELS, STATE_SIGNAL, Stage

from positronic.cfg import server as cfg_server
from positronic.dataset.episode import EpisodeContainer
from positronic.server.positronic_server import filter_spelling
from positronic.server.rollouts import NO_RATE, NO_STAGE, StageCell


def _with_stages(*codes):
    """An episode carrying the rungs a rollout recorded, as `progress.state` writes them."""
    return EpisodeContainer({STATE_SIGNAL: [(code, 0) for code in codes]})


def _row(outcome, stage=NO_STAGE, model='pi05'):
    """One episode as the model table reads it: the three fields the preset derived onto it."""
    return EpisodeContainer({
        cfg_server.DERIVED_MODEL: model,
        cfg_server.DERIVED_OUTCOME: outcome,
        cfg_server.DERIVED_STAGE: stage,
    })


def test_the_stage_cell_sorts_by_the_ladder_and_not_by_the_label():
    """The two orders disagree at the one place an operator reads: a round's best episodes would sit
    at the bottom of an ascending sort."""
    cells = [cfg_server.highest_rollout_stage(_with_stages(stage.value)) for stage in LADDER]
    ranks = [cell.rank for cell in cells]
    labels = [cell.label for cell in cells]

    assert ranks == sorted(ranks)
    assert labels != sorted(labels)


def test_the_stage_cell_shows_the_operators_word_for_the_rung():
    top = Stage.AT_TARGET

    assert cfg_server.highest_rollout_stage(_with_stages(top)) == StageCell(LADDER.index(top), STAGE_LABELS[top])


def test_the_highest_rung_reached_is_the_one_shown():
    """A rollout records every rung it passes, so the cell is the maximum and not the last written."""
    cell = cfg_server.highest_rollout_stage(_with_stages('at-target', 'reaching', 'contact'))

    assert cell.label == STAGE_LABELS[Stage.AT_TARGET]


def test_an_episode_that_recorded_no_progress_still_sorts_against_the_ladder():
    """A string cell among numeric ones is not an ordering: the page compares whatever each cell holds,
    so a no-progress episode could sit above a higher rung from one end of the sort."""
    assert cfg_server.highest_rollout_stage(EpisodeContainer({})) == NO_STAGE
    assert NO_STAGE.rank < min(LADDER.index(stage) for stage in LADDER)
    assert isinstance(NO_STAGE.rank, int)


def test_a_rung_the_ladder_does_not_carry_is_not_a_stage():
    """The codes are the platform console's, and it may write one this ladder predates."""
    assert cfg_server.highest_rollout_stage(_with_stages('teleporting')) == NO_STAGE


def test_every_column_of_the_episode_table_is_a_key_the_preset_derives():
    """A derived name that drifts renders an empty column rather than failing, and this catches it."""
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
    """Why the Stage column offers no filter, recorded here rather than discovered by whoever adds
    the dropdown back."""
    top = Stage.AT_TARGET
    cell = cfg_server.highest_rollout_stage(_with_stages(top))

    assert filter_spelling(cell) == f'StageCell(rank={LADDER.index(top)}, label={STAGE_LABELS[top]!r})'
    assert cfg_server.rollouts_episodes_table.instantiate()[cfg_server.DERIVED_STAGE].filter is False


def test_the_model_table_counts_a_target_reached_off_the_stage_cell():
    """The rank lives in the cell and nowhere else, so this reads it there rather than from a second
    derived field that would have to be kept equal to it."""
    group = cfg_server.rollouts_by_model.instantiate()
    top = Stage.AT_TARGET
    at_target = _row(Outcome.SUCCESS, StageCell(LADDER.index(top), STAGE_LABELS[top]))
    short = _row(Outcome.OUT_OF_TIME, NO_STAGE)

    row = group.group_fn([at_target, short])

    assert row['at_target'] == 1
    assert row['successes'] == 1


def test_an_episode_nobody_scored_reads_as_unscored():
    """A console seeds the field with `UNSCORED` and an unattended end leaves it out entirely. Both
    are the same state, so both reach the page as the one word the badge carries."""
    assert cfg_server.rollout_outcome(EpisodeContainer({})) == Outcome.UNSCORED
    assert cfg_server.rollout_outcome(EpisodeContainer({OUTCOME: Outcome.UNSCORED})) == Outcome.UNSCORED


def test_a_word_this_vocabulary_does_not_carry_reaches_the_page_as_itself():
    """Constructing the enum here would take the whole table down over one such recording."""
    assert cfg_server.rollout_outcome(EpisodeContainer({OUTCOME: 'Rescored by hand'})) == 'Rescored by hand'


def test_an_episode_the_operator_discarded_is_listed_and_left_out_of_the_rate():
    """A discarded attempt measured nothing, so counting it as a failure understates the endpoint.
    The report drops it from the same round, and these two must not disagree about one number."""
    group = cfg_server.rollouts_by_model.instantiate()

    row = group.group_fn([_row(Outcome.SUCCESS), _row(Outcome.DISCARDED), _row(Outcome.UNSCORED)])

    assert row['count'] == 3
    assert row['scored'] == 1
    assert row['successes'] == 1
    assert row['success_rate'].rate == 100


def test_a_model_nobody_scored_has_no_rate_rather_than_a_zero():
    """Zero would read as an endpoint that failed every attempt, which is the opposite of unscored."""
    group = cfg_server.rollouts_by_model.instantiate()

    row = group.group_fn([_row(Outcome.UNSCORED), _row(Outcome.DISCARDED)])

    assert row['success_rate'] == NO_RATE
    # The cell carries its own text, so the column formats nothing and never falls back to a string.
    assert group.format_table['success_rate'].format is None
    assert group.format_table['success_rate'].default is None
