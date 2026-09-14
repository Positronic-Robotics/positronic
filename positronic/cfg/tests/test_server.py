"""The rollouts preset: what it derives, and what the page does with each cell it writes."""

from positronic.cfg import server as cfg_server
from positronic.dataset.episode import EpisodeContainer
from positronic.server.positronic_server import filter_spelling


def _with_stages(*codes):
    """An episode carrying the rungs a rollout recorded, as `progress.state` writes them."""
    return EpisodeContainer({cfg_server.PROGRESS_STATE: [(code, 0) for code in codes]})


def test_the_stage_cell_sorts_by_the_ladder_and_not_by_the_label():
    """The page sorts the cell it shows, so a bare label would order the column alphabetically.

    `at the target` is the top rung and sorts FIRST alphabetically, so the two orders disagree at the
    one place an operator reads: a round's best episodes would sit at the bottom of an ascending sort.
    """
    cells = [cfg_server.highest_rollout_stage(_with_stages(stage.value)) for stage in cfg_server.ProgressStage]
    ranks = [cell.rank for cell in cells]
    labels = [cell.label for cell in cells]

    assert ranks == sorted(ranks)
    assert labels != sorted(labels)


def test_the_stage_cell_shows_the_operators_word_for_the_rung():
    top = cfg_server.ProgressStage.AT_TARGET

    assert cfg_server.highest_rollout_stage(_with_stages(top.value)) == cfg_server.StageCell(top.rank, top.label)


def test_the_highest_rung_reached_is_the_one_shown():
    """A rollout records every rung it passes, so the cell is the maximum and not the last written."""
    cell = cfg_server.highest_rollout_stage(_with_stages('at-target', 'reaching', 'contact'))

    assert cell.label == cfg_server.ProgressStage.AT_TARGET.label


def test_an_episode_that_recorded_no_progress_still_sorts_against_the_ladder():
    """A string cell among numeric ones is not an ordering: the page compares whatever each cell holds,
    so a no-progress episode could sit above a higher rung from one end of the sort."""
    assert cfg_server.highest_rollout_stage(EpisodeContainer({})) == cfg_server.NO_STAGE
    assert cfg_server.NO_STAGE.rank < min(stage.rank for stage in cfg_server.ProgressStage)
    assert isinstance(cfg_server.NO_STAGE.rank, int)


def test_a_rung_the_ladder_does_not_carry_is_not_a_stage():
    """The codes are the platform console's, and it may write one this ladder predates."""
    assert cfg_server.highest_rollout_stage(_with_stages('teleporting')) == cfg_server.NO_STAGE


def test_every_column_of_the_episode_table_is_a_key_the_preset_derives():
    """The producer and the consumers are 40 lines apart, so this is what holds them equal.

    `Derive` writes the keys, the table addresses them again, and a name that drifts in one place
    renders an empty column rather than failing.
    """
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
    """Why the Stage column offers no filter: the filter path spells a static with `str`, and a named
    pair spells itself as its own repr, class name and field names included. This is what the rank the
    sort needs costs, recorded here rather than discovered by whoever adds the dropdown back."""
    top = cfg_server.ProgressStage.AT_TARGET
    cell = cfg_server.highest_rollout_stage(_with_stages(top.value))

    assert filter_spelling(cell) == f'StageCell(rank={top.rank}, label={top.label!r})'
    assert cfg_server.rollouts_episodes_table.instantiate()[cfg_server.DERIVED_STAGE].filter is False


def test_the_model_table_counts_a_target_reached_off_the_stage_cell():
    """The rank lives in the cell and nowhere else, so this reads it there rather than from a second
    derived field that would have to be kept equal to it."""
    group = cfg_server.rollouts_by_model.instantiate()
    top = cfg_server.ProgressStage.AT_TARGET
    at_target = EpisodeContainer({
        cfg_server.DERIVED_MODEL: 'pi05',
        cfg_server.DERIVED_OUTCOME: cfg_server.SUCCESS,
        cfg_server.DERIVED_STAGE: cfg_server.StageCell(top.rank, top.label),
    })
    short = EpisodeContainer({
        cfg_server.DERIVED_MODEL: 'pi05',
        cfg_server.DERIVED_OUTCOME: cfg_server.RAN_OUT_OF_TIME,
        cfg_server.DERIVED_STAGE: cfg_server.NO_STAGE,
    })

    row = group.group_fn([at_target, short])

    assert row['at_target'] == 1
    assert row['successes'] == 1
