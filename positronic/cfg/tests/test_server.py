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
    cells = [cfg_server.rollout_stage_cell(_with_stages(stage.value)) for stage in cfg_server.ProgressStage]
    assert all(cell is not None for cell in cells)
    ranks = [cell[0] for cell in cells if cell is not None]
    labels = [cell[1] for cell in cells if cell is not None]

    assert ranks == sorted(ranks)
    assert labels != sorted(labels)


def test_the_stage_cell_shows_the_operators_word_for_the_rung():
    assert cfg_server.rollout_stage_cell(_with_stages('at-target')) == (4, 'at the target')


def test_the_highest_rung_reached_is_the_one_shown():
    """A rollout records every rung it passes, so the cell is the maximum and not the last written."""
    cell = cfg_server.rollout_stage_cell(_with_stages('at-target', 'reaching', 'contact'))

    assert cell is not None and cell[1] == 'at the target'


def test_an_episode_that_recorded_no_progress_has_no_stage():
    """None, so the column's own default renders rather than a made-up rung."""
    assert cfg_server.rollout_stage_cell(EpisodeContainer({})) is None
    assert cfg_server.rollout_stage_rank(EpisodeContainer({})) is None


def test_a_rung_the_ladder_does_not_carry_is_not_a_stage():
    """The codes are the platform console's, and it may write one this ladder predates."""
    assert cfg_server.rollout_stage_cell(_with_stages('teleporting')) is None


def test_every_column_of_the_episode_table_is_a_key_the_preset_derives():
    """The producer and the consumers are 40 lines apart, so this is what holds them equal.

    `Derive` writes the keys, the table addresses them again, and a name that drifts in one place
    renders an empty column rather than failing.
    """
    derived = {
        cfg_server.DERIVED_MODEL,
        cfg_server.DERIVED_OUTCOME,
        cfg_server.DERIVED_STAGE,
        cfg_server.DERIVED_STAGE_RANK,
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
    """Why the Stage column offers no filter: the filter path spells a static with `str`, so the pair
    would be offered as its own repr. A label-only cell filters cleanly, and this is the cost of the
    rank the sort needs."""
    cell = cfg_server.rollout_stage_cell(_with_stages('at-target'))

    assert filter_spelling(cell) == "(4, 'at the target')"
    assert cfg_server.rollouts_episodes_table.instantiate()[cfg_server.DERIVED_STAGE].filter is False
