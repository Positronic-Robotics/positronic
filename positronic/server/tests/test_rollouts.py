"""What the rollouts table draws: a colour for every word, and a rung the page can sort."""

import re
from pathlib import Path

from eval_vocabulary.outcome import Outcome
from eval_vocabulary.progress import LADDER, STAGE_LABELS, Stage

from positronic.server import rollouts
from positronic.server.positronic_server import ASSET_ROUTE, _pkg_path


def _variants_app_js_accepts() -> set[str]:
    """The variant names the page's own badge renderer allows; anything else draws as `default`."""
    source = (Path(_pkg_path(ASSET_ROUTE)) / 'app.js').read_text()
    listed = re.search(r'const VARIANTS = \[([^\]]*)\]', source)
    assert listed, 'app.js no longer declares the badge variants in one list'
    return set(re.findall(r"'([^']+)'", listed.group(1)))


def test_every_word_this_release_knows_has_a_colour_chosen_for_it():
    """The fallback below keeps a newer vocabulary rendering, and would also swallow a word added
    here with no colour picked for it. This is what catches that."""
    assert set(rollouts.OUTCOME_VARIANT) == set(Outcome)
    assert set(rollouts.OUTCOME_BADGE.options) == set(Outcome)


def test_a_word_added_after_this_release_draws_neutral(monkeypatch):
    """The vocabulary floats and is append-only, so an install can hold a word this module has no
    colour for — the state this makes, by taking one away. Reading it must not raise at import."""
    monkeypatch.delitem(rollouts.OUTCOME_VARIANT, Outcome.DISCARDED)

    assert rollouts.outcome_variant(Outcome.DISCARDED) == rollouts.NEUTRAL


def test_every_colour_is_one_the_page_accepts():
    assert set(rollouts.OUTCOME_VARIANT.values()) <= _variants_app_js_accepts()


def test_the_only_word_shown_differently_is_the_one_a_console_shouts():
    shown = {outcome: rollouts.outcome_label(outcome) for outcome in Outcome}

    assert {word for word, label in shown.items() if label != word.value} == {Outcome.UNSCORED}
    assert shown[Outcome.UNSCORED] == 'Unscored'


def test_a_rung_carries_the_operators_word_and_its_place_on_the_ladder():
    top = Stage.AT_TARGET

    assert rollouts.stage_cell([top]) == (LADDER.index(top), STAGE_LABELS[top])


def test_an_episode_that_marked_nothing_sorts_under_every_rung():
    assert rollouts.stage_cell([]) == rollouts.NO_STAGE
    assert rollouts.NO_STAGE.rank < 0


def test_a_rate_carries_the_number_it_sorts_on_and_the_text_it_shows():
    assert rollouts.rate_cell(1, 4) == (25.0, '25%')


def test_a_model_nobody_scored_sorts_under_every_rate():
    """A bare string among numbers is not an ordering, and this column is the table's default sort."""
    assert rollouts.rate_cell(0, 0) == rollouts.NO_RATE
    assert rollouts.NO_RATE.rate < 0
    assert isinstance(rollouts.NO_RATE.rate, float)
