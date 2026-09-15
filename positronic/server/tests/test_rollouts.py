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


def test_every_word_the_vocabulary_carries_has_a_badge():
    """A word with no option draws as itself on a neutral badge, which reads as unscored to an
    operator scanning the column."""
    assert set(rollouts.OUTCOME_BADGE.options) == set(Outcome)


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
