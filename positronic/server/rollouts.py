"""How the rollouts table draws what an attended episode recorded.

`eval_vocabulary` owns the words; this owns their colour, the word the page shows, and the sortable
cell a rung needs.
"""

from collections.abc import Iterable
from typing import NamedTuple

from eval_vocabulary.outcome import Outcome
from eval_vocabulary.progress import LADDER, STAGE_LABELS, highest_stage

from positronic.server.positronic_server import RendererConfig

# What a word this release has no colour for draws as. The vocabulary is append-only and floats,
# so an install can hold a word added after this module was written.
NEUTRAL = 'default'

# The badge colour each verdict carries. `app.js` accepts these four names and nothing else.
OUTCOME_VARIANT: dict[Outcome, str] = {
    Outcome.SUCCESS: 'success',
    Outcome.FAIL: 'danger',
    Outcome.SAFETY: 'warning',
    Outcome.SYSTEM: 'warning',
    Outcome.OUT_OF_TIME: 'default',
    Outcome.UNSCORED: 'default',
    Outcome.DISCARDED: 'default',
}

# The one word the page spells differently from the wire: a console seeds the field with `UNSCORED`,
# and a table does not shout.
LABEL_OVERRIDES: dict[Outcome, str] = {Outcome.UNSCORED: 'Unscored'}


def outcome_label(outcome: Outcome) -> str:
    return LABEL_OVERRIDES.get(outcome, outcome.value)


def outcome_variant(outcome: Outcome) -> str:
    return OUTCOME_VARIANT.get(outcome, NEUTRAL)


# A word this vocabulary does not carry is not listed, and `app.js` then draws it as itself on a
# neutral badge — a recording from a console one word ahead still reads.
OUTCOME_BADGE = RendererConfig(
    type='badge',
    options={outcome: {'label': outcome_label(outcome), 'variant': outcome_variant(outcome)} for outcome in Outcome},
)


class RateCell(NamedTuple):
    """A percentage as the page reads a cell: it sorts on the first item and shows the second."""

    rate: float
    shown: str


# The cell for a model nobody scored. Its rate is below every real one, and a bare string here would
# leave the column with no ordering: the page compares whatever each cell holds.
NO_RATE = RateCell(-1.0, '-')


def rate_cell(successes: int, scored: int) -> RateCell:
    """The success rate over the episodes somebody scored, or `NO_RATE` where nobody did."""
    if not scored:
        return NO_RATE
    rate = 100 * successes / scored
    return RateCell(rate, f'{rate:.0f}%')


class StageCell(NamedTuple):
    """A rung as the page reads a cell: it sorts on the first item and shows the second.

    A NamedTuple IS a tuple, so this serializes to the `[raw, formatted]` pair `app.js` documents
    while every reader in this process addresses `rank` and `label` by name.
    """

    rank: int
    label: str


# The cell for an episode that recorded no progress. Its rank is below every rung, so it sorts under
# them from either end; a rung's rank is its index in `LADDER` and counts from 0.
NO_STAGE = StageCell(-1, '-')


def stage_cell(codes: Iterable[str]) -> StageCell:
    """The highest rung `codes` reached, as a cell.

    FOOTGUN: a bare label sorts alphabetically, which is not the ladder — `at the target` would lead
    and `reaching` would trail. Every episode gets a pair, `NO_STAGE` included, because the page
    compares whatever the cell holds and a string against these numbers is not an ordering.
    """
    stage = highest_stage(codes)
    return NO_STAGE if stage is None else StageCell(LADDER.index(stage), STAGE_LABELS[stage])
