"""The outcome an operator gives an episode, recorded as `eval.outcome`.

A console writes these values and a viewer, a report and a coordinator read them back. An episode on
disk carries the value it was scored with, so a member is append-only and a value never changes.
"""

from __future__ import annotations

from enum import StrEnum

# The statics an attended episode carries beside what the harness itself writes. The verdict, and
# the items the operator counted.
OUTCOME = 'eval.outcome'
SUCCESSFUL_ITEMS = 'eval.successful_items'
TOTAL_ITEMS = 'eval.total_items'


class Outcome(StrEnum):
    """How an episode ended, as `eval.outcome` records it."""

    SUCCESS = 'Success'
    FAIL = 'Fail'
    # What an episode the harness ended on its budget carries.
    OUT_OF_TIME = 'Ran out of time'
    SAFETY = 'Safety'
    SYSTEM = 'System'
    # What an episode nobody scored carries. A reader reads an absent outcome as this too.
    UNSCORED = 'UNSCORED'
    # What the operator's Discard writes: the attempt is not scored at all.
    DISCARDED = 'Discarded'

    # A member reads as its wire value inside a container's repr, not as its enum name.
    __repr__ = str.__repr__


# The verdicts the operator gives, in the order the console's radio lists them.
VERDICTS: tuple[Outcome, ...] = (Outcome.SUCCESS, Outcome.FAIL, Outcome.OUT_OF_TIME, Outcome.SAFETY, Outcome.SYSTEM)

# What an episode carrying no `eval.outcome` at all means. One name, so a reader that defaults and a
# console that seeds the field cannot disagree about it.
ABSENT = Outcome.UNSCORED


def is_scored(value: str | None) -> bool:
    """True only for a verdict. An absent outcome, `UNSCORED` and `DISCARDED` are not scored."""
    return value in VERDICTS
