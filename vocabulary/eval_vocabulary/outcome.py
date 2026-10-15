"""The outcome an operator gives an episode, recorded as `eval.outcome`, and the outcome a reader counts.

An episode on disk carries the value it was scored with, so a member is append-only and a value
never changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from eval_vocabulary.episode import (
    ENDED_BY,
    ENDED_BY_POLICY,
    ENV_ENDS_ON_SUCCESS,
    POLICY_END,
    SUCCESS,
    TERMINATED,
    PolicyEnd,
)

# The statics that the operator gives an attended episode: the verdict and the item counts. The
# harness writes the rest.
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


def counted_outcome(static: Mapping[str, object]) -> Outcome:
    """The outcome that every reader counts for the episode with the statics `static`."""
    verdict = static.get(OUTCOME)
    if verdict in VERDICTS or verdict == Outcome.DISCARDED:
        return Outcome(verdict)
    if static.get(TERMINATED) is False:
        return Outcome.OUT_OF_TIME
    success = static.get(SUCCESS)
    if isinstance(success, bool):
        return Outcome.SUCCESS if success else Outcome.FAIL
    # A policy end that counts as nothing lets a policy keep a failing trial out of its rate.
    if static.get(ENDED_BY) == ENDED_BY_POLICY and (
        static.get(POLICY_END) == PolicyEnd.GIVE_UP or static.get(ENV_ENDS_ON_SUCCESS) is True
    ):
        return Outcome.FAIL
    return Outcome.UNSCORED
