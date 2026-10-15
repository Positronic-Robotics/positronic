"""The outcome a judge gives an episode, recorded as `eval.outcome`, and the outcome a reader counts.

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

# The statics that the judge gives an attended episode: the verdict and the item counts. The
# harness writes the rest.
OUTCOME = 'eval.outcome'
SUCCESSFUL_ITEMS = 'eval.successful_items'
TOTAL_ITEMS = 'eval.total_items'


class Outcome(StrEnum):
    """How an episode ended, as `eval.outcome` records it."""

    SUCCESS = 'Success'
    FAIL = 'Fail'
    # The verdict for an episode that ran out of time. A console writes it. The harness records a timeout as
    # `eval.terminated = False`, which `counted_outcome` counts as this.
    OUT_OF_TIME = 'Ran out of time'
    SAFETY = 'Safety'
    SYSTEM = 'System'
    # What an episode nobody scored carries. A reader counts an episode, with or without an outcome, by
    # `counted_outcome`.
    UNSCORED = 'UNSCORED'
    # What the judge's Discard writes: the attempt is not scored at all.
    DISCARDED = 'Discarded'

    # A member reads as its wire value inside a container's repr, not as its enum name.
    __repr__ = str.__repr__


# The verdicts the judge gives, in the order the console's radio lists them.
VERDICTS: tuple[Outcome, ...] = (Outcome.SUCCESS, Outcome.FAIL, Outcome.OUT_OF_TIME, Outcome.SAFETY, Outcome.SYSTEM)

# The value a console seeds `eval.outcome` with before a judgement. `counted_outcome` reads an absent outcome by the
# other facts of the episode.
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
