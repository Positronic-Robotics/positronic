"""The ladder an operator marks as the arm works, recorded as the `progress.state` signal.

Each mark appends the code of a state the arm has reached. The codes are ordered, and an operator
can mark a lower state after a higher one, so the rung reached is the highest code marked.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

# The signal a marked episode carries the state codes in.
STATE_SIGNAL = 'progress.state'


class Stage(StrEnum):
    """One state on the ladder every marked task passes through, declared lowest first."""

    FLOATING = 'floating'
    REACHING = 'reaching'
    CONTACT = 'contact'
    CONTROL = 'control'
    AT_TARGET = 'at-target'


# The stages in the order a task passes through them, derived from `Stage` so the two cannot disagree.
LADDER: tuple[Stage, ...] = tuple(Stage)

# How a reader is told each stage. `floating` reads as `moving free`: the arm moves while it holds
# nothing, and a word like `not started` calls a running episode idle.
STAGE_LABELS: dict[Stage, str] = {
    Stage.FLOATING: 'moving free',
    Stage.REACHING: 'reaching',
    Stage.CONTACT: 'in contact',
    Stage.CONTROL: 'moving it',
    Stage.AT_TARGET: 'at the target',
}


def highest_stage(codes: Iterable[str]) -> Stage | None:
    """The highest stage in `LADDER` among `codes`, or None where none is a stage.

    A code that is not a stage is ignored, so a recording made against a longer ladder still reads.
    """
    reached = [Stage(code) for code in codes if code in LADDER]
    return max(reached, key=LADDER.index) if reached else None
