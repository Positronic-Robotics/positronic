"""When a device counts as having arrived, for drivers that answer a synchronous move."""

from enum import Enum, auto
from typing import Any

import numpy as np

from pimm.calls import Call

# A device stopped by what it is holding never reaches its target
ARRIVAL_TIMEOUT_S = 3.0


class MoveStatus(Enum):
    """Where a move stands."""

    MOVING = auto()
    ARRIVED = auto()
    GAVE_UP = auto()


class PendingMove:
    """The synchronous move a driver has in flight, if any.

    For a device whose control loop cannot be held for the duration of a move: the driver carries one of
    these across ticks and settles it against what the device reads back. A move owns the device until it
    answers, so a driver leaves its command stream unread while ``active`` — a superseding setpoint would
    otherwise fail a move for something its asker never did.
    """

    def __init__(self, tol: float, timeout_s: float = ARRIVAL_TIMEOUT_S):
        self._tol = tol
        self._timeout_s = timeout_s
        self._call: Call[Any, None] | None = None
        self._target: np.ndarray | float = 0.0
        self._deadline = 0.0
        # Set by a move that does not arrive, cleared by the next that does: the device is not where it was put
        self.errored = False

    @property
    def active(self) -> bool:
        return self._call is not None

    def accept(self, call: Call[Any, None], target: np.ndarray | float, now: float) -> None:
        """Take `call` as the move in flight, aiming at `target`."""
        self._call, self._target, self._deadline = call, target, now + self._timeout_s

    def settle(self, position: np.ndarray | float, now: float) -> MoveStatus:
        """Answer the move in flight once the device reads back at its target, or once it runs out of time."""
        assert self._call is not None, 'no move is in flight'
        if bool(np.all(np.abs(np.asarray(position) - np.asarray(self._target)) < self._tol)):
            self._call.set_result(None)
            self._call, self.errored = None, False
            return MoveStatus.ARRIVED
        if now >= self._deadline:
            self._call.set_exception(
                TimeoutError(f'stopped at {np.round(position, 3)}, short of {np.round(self._target, 3)}')
            )
            self._call, self.errored = None, True
            return MoveStatus.GAVE_UP
        return MoveStatus.MOVING
