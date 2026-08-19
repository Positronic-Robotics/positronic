"""When a device counts as having arrived, for drivers that answer a synchronous move."""

from enum import Enum, auto
from typing import TypeVar

import numpy as np

from pimm.calls import Call

Req = TypeVar('Req')

# A device stopped by what it is holding never reaches its target
ARRIVAL_TIMEOUT_S = 3.0


class MoveStatus(Enum):
    """Where a move stands."""

    MOVING = auto()
    ARRIVED = auto()
    GAVE_UP = auto()


def answer_when_arrived(
    call: Call[Req, None], position: np.ndarray | float, target: np.ndarray | float, tol: float, out_of_time: bool
) -> MoveStatus:
    """Answer `call` once the device is within `tol` of `target`, or once it has run out of time."""
    if bool(np.all(np.abs(np.asarray(position) - np.asarray(target)) < tol)):
        call.set_result(None)
        return MoveStatus.ARRIVED
    if out_of_time:
        call.set_exception(TimeoutError(f'stopped at {np.round(position, 3)}, short of {np.round(target, 3)}'))
        return MoveStatus.GAVE_UP
    return MoveStatus.MOVING
