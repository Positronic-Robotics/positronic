"""When a device counts as arrived, and what a caller waiting on it hears."""

import numpy as np

import pimm
from positronic.drivers.arrival import MoveStatus, PendingMove

TOL = 0.05


class _Call(pimm.calls.Call[float, None]):
    """Records the one answer a call is allowed."""

    def __init__(self, request: float):
        self._request = request
        self.answered = False
        self.exception: BaseException | None = None

    @property
    def request(self) -> float:
        return self._request

    def set_result(self, value: None) -> None:
        self.answered = True

    def set_exception(self, exc: BaseException) -> None:
        self.answered = True
        self.exception = exc


def _accepted(target: float | np.ndarray, tol: float = TOL) -> tuple[PendingMove, _Call]:
    move = PendingMove(tol, timeout_s=3.0)
    call = _Call(0.0)
    move.accept(call, target, now=0.0)
    return move, call


def test_a_device_within_tolerance_has_arrived():
    move, call = _accepted(0.0)

    assert move.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert call.answered and call.exception is None
    assert not move.active and not move.errored


def test_a_device_still_moving_keeps_the_caller_waiting():
    move, call = _accepted(1.0)

    assert move.settle(0.0, now=0.1) is MoveStatus.MOVING
    assert not call.answered
    assert move.active


def test_a_device_that_never_arrives_fails_the_caller_rather_than_holding_it():
    """Fingers stopped by what they are holding never reach their target, and a caller told nothing waits for
    the rest of the run."""
    move, call = _accepted(1.0)

    assert move.settle(0.4, now=3.0) is MoveStatus.GAVE_UP
    assert isinstance(call.exception, TimeoutError)
    assert 'stopped at 0.4' in str(call.exception)
    assert not move.active and move.errored


def test_a_device_out_of_time_but_at_its_target_has_arrived():
    """Arrival is judged before the clock: a move that lands on the deadline succeeded."""
    move, call = _accepted(1.0)

    assert move.settle(1.0, now=3.0) is MoveStatus.ARRIVED
    assert call.exception is None
    assert not move.errored


def test_an_arm_arrives_only_once_every_joint_is_within_tolerance():
    """A move is over when the whole arm is there, not when most of it is."""
    move, call = _accepted(np.zeros(5), tol=0.02)

    assert move.settle(np.array([0.0, 0.0, 0.0, 0.0, 0.5]), now=0.1) is MoveStatus.MOVING
    assert move.settle(np.full(5, 0.01), now=0.2) is MoveStatus.ARRIVED
    assert call.answered and call.exception is None


def test_a_move_that_arrives_clears_the_error_left_by_one_that_did_not():
    """ERROR stands until a move genuinely lands, so a caller cannot read the arm as sound in between."""
    move, _ = _accepted(1.0)
    move.settle(0.4, now=3.0)
    assert move.errored

    move.accept(_Call(0.0), 1.0, now=3.0)
    assert move.settle(1.0, now=3.1) is MoveStatus.ARRIVED
    assert not move.errored
