"""When a device counts as arrived, and what a caller waiting on it hears."""

import numpy as np

import pimm
from positronic.drivers.arrival import MoveStatus, answer_when_arrived

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


def test_a_gripper_within_tolerance_has_arrived():
    call = _Call(0.0)

    assert answer_when_arrived(call, TOL / 2, 0.0, TOL, out_of_time=False) is MoveStatus.ARRIVED
    assert call.answered and call.exception is None


def test_a_gripper_still_moving_keeps_the_caller_waiting():
    call = _Call(1.0)

    assert answer_when_arrived(call, 0.0, 1.0, TOL, out_of_time=False) is MoveStatus.MOVING
    assert not call.answered


def test_a_gripper_that_never_arrives_fails_the_caller_rather_than_holding_it():
    """Fingers stopped by what they are holding never reach their target, and a caller told nothing waits for
    the rest of the run."""
    call = _Call(1.0)

    assert answer_when_arrived(call, 0.4, 1.0, TOL, out_of_time=True) is MoveStatus.GAVE_UP
    assert isinstance(call.exception, TimeoutError)
    assert 'stopped at 0.4' in str(call.exception)


def test_a_gripper_out_of_time_but_at_its_target_has_arrived():
    """Arrival is judged before the clock: a move that lands on the deadline succeeded."""
    call = _Call(1.0)

    assert answer_when_arrived(call, 1.0, 1.0, TOL, out_of_time=True) is MoveStatus.ARRIVED
    assert call.exception is None


def test_an_arm_arrives_only_once_every_joint_is_within_tolerance():
    """A move is over when the whole arm is there, not when most of it is."""
    call = _Call(0.0)
    target = np.zeros(5)

    lagging = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    assert answer_when_arrived(call, lagging, target, 0.02, out_of_time=False) is MoveStatus.MOVING
    assert answer_when_arrived(call, np.full(5, 0.01), target, 0.02, out_of_time=False) is MoveStatus.ARRIVED
    assert call.answered and call.exception is None
