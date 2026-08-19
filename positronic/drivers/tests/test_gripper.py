"""When a gripper counts as arrived, and what a caller waiting on it hears."""

import pimm
from positronic.drivers.gripper import ARRIVED_TOL, answer_when_arrived


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

    assert answer_when_arrived(call, ARRIVED_TOL / 2, 0.0, out_of_time=False) is None
    assert call.answered and call.exception is None


def test_a_gripper_still_moving_keeps_the_caller_waiting():
    call = _Call(1.0)

    assert answer_when_arrived(call, 0.0, 1.0, out_of_time=False) is call
    assert not call.answered


def test_a_gripper_that_never_arrives_fails_the_caller_rather_than_holding_it():
    """Fingers stopped by what they are holding never reach their target, and a caller told nothing waits for
    the rest of the run."""
    call = _Call(1.0)

    assert answer_when_arrived(call, 0.4, 1.0, out_of_time=True) is None
    assert isinstance(call.exception, TimeoutError)
    assert 'stopped at 0.40' in str(call.exception)


def test_a_gripper_out_of_time_but_at_its_target_has_arrived():
    """Arrival is judged before the clock: a move that lands on the deadline succeeded."""
    call = _Call(1.0)

    assert answer_when_arrived(call, 1.0, 1.0, out_of_time=True) is None
    assert call.exception is None
