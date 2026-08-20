"""When a device counts as arrived, and what a caller waiting on it hears."""

import numpy as np
import pytest

import pimm
from pimm.tests.testing import wire_call
from positronic.drivers.utils import _GRIP_TIMEOUT_S, MoveStatus, PendingMove, grip_setpoint
from positronic.tests.testing_coutils import ManualCommandReceiver

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


def _accepted(target: float | np.ndarray, tol: float = TOL, timeout_s: float = 3.0) -> tuple[PendingMove, _Call]:
    move = PendingMove(tol)
    call = _Call(0.0)
    move.accept(call, target, now=0.0, timeout_s=timeout_s)
    return move, call


def test_a_device_within_tolerance_has_arrived():
    move, call = _accepted(0.0)

    assert move.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert not move.active and not move.errored
    move.answer()
    assert call.answered and call.exception is None


def test_a_settled_move_waits_for_the_driver_to_publish_before_it_is_answered():
    """Answering before the state that says so is out hands the caller the sample from mid-travel."""
    move, call = _accepted(0.0)

    assert move.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert not call.answered, 'settling is not answering'
    assert not move.active, 'the device is free, whatever its asker has been told'

    move.answer()
    assert call.answered


def test_a_device_still_moving_keeps_the_caller_waiting():
    move, call = _accepted(1.0)

    assert move.settle(0.0, now=0.1) is MoveStatus.MOVING
    assert not call.answered
    assert move.active


def test_a_device_that_never_arrives_fails_the_caller_rather_than_holding_it():
    """Fingers stopped by what they are holding never reach their target."""
    move, call = _accepted(1.0)

    assert move.settle(0.4, now=3.0) is MoveStatus.GAVE_UP
    move.answer()
    assert isinstance(call.exception, TimeoutError)
    assert 'stopped at 0.4' in str(call.exception)
    assert not move.active and move.errored


def test_a_device_out_of_time_but_at_its_target_has_arrived():
    """Arrival is judged before the clock: a move that lands on the deadline succeeded."""
    move, call = _accepted(1.0)

    assert move.settle(1.0, now=3.0) is MoveStatus.ARRIVED
    move.answer()
    assert call.exception is None
    assert not move.errored


def test_an_arm_arrives_only_once_every_joint_is_within_tolerance():
    """A move is over when the whole arm is there, not when most of it is."""
    move, call = _accepted(np.zeros(5), tol=0.02)

    assert move.settle(np.array([0.0, 0.0, 0.0, 0.0, 0.5]), now=0.1) is MoveStatus.MOVING
    assert move.settle(np.full(5, 0.01), now=0.2) is MoveStatus.ARRIVED
    move.answer()
    assert call.answered and call.exception is None


def test_a_move_that_arrives_clears_the_error_left_by_one_that_did_not():
    """ERROR stands until a move genuinely lands, so a caller cannot read the arm as sound in between."""
    move, _ = _accepted(1.0)
    move.settle(0.4, now=3.0)
    assert move.errored

    move.accept(_Call(0.0), 1.0, now=3.0, timeout_s=3.0)
    assert move.settle(1.0, now=3.1) is MoveStatus.ARRIVED
    assert not move.errored


class _Idle(pimm.ControlSystem):
    """An owner for a caller/handler pair that no test schedules."""

    def run(self, should_stop, clock):
        yield pimm.Sleep(0.0)


@pytest.fixture
def asking():
    """A caller wired to the handler a gripper polls, so a test asks the way a client does."""
    caller = pimm.calls.ControlSystemCaller[float, None](_Idle())
    handler = pimm.calls.ControlSystemHandler[float, None](_Idle())
    with pimm.World() as world:
        wire_call(world, caller, handler)
        yield caller, handler


def test_a_grip_call_takes_the_fingers_until_it_arrives(asking):
    ask, calls = asking
    move, stream = PendingMove(TOL), ManualCommandReceiver()
    answer = ask(0.7)

    assert grip_setpoint(move, calls, stream, grip=0.0, now=0.0) == 0.7
    assert move.active
    assert grip_setpoint(move, calls, stream, grip=0.3, now=0.1) is None, 'commanded again mid-travel'
    assert grip_setpoint(move, calls, stream, grip=0.7, now=0.2) is None
    assert not move.active
    move.answer()
    assert answer.result() is None


def test_a_grip_that_gives_up_hands_back_the_width_the_fingers_stopped_at(asking):
    """The answer waits for the driver to write the width handed back here, so the fingers stop first."""
    ask, calls = asking
    move, stream = PendingMove(TOL), ManualCommandReceiver()
    answer = ask(1.0)
    grip_setpoint(move, calls, stream, grip=0.0, now=0.0)

    assert grip_setpoint(move, calls, stream, grip=0.42, now=_GRIP_TIMEOUT_S) == 0.42
    assert not move.active and move.errored
    assert not answer.done(), 'the fingers are still on the width they missed'

    move.answer()
    with pytest.raises(TimeoutError, match='stopped at 0.42'):
        answer.result()


def test_a_grip_asked_for_past_the_range_is_tracked_against_a_width_the_fingers_report(asking):
    """The fingers read back 0..1, so a move aimed past that would sit at the endpoint until its deadline."""
    ask, calls = asking
    move, stream = PendingMove(TOL), ManualCommandReceiver()
    answer = ask(1.5)

    assert grip_setpoint(move, calls, stream, grip=0.0, now=0.0) == 1.0
    assert grip_setpoint(move, calls, stream, grip=1.0, now=0.1) is None
    move.answer()
    assert answer.result() is None


def test_a_streamed_grip_waits_for_the_call_queue_to_be_empty(asking):
    """A signal holds only its latest value, so a stream read in the same tick as a call would be lost."""
    ask, calls = asking
    move, stream = PendingMove(TOL), ManualCommandReceiver()
    stream.push(0.25)
    ask(0.9)

    assert grip_setpoint(move, calls, stream, grip=0.0, now=0.0) == 0.9
    assert grip_setpoint(move, calls, stream, grip=0.9, now=0.1) is None  # the call arrives
    move.answer()
    assert grip_setpoint(move, calls, stream, grip=0.9, now=0.2) == 0.25  # the stream, still waiting
    assert grip_setpoint(move, calls, stream, grip=0.25, now=0.3) is None


def test_how_long_a_move_gets_is_the_driver_s_to_say():
    """An arm ramping to a capped speed needs longer for a longer trip than fingers closing on an object."""
    brief, brief_call = _accepted(1.0, timeout_s=1.0)
    patient, patient_call = _accepted(1.0, timeout_s=10.0)

    assert brief.settle(0.0, now=2.0) is MoveStatus.GAVE_UP
    brief.answer()
    assert isinstance(brief_call.exception, TimeoutError)
    assert patient.settle(0.0, now=2.0) is MoveStatus.MOVING
    assert not patient_call.answered


def test_a_run_that_dies_hands_what_killed_it_to_the_move_in_flight():
    """The asker is blocked on an answer, and a driver that stops looping will never produce one."""
    move, call = _accepted(1.0)

    move.fail(RuntimeError('the bus went away'))

    assert isinstance(call.exception, RuntimeError)
    assert not move.active and move.errored


def test_a_run_that_dies_with_nothing_in_flight_has_nobody_to_tell():
    move = PendingMove(TOL)

    move.fail(RuntimeError('the bus went away'))

    assert not move.active and not move.errored


def test_a_settled_move_holds_the_device_against_the_next_one():
    """Taking another move first would put BUSY over the state the settled move's asker is owed."""
    move, _ = _accepted(0.0)

    assert move.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert not move.active and move.settled

    move.answer()
    assert not move.settled


def test_a_run_that_dies_with_one_move_settled_and_another_in_flight_answers_both():
    """One outcome each: the settled move earned its answer, the travelling one is owed what killed it."""
    move, landed = _accepted(0.0)
    move.settle(TOL / 2, now=0.1)
    travelling = _Call(0.0)
    move.accept(travelling, 1.0, now=0.1, timeout_s=3.0)

    move.fail(RuntimeError('the bus went away'))

    assert landed.answered and landed.exception is None
    assert isinstance(travelling.exception, RuntimeError)


def test_a_run_that_dies_after_a_move_settled_still_hands_over_the_outcome():
    """The move is over; what killed the run afterwards did not change that."""
    move, call = _accepted(0.0)
    move.settle(TOL / 2, now=0.1)

    move.fail(RuntimeError('the bus went away'))

    assert call.answered and call.exception is None
