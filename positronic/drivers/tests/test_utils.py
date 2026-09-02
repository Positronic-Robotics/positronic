"""When a device counts as arrived, and what a caller waiting on it hears."""

import pickle

import numpy as np
import pytest

import pimm
from pimm.tests.testing import FakeCall, Passive, wire_call
from positronic.drivers.utils import _GRIP_TIMEOUT_S, MoveAbandoned, Moves, MoveStatus, grip_setpoint
from positronic.tests.testing_coutils import ManualCommandReceiver

TOL = 0.05


def _accepted(
    target: float | np.ndarray, tol: float = TOL, timeout_s: float = 3.0
) -> tuple[Moves[float], FakeCall[float, None]]:
    moves = _unasked()
    call = FakeCall[float, None](0.0)
    moves.accept(call, target, tol, now=0.0, timeout_s=timeout_s)
    return moves, call


def test_an_abandoned_move_survives_the_trip_to_another_process():
    """A driver in a background process answers over a pipe, and an exception is rebuilt from its args."""
    assert isinstance(pickle.loads(pickle.dumps(MoveAbandoned())), MoveAbandoned)


def test_a_device_within_tolerance_has_arrived():
    moves, call = _accepted(0.0)

    assert moves.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert not moves.active and not moves.errored
    moves.answer()
    assert call.answered and call.exception is None


def test_a_settled_move_waits_for_the_driver_to_publish_before_it_is_answered():
    """Answering before the state that says so is out hands the caller the sample from mid-travel."""
    moves, call = _accepted(0.0)

    assert moves.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert not call.answered, 'settling is not answering'
    assert not moves.active, 'the device is free, whatever its asker has been told'

    moves.answer()
    assert call.answered


def test_a_device_still_moving_keeps_the_caller_waiting():
    moves, call = _accepted(1.0)

    assert moves.settle(0.0, now=0.1) is MoveStatus.MOVING
    assert not call.answered
    assert moves.active


def test_a_device_that_never_arrives_fails_the_caller_rather_than_holding_it():
    """Fingers stopped by what they are holding never reach their target."""
    moves, call = _accepted(1.0)

    assert moves.settle(0.4, now=3.0) is MoveStatus.GAVE_UP
    moves.answer()
    assert isinstance(call.exception, TimeoutError)
    assert 'stopped at 0.4' in str(call.exception)
    assert not moves.active and moves.errored


def test_a_device_out_of_time_but_at_its_target_has_arrived():
    """Arrival is judged before the clock: a move that lands on the deadline succeeded."""
    moves, call = _accepted(1.0)

    assert moves.settle(1.0, now=3.0) is MoveStatus.ARRIVED
    moves.answer()
    assert call.exception is None
    assert not moves.errored


def test_an_arm_arrives_only_once_every_joint_is_within_tolerance():
    """A move is over when the whole arm is there, not when most of it is."""
    moves, call = _accepted(np.zeros(5), tol=0.02)

    assert moves.settle(np.array([0.0, 0.0, 0.0, 0.0, 0.5]), now=0.1) is MoveStatus.MOVING
    assert moves.settle(np.full(5, 0.01), now=0.2) is MoveStatus.ARRIVED
    moves.answer()
    assert call.answered and call.exception is None


def test_a_move_that_arrives_clears_the_error_left_by_one_that_did_not():
    """ERROR stands until a move genuinely lands, so a caller cannot read the arm as sound in between."""
    moves, _ = _accepted(1.0)
    moves.settle(0.4, now=3.0)
    assert moves.errored

    moves.accept(FakeCall[float, None](0.0), 1.0, TOL, now=3.0, timeout_s=3.0)
    assert moves.settle(1.0, now=3.1) is MoveStatus.ARRIVED
    assert not moves.errored


def _unasked() -> Moves[float]:
    """Moves driven by hand rather than asked for: nothing is bound to either way of asking."""
    handler = pimm.calls.ControlSystemHandler[float, None](Passive())
    return Moves[float](handler, ManualCommandReceiver())


@pytest.fixture
def asking():
    """The two ways a gripper is asked for a width: a caller wired to the moves it serves, so a test asks
    the way a client does, and the stream its setpoints arrive on."""
    caller = pimm.calls.ControlSystemCaller[float, None](Passive())
    handler = pimm.calls.ControlSystemHandler[float, None](Passive())
    stream = ManualCommandReceiver()
    with pimm.World() as world:
        wire_call(world, caller, handler)
        yield caller, Moves[float](handler, stream), stream


def test_a_grip_target_that_is_no_width_is_refused_rather_than_saturated(asking):
    """``min``/``max`` turn NaN into a bound, so an unchecked target would close the fingers at full force."""
    ask, moves, _ = asking
    answer = ask(float('nan'))

    assert grip_setpoint(moves, grip=0.0, now=0.0) is None
    assert not moves.active
    with pytest.raises(ValueError, match='not a grip width'):
        answer.result()


def test_a_streamed_grip_target_that_is_no_width_leaves_the_fingers_alone(asking):
    """A command stream cannot end the run, so a malformed target is dropped and the last one stands."""
    _, moves, stream = asking
    stream.push(float('inf'))

    assert grip_setpoint(moves, grip=0.4, now=0.0) is None


def test_a_grip_call_takes_the_fingers_until_it_arrives(asking):
    ask, moves, _ = asking
    answer = ask(0.7)

    assert grip_setpoint(moves, grip=0.0, now=0.0) == 0.7
    assert moves.active
    assert grip_setpoint(moves, grip=0.3, now=0.1) is None, 'commanded again mid-travel'
    assert grip_setpoint(moves, grip=0.7, now=0.2) is None
    assert not moves.active
    moves.answer()
    assert answer.result() is None


def test_a_grip_that_gives_up_hands_back_the_width_the_fingers_stopped_at(asking):
    """The answer waits for the driver to write the width handed back here, so the fingers stop first."""
    ask, moves, _ = asking
    answer = ask(1.0)
    grip_setpoint(moves, grip=0.0, now=0.0)

    assert grip_setpoint(moves, grip=0.42, now=_GRIP_TIMEOUT_S) == 0.42
    assert not moves.active and moves.errored
    assert not answer.done(), 'the fingers are still on the width they missed'

    moves.answer()
    with pytest.raises(TimeoutError, match='stopped at 0.42'):
        answer.result()


def test_a_grip_asked_for_past_the_range_is_tracked_against_a_width_the_fingers_report(asking):
    """The fingers read back 0..1, so a move aimed past that would sit at the endpoint until its deadline."""
    ask, moves, _ = asking
    answer = ask(1.5)

    assert grip_setpoint(moves, grip=0.0, now=0.0) == 1.0
    assert grip_setpoint(moves, grip=1.0, now=0.1) is None
    moves.answer()
    assert answer.result() is None


def test_a_setpoint_the_device_was_moved_away_from_is_let_go(asking):
    """A setpoint says where the device is wanted now, and the move taken after it puts the device
    somewhere else. Applying the setpoint once the move lands takes the device back off the pose it was
    asked for, in one step and with nobody asking."""
    ask, moves, stream = asking
    stream.push(0.25)
    ask(0.9)

    assert grip_setpoint(moves, grip=0.0, now=0.0) == 0.9
    assert grip_setpoint(moves, grip=0.9, now=0.1) is None  # the call arrives
    moves.answer()
    assert grip_setpoint(moves, grip=0.9, now=0.2) is None, 'the setpoint the move superseded reached the device'


def test_the_newest_setpoint_is_what_the_device_is_asked_for(asking):
    """A transport that queues setpoints hands the oldest over first. A device asked for one a tick is
    driven through what its asker has already done, and falls further behind the longer it runs."""
    _ask, moves, stream = asking
    for width in (0.1, 0.2, 0.3):
        stream.push(width)

    assert grip_setpoint(moves, grip=0.0, now=0.0) == 0.3
    assert grip_setpoint(moves, grip=0.3, now=0.1) is None, 'the device was asked for a width it had passed'


def test_setpoints_streamed_at_a_travelling_device_do_not_reach_it_when_the_move_lands(asking):
    """Nothing reads the stream for as long as a move owns the device, so what arrives in that time is
    where the asker wanted the device before the move — every setpoint of it, oldest first."""
    ask, moves, stream = asking
    ask(0.9)
    assert grip_setpoint(moves, grip=0.0, now=0.0) == 0.9

    stream.push(0.25)
    stream.push(0.30)
    assert grip_setpoint(moves, grip=0.0, now=0.1) is None  # the move still travels
    assert grip_setpoint(moves, grip=0.9, now=0.2) is None  # ... and lands
    moves.answer()

    assert grip_setpoint(moves, grip=0.9, now=0.3) is None, 'the device was driven back through the stream'


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
    moves, call = _accepted(1.0)

    moves.fail(RuntimeError('the bus went away'))

    assert isinstance(call.exception, RuntimeError)
    assert not moves.active and moves.errored


def test_a_run_that_dies_with_nothing_in_flight_has_nobody_to_tell():
    moves = _unasked()

    moves.fail(RuntimeError('the bus went away'))

    assert not moves.active and not moves.errored


def test_a_settled_move_holds_the_device_against_the_next_one(asking):
    """Taking another move first would put BUSY over the state the settled move's asker is owed."""
    ask, moves, _ = asking
    ask(0.0)
    accepted = moves.next_request()
    assert isinstance(accepted, pimm.calls.Call)
    moves.accept(accepted, 0.0, TOL, now=0.0, timeout_s=3.0)
    ask(1.0)

    assert moves.settle(TOL / 2, now=0.1) is MoveStatus.ARRIVED
    assert moves.next_request() is None, 'settled, and its asker not yet told'

    moves.answer()
    assert isinstance(moves.next_request(), pimm.calls.Call)


def test_a_device_still_travelling_is_asked_for_nothing(asking):
    """A setpoint applied mid-travel fights the move, and its asker is owed the arrival it was promised."""
    ask, moves, stream = asking
    ask(1.0)
    travelling = moves.next_request()
    assert isinstance(travelling, pimm.calls.Call)
    moves.accept(travelling, 1.0, TOL, now=0.0, timeout_s=3.0)

    stream.push(0.25)
    ask(0.5)

    assert moves.next_request() is None
    assert moves.settle(0.0, now=0.1) is MoveStatus.MOVING


def test_a_run_that_dies_with_one_move_settled_and_another_in_flight_answers_both():
    """One outcome each: the settled move earned its answer, the travelling one is owed what killed it."""
    moves, landed = _accepted(0.0)
    moves.settle(TOL / 2, now=0.1)
    travelling = FakeCall[float, None](0.0)
    moves.accept(travelling, 1.0, TOL, now=0.1, timeout_s=3.0)

    moves.fail(RuntimeError('the bus went away'))

    assert landed.answered and landed.exception is None
    assert isinstance(travelling.exception, RuntimeError)


def test_a_run_that_dies_after_a_move_settled_still_hands_over_the_outcome():
    """The move is over; what killed the run afterwards did not change that."""
    moves, call = _accepted(0.0)
    moves.settle(TOL / 2, now=0.1)

    moves.fail(RuntimeError('the bus went away'))

    assert call.answered and call.exception is None
