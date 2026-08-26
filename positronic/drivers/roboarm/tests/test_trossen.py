"""What the Trossen driver puts on the link, and what a caller waiting on a move hears."""

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm import trossen as trossen_driver
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import MoveAbandoned
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

GRIP_TRAVEL_M = 0.04  # the gripper joint's range, which the arm reports and grip is normalized against
JOGGED = np.array([0.2, 0.4, 0.3, 0.0, 0.1, 0.0])


class FakeArm(trossen_driver._FakeTrossen):
    """The driver's fake with the link under the test's control.

    ``blocked`` holds the joints where they stand while the controller keeps streaming; ``frozen`` stops
    the stream, as a dropped link does. ``raises`` is what reading the arm raises, ``write_raises`` what
    a write raises.
    """

    def __init__(self, position=None):
        super().__init__()
        if position is not None:
            self._position = np.asarray(position, dtype=np.float64)
        self.raises: Exception | None = None
        self.write_raises: Exception | None = None
        self.blocked = False

    def get_robot_output(self):
        if self.raises is not None:
            raise self.raises
        return super().get_robot_output()

    def set_all_modes(self, mode) -> None:
        if self.write_raises is not None:
            raise self.write_raises
        super().set_all_modes(mode)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        if self.write_raises is not None:
            raise self.write_raises
        super().set_all_positions(goal_positions, goal_time, blocking)

    def _servo(self) -> None:
        if not self.blocked:
            super()._servo()


def _driven(arm: FakeArm, clock: MockClock | None = None, stop: StopFlag | None = None):
    """A driver over ``arm`` with its state recorded, and its loop ready to pump."""
    driver = trossen_driver.Robot('192.168.1.4', connect=lambda ip: arm)
    states = RecordingEmitter()
    driver.state._bind(states)
    return driver, states, driver.run(stop or StopFlag(), clock or MockClock())


def test_the_link_is_written_only_once_something_has_asked_for_a_setpoint():
    """A goal time re-sent every tick restarts the trajectory it plans, so a held setpoint is not rewritten."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    next(loop)  # entering position mode writes the hold the servo starts from
    written = len(arm.goals)
    for _ in range(3):  # and nothing has asked since
        next(loop)
    assert len(arm.goals) == written

    grip.push(0.25)
    next(loop)
    assert len(arm.goals) == written + 1

    next(loop)
    assert len(arm.goals) == written + 1


def test_an_open_grip_reaches_the_arm_as_the_joint_at_its_upper_limit():
    """positronic counts 1 as closed; the joint counts its lower limit as closed."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(0.0)
    next(loop)

    assert arm.goals[-1][trossen_driver._GRIPPER_JOINT] == pytest.approx(GRIP_TRAVEL_M)


def test_a_closed_grip_reaches_the_arm_as_the_joint_at_its_lower_limit():
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(1.0)
    next(loop)

    assert arm.goals[-1][trossen_driver._GRIPPER_JOINT] == pytest.approx(0.0)


def test_the_joint_the_arm_reports_comes_back_as_a_normalized_grip():
    """The reading crosses the same conversion as the goal, the other way."""
    arm = FakeArm(position=np.append(np.zeros(6), GRIP_TRAVEL_M / 4))
    driver, _, loop = _driven(arm)
    grips = RecordingEmitter()
    driver.grip._bind(grips)

    next(loop)

    assert grips.emitted[-1][1] == pytest.approx(0.75)


def test_a_joint_reading_just_outside_its_range_still_comes_back_a_grip():
    """A closed gripper reads a shade below its lower limit, and the port carries 0..1."""
    arm = FakeArm(position=np.append(np.zeros(6), -0.000945))
    driver, _, loop = _driven(arm)
    grips = RecordingEmitter()
    driver.grip._bind(grips)

    next(loop)

    assert grips.emitted[-1][1] == pytest.approx(1.0)


def test_a_streamed_joint_command_reaches_the_arm():
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(JOGGED))
    next(loop)

    np.testing.assert_allclose(arm.goals[-1][: trossen_driver._ARM_JOINTS], JOGGED)


def test_a_joint_target_outside_the_range_is_clipped_to_it():
    """The second joint has no negative half, and a target below it is held at the limit, not refused."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
    next(loop)

    assert arm.goals[-1][1] == pytest.approx(0.0)


def test_a_streamed_command_the_arm_cannot_be_put_at_leaves_it_where_it_is():
    """A command stream cannot end the run: the next command supersedes one that could not be applied."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    next(loop)
    written = len(arm.goals)
    commands.push(command.CartesianPosition(geom.Transform3D(np.array([0.3, 0.0, 0.2]))))
    next(loop)

    assert len(arm.goals) == written  # nothing new was written


def test_a_read_that_raises_reads_error_and_the_run_carries_on():
    arm = FakeArm()
    _, states, loop = _driven(arm)

    next(loop)
    arm.raises = trossen_driver.trossen_arm.RuntimeError('the controller stopped answering')
    next(loop)
    assert states.emitted[-1][1].status == RobotStatus.ERROR

    arm.raises = None
    next(loop)
    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE


def test_an_arm_that_stops_streaming_reads_error_though_the_read_still_answers():
    """A dropped link does not make the read raise: it hands back the last telemetry, over and over."""
    arm = FakeArm()
    clock = MockClock()
    _, states, loop = _driven(arm, clock)

    next(loop)
    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE

    arm.frozen = True
    clock.advance(trossen_driver._STALE_AFTER_S + 0.1)
    next(loop)
    assert states.emitted[-1][1].status == RobotStatus.ERROR

    arm.frozen = False
    next(loop)
    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE


def test_a_setpoint_the_link_refuses_reads_error_and_goes_out_again_when_the_arm_answers():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    next(loop)
    written = len(arm.goals)
    arm.write_raises = trossen_driver.trossen_arm.RuntimeError('Broken pipe')
    commands.push(command.JointPosition(JOGGED))
    next(loop)
    assert len(arm.goals) == written
    assert states.emitted[-1][1].status == RobotStatus.ERROR

    arm.write_raises = None
    next(loop)
    np.testing.assert_allclose(arm.goals[-1][: trossen_driver._ARM_JOINTS], JOGGED)


def test_a_run_that_ends_on_a_dead_link_still_gives_the_handle_back():
    """Setting the arm idle is what the run tries last, and an arm it cannot reach must not end it badly."""
    arm = FakeArm()
    stop = StopFlag()
    _, _, loop = _driven(arm, stop=stop)

    next(loop)
    arm.write_raises = trossen_driver.trossen_arm.RuntimeError('Connection reset by peer')
    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    assert arm.cleaned_up


def test_a_sync_move_answers_once_the_arm_reads_back_at_its_target(world):
    arm = FakeArm()
    clock = MockClock()
    driver, states, loop = _driven(arm, clock)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)

    answer = caller(command.JointPosition(JOGGED))
    for _ in range(60):
        if answer.done():
            break
        next(loop)
    answer.result()

    np.testing.assert_allclose(states.emitted[-1][1].q, JOGGED, atol=trossen_driver._ARRIVED_TOL)


def test_a_sync_move_hands_the_firmware_a_goal_time_and_a_streamed_setpoint_does_not(world):
    """The firmware plans the trajectory a move asks for; a streamed setpoint is one tick away already."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)

    commands.push(command.JointPosition(JOGGED))
    next(loop)
    assert arm.goal_times[-1] == pytest.approx(trossen_driver._STREAM_GOAL_TIME_S)

    caller(command.JointPosition(np.zeros(6)))
    next(loop)
    assert arm.goal_times[-1] == pytest.approx(trossen_driver._MOVE_GOAL_TIME_S)


def test_a_move_the_world_stops_under_is_handed_back_to_its_asker(world):
    arm = FakeArm()
    arm.blocked = True  # the arm never reaches the target, so the move is still in flight when the run ends
    stop = StopFlag()
    driver, _, loop = _driven(arm, stop=stop)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)

    answer = caller(command.JointPosition(JOGGED))
    next(loop)
    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    assert answer.done()
    with pytest.raises(MoveAbandoned):
        answer.result()
