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
        self.configure_raises: Exception | None = None
        self.attempts = 0
        self.blocked = False

    def configure(self, model, end_effector, serv_ip, clear_error, timeout=20.0) -> None:
        self.attempts += 1
        if self.configure_raises is not None:
            raise self.configure_raises
        super().configure(model, end_effector, serv_ip, clear_error, timeout)

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


def _refuse_the_next_write(arm: FakeArm, commands: ManualCommandReceiver, loop) -> None:
    """Break the command half of the link, and give the driver a setpoint that finds out."""
    arm.write_raises = trossen_driver.trossen_arm.RuntimeError('Broken pipe')
    commands.push(command.JointPosition(JOGGED))
    next(loop)


def test_a_setpoint_the_link_refuses_reads_error_and_stops_the_writes():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    next(loop)
    written = len(arm.goals)
    _refuse_the_next_write(arm, commands, loop)
    assert states.emitted[-1][1].status == RobotStatus.ERROR

    for _ in range(5):  # a refused write is not retried a hundred times a second
        next(loop)
    assert len(arm.goals) == written


def test_telemetry_alone_does_not_bring_a_refused_command_channel_back():
    """The controller keeps streaming over a link whose command half is gone, so freshness proves nothing."""
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    _refuse_the_next_write(arm, commands, loop)
    arm.write_raises = None  # the wire is back, but the session the controller dropped is not
    for _ in range(5):
        next(loop)

    assert states.emitted[-1][1].status == RobotStatus.ERROR
    assert arm.sessions == 1


def test_a_link_that_stays_down_gets_a_new_session_and_the_arm_answers_again():
    arm = FakeArm()
    clock = MockClock()
    driver, states, loop = _driven(arm, clock)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)
    assert arm.sessions == 1

    _refuse_the_next_write(arm, commands, loop)
    assert states.emitted[-1][1].status == RobotStatus.ERROR

    arm.write_raises = None
    clock.advance(trossen_driver._RECONNECT_AFTER_S + 0.01)
    next(loop)

    assert arm.sessions == 2
    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE


def test_a_new_session_holds_the_arm_where_it_finds_it():
    """The arm ends up wherever the lost session left it, and driving it back to an old target is a jump."""
    arm = FakeArm()
    clock = MockClock()
    driver, _, loop = _driven(arm, clock)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(JOGGED))
    for _ in range(30):  # let the arm arrive, so its own position is not the zero it booted at
        next(loop)
    where_it_is = np.asarray(arm.get_robot_output().joint.arm.positions)

    _refuse_the_next_write(arm, commands, loop)
    arm.write_raises = None
    clock.advance(trossen_driver._RECONNECT_AFTER_S + 0.01)
    next(loop)

    np.testing.assert_allclose(arm.goals[-1][: trossen_driver._ARM_JOINTS], where_it_is, atol=1e-3)


def test_a_new_session_that_fails_is_tried_again_rather_than_hammered():
    arm = FakeArm()
    clock = MockClock()
    driver, _, loop = _driven(arm, clock)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    _refuse_the_next_write(arm, commands, loop)
    arm.configure_raises = trossen_driver.trossen_arm.RuntimeError('Network is unreachable')
    clock.advance(trossen_driver._RECONNECT_AFTER_S + 0.01)
    for _ in range(20):  # many ticks inside one reconnect interval, one attempt between them
        next(loop)
    assert arm.attempts == 1

    clock.advance(trossen_driver._RECONNECT_EVERY_S + 0.01)
    next(loop)
    assert arm.attempts == 2
    assert arm.sessions == 1  # none of them took


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


HELD_POSE = geom.Transform3D(np.array([0.254, -0.0039, 0.1618]), geom.Rotation.from_rotvec(np.zeros(3)))


def _pose_of(goal: list[float]) -> geom.Transform3D:
    return geom.Transform3D(np.asarray(goal[:3]), geom.Rotation.from_rotvec(np.asarray(goal[3:6])))


def test_a_streamed_cartesian_command_reaches_the_arm_as_a_pose():
    """The controller speaks angle-axis where positronic speaks a rotation."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    target = geom.Transform3D(HELD_POSE.translation + np.array([0.01, 0.0, 0.0]), HELD_POSE.rotation)
    commands.push(command.CartesianPosition(target))
    next(loop)

    reached = _pose_of(arm.poses[-1])
    np.testing.assert_allclose(reached.translation, target.translation, atol=1e-6)
    np.testing.assert_allclose(reached.rotation.as_quat, target.rotation.as_quat, atol=1e-6)


def test_a_cartesian_delta_composes_onto_the_pose_the_controller_reports():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)
    measured = states.emitted[-1][1].ee_pose

    commands.push(command.CartesianDelta(geom.Transform3D(np.array([0.01, 0.0, 0.0]))))
    next(loop)

    expected = measured.translation + np.array([0.01, 0.0, 0.0])
    np.testing.assert_allclose(_pose_of(arm.poses[-1]).translation, expected, atol=1e-5)


def test_a_cartesian_target_out_of_reach_is_capped_to_one_step():
    """A teleoperator reaching past the arm asks for a target that runs away from it."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)
    held = _pose_of(arm.poses[-1]) if arm.poses else HELD_POSE

    commands.push(command.CartesianPosition(geom.Transform3D(held.translation + np.array([5.0, 0.0, 0.0]))))
    next(loop)

    step = np.linalg.norm(_pose_of(arm.poses[-1]).translation - held.translation)
    assert step == pytest.approx(trossen_driver._MAX_STEP_M, abs=1e-6)


def test_a_turn_the_long_way_round_is_capped_along_the_short_one():
    """`as_rotvec` keeps the way round it was given, and a cap along it would drive the arm backwards."""
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)
    held = states.emitted[-1][1].ee_pose

    away = geom.Rotation.from_rotvec(np.array([0.0, 0.0, np.deg2rad(350)]))
    commands.push(command.CartesianPosition(geom.Transform3D(held.translation, held.rotation * away)))
    next(loop)

    turn = (held.rotation.inv * _pose_of(arm.poses[-1]).rotation).as_rotvec
    assert np.linalg.norm(turn) == pytest.approx(trossen_driver._MAX_STEP_RAD, abs=1e-6)
    assert turn[2] < 0  # ten degrees back, not three hundred and fifty forward


def test_the_firmware_is_asked_to_check_the_path_before_it_starts_one():
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    commands.push(command.CartesianPosition(HELD_POSE))
    next(loop)

    assert arm.checked_samples[-1] == trossen_driver._TRAJECTORY_CHECK_SAMPLES


def test_the_fingers_take_their_own_call_while_the_arm_holds_a_pose():
    """A Cartesian goal names the arm alone, so one goal vector cannot carry both."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    grip = ManualCommandReceiver()
    driver.commands._bind(commands)
    driver.target_grip._bind(grip)
    next(loop)

    commands.push(command.CartesianPosition(HELD_POSE))
    next(loop)
    poses = len(arm.poses)

    grip.push(0.0)
    next(loop)

    assert arm.gripper_goals[-1] == pytest.approx(GRIP_TRAVEL_M)
    assert len(arm.poses) == poses  # and the arm was not asked to plan its path again


def test_a_cartesian_move_nobody_can_judge_the_arrival_of_is_refused(world):
    """Arrival is judged from the joints, and a pose does not say which joints reach it."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)

    answer = caller(command.CartesianPosition(HELD_POSE))
    next(loop)

    with pytest.raises(NotImplementedError):
        answer.result()
