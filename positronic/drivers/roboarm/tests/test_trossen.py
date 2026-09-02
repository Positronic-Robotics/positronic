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
# Mid-range on every joint. The arm rests on the lower limit of joints 1 and 2, where Cartesian targets have
# no solution in half the directions, so every Cartesian test starts from here instead.
HOME = np.array([0.0, 1.571, 1.178, 0.0, 0.0, 0.0])
ARM = trossen_driver._ARM_JOINTS
# What the wxai_v0 controller reports as the following error it allows, and what caps a streamed solution
TOLERANCE = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.4])


class FakeArm(trossen_driver._FakeTrossen):
    """The driver's fake with the link under the test's control.

    ``blocked`` holds the joints where they stand while the controller keeps streaming; ``frozen`` stops
    the stream, as a dropped link does. ``raises`` is what reading the arm raises, ``write_raises`` what
    a write raises, and ``velocities`` what the joints read as running at.
    """

    def __init__(self, position=None):
        super().__init__()
        if position is not None:
            self._position = np.asarray(position, dtype=np.float64)
        self.raises: Exception | None = None
        self.write_raises: Exception | None = None
        self.configure_raises: Exception | None = None
        self.velocities: np.ndarray | None = None
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
        out = super().get_robot_output()
        if self.velocities is not None:
            out.joint.arm.velocities = np.asarray(self.velocities, dtype=np.float64)
        return out

    def set_all_modes(self, mode) -> None:
        if self.write_raises is not None:
            raise self.write_raises
        super().set_all_modes(mode)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        if self.write_raises is not None:
            raise self.write_raises
        super().set_all_positions(goal_positions, goal_time, blocking)

    def set_gripper_position(self, goal_position, goal_time=2.0, blocking=True) -> None:
        if self.write_raises is not None:
            raise self.write_raises
        super().set_gripper_position(goal_position, goal_time, blocking)

    def _servo(self) -> None:
        if not self.blocked:
            super()._servo()


class SaggingArm(FakeArm):
    """The arm's fake, holding itself up with a following error, as the real one does.

    The joints settle a fixed distance short of every goal. The values are what the arm at the rig reads
    against what it was asked for at the nominal pose.
    """

    DROOP = np.array([0.0, 0.071, -0.064, -0.051, 0.0, 0.0])

    def _servo(self) -> None:
        super()._servo()
        if self.goals:
            settled = np.append(np.asarray(self.goals[-1])[:ARM] + SaggingArm.DROOP, self._position[ARM])
            self._position = self._position + 0.1 * (settled - self._position)


def _driven(arm: FakeArm, clock: MockClock | None = None, stop: StopFlag | None = None):
    """A driver over ``arm`` with its state recorded, and its loop ready to pump."""
    driver = trossen_driver.Robot('192.168.1.4', connect=lambda ip: arm)
    states = RecordingEmitter()
    driver.state._bind(states)
    return driver, states, driver.run(stop or StopFlag(), clock or MockClock())


def _settle(loop, ticks: int = 400) -> None:
    """Run the loop long enough for the arm to reach what it was last asked for.

    Every setpoint is held to what the joints may travel in one tick, so arriving takes many of them.
    """
    for _ in range(ticks):
        next(loop)


def _held(arm: FakeArm) -> np.ndarray:
    return np.asarray(arm.goals[-1][:ARM])


def _at_home(commands: ManualCommandReceiver, loop) -> None:
    commands.push(command.JointPosition(HOME))
    _settle(loop)


def test_the_link_is_written_only_once_something_has_asked_for_a_setpoint():
    """An arm already held where it stands is not written to again."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    next(loop)  # entering position mode writes the hold the servo starts from
    written = len(arm.goals) + len(arm.gripper_goals)
    for _ in range(3):  # and nothing has asked since
        next(loop)
    assert len(arm.goals) + len(arm.gripper_goals) == written

    grip.push(0.25)
    next(loop)
    assert len(arm.goals) + len(arm.gripper_goals) == written + 1


def test_an_open_grip_reaches_the_arm_as_the_joint_at_its_upper_limit():
    """positronic counts 1 as closed; the joint counts its lower limit as closed."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(0.0)
    next(loop)

    assert arm.gripper_goals[-1] == pytest.approx(GRIP_TRAVEL_M)


def test_a_closed_grip_reaches_the_arm_as_the_joint_at_its_lower_limit():
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(1.0)
    next(loop)

    assert arm.gripper_goals[-1] == pytest.approx(0.0)


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
    _settle(loop)

    np.testing.assert_allclose(_held(arm), JOGGED, atol=1e-6)


def test_a_setpoint_never_asks_a_joint_for_more_than_it_may_travel_in_a_tick():
    """Past its velocity limit the controller faults and drops the arm, so no goal may ask for that.

    The ramp also comes up to speed rather than starting at it, so the first steps are smaller still.
    """
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    commands.push(command.JointPosition(np.array([3.0, 3.0, 2.0, 1.5, 1.5, 3.0])))
    for _ in range(40):
        next(loop)

    per_tick = np.array([limit.velocity_max for limit in arm.get_joint_limits()[:ARM]])
    per_tick = per_tick * trossen_driver._COMMANDED_SHARE / trossen_driver._HZ
    steps = np.abs(np.diff(np.array([goal[:ARM] for goal in arm.goals]), axis=0))
    assert np.all(steps <= per_tick + 1e-9), steps.max(axis=0)


def test_a_joint_target_outside_the_range_is_clipped_to_it():
    """The second joint has no negative half, and a target below it is held at the limit, not refused."""
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
    _settle(loop)

    assert _held(arm)[1] == pytest.approx(0.0)


def test_a_streamed_command_the_arm_cannot_be_put_at_leaves_it_where_it_is(monkeypatch):
    """A command stream cannot end the run: the next command supersedes one that could not be applied.

    Every target is brought within a step of where the arm stands before it is solved, so one the solver
    cannot reach is rare enough that the solver is what stands in for it here.
    """
    arm = FakeArm()
    driver, _, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    where_it_is = _held(arm)

    monkeypatch.setattr(trossen_driver._Kinematics, 'ik', lambda self, target, current_q: None)
    commands.push(command.CartesianPosition(geom.Transform3D(np.array([3.0, 0.0, 0.2]))))
    next(loop)

    np.testing.assert_allclose(_held(arm), where_it_is, atol=1e-6)


def _reject_writes(arm: FakeArm, commands: ManualCommandReceiver, loop) -> None:
    """Break the command half of the link, and give the driver a setpoint that finds out."""
    arm.write_raises = trossen_driver.trossen_arm.RuntimeError('Broken pipe')
    commands.push(command.JointPosition(JOGGED))
    next(loop)


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


def test_a_setpoint_the_link_refuses_reads_error_and_stops_the_writes():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    next(loop)
    written = len(arm.goals)
    _reject_writes(arm, commands, loop)
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

    _reject_writes(arm, commands, loop)
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

    _reject_writes(arm, commands, loop)
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
    _settle(loop)
    where_it_is = np.asarray(arm.get_robot_output().joint.arm.positions)

    _reject_writes(arm, commands, loop)
    arm.write_raises = None
    clock.advance(trossen_driver._RECONNECT_AFTER_S + 0.01)
    next(loop)

    np.testing.assert_allclose(_held(arm), where_it_is, atol=1e-3)


def test_a_new_session_that_fails_is_tried_again_further_and_further_apart():
    """A fault the controller latches outlives a new session, so retrying at the same pace stalls the loop."""
    arm = FakeArm()
    clock = MockClock()
    driver, _, loop = _driven(arm, clock)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    _reject_writes(arm, commands, loop)
    arm.configure_raises = trossen_driver.trossen_arm.RuntimeError('Network is unreachable')
    clock.advance(trossen_driver._RECONNECT_AFTER_S + 0.01)
    for _ in range(20):  # many ticks inside one reconnect interval, one attempt between them
        next(loop)
    assert arm.attempts == 1

    clock.advance(trossen_driver._RECONNECT_EVERY_S + 0.01)
    for _ in range(5):
        next(loop)
    assert arm.attempts == 1  # the interval doubled after the first attempt failed

    clock.advance(trossen_driver._RECONNECT_EVERY_S + 0.01)
    next(loop)
    assert arm.attempts == 2
    assert arm.sessions == 1  # none of them took


def test_a_sync_move_answers_once_the_arm_reads_back_at_its_target(world):
    arm = FakeArm()
    clock = MockClock()
    driver, states, loop = _driven(arm, clock)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)

    answer = caller(command.JointPosition(JOGGED))
    for _ in range(400):
        if answer.done():
            break
        next(loop)
    answer.result()

    np.testing.assert_allclose(states.emitted[-1][1].q, JOGGED, atol=0.1)


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


def test_an_arm_running_past_a_joint_limit_is_left_alone_until_it_slows():
    """Past its limit the controller faults and drops the arm, so the driver stops driving before that."""
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    next(loop)

    arm.velocities = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 0.0])  # joint 4 stops at 9.4248 rad/s
    commands.push(command.JointPosition(JOGGED))
    for _ in range(5):
        next(loop)

    assert states.emitted[-1][1].status == RobotStatus.ERROR
    np.testing.assert_allclose(_held(arm), states.emitted[-1][1].q, atol=1e-3)

    arm.velocities = np.zeros(6)
    commands.push(command.JointPosition(JOGGED))
    _settle(loop)
    np.testing.assert_allclose(_held(arm), JOGGED, atol=1e-6)


# --- Cartesian, which the driver solves itself ---


def _ee(states: RecordingEmitter) -> geom.Transform3D:
    return states.emitted[-1][1].ee_pose


def test_the_pose_that_goes_out_is_the_one_the_joints_put_the_end_effector_at():
    """Measured on the arm at rest: joints all but zero put ``ee_site`` here."""
    arm = FakeArm()
    _, states, loop = _driven(arm)

    next(loop)

    np.testing.assert_allclose(_ee(states).translation, [0.2537, 0.0, 0.1635], atol=5e-4)


def test_a_streamed_cartesian_command_takes_the_end_effector_there():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    target = geom.Transform3D(_ee(states).translation + np.array([0.0, 0.04, -0.03]), _ee(states).rotation)

    for _ in range(60):  # the target is held, as a teleoperator holds one
        commands.push(command.CartesianPosition(target))
        _settle(loop, 10)

    np.testing.assert_allclose(_ee(states).translation, target.translation, atol=2e-3)


def test_a_cartesian_delta_composes_onto_the_pose_the_joints_put_the_arm_at():
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    started = _ee(states).translation.copy()

    commands.push(command.CartesianDelta(geom.Transform3D(np.array([0.0, 0.01, 0.0]))))
    _settle(loop)

    np.testing.assert_allclose(_ee(states).translation, started + np.array([0.0, 0.01, 0.0]), atol=2e-3)


def test_a_cartesian_target_out_of_reach_is_solved_one_step_at_a_time():
    """A teleoperator reaching past the arm asks for a target that runs away from it."""
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    started = _ee(states).translation.copy()

    commands.push(command.CartesianPosition(geom.Transform3D(started + np.array([0.0, 5.0, 0.0]))))
    next(loop)

    step = np.linalg.norm(_ee(states).translation - started)
    assert step < trossen_driver._MAX_STEP_M + 1e-6


def test_a_streamed_turn_is_walked_the_whole_way_by_an_arm_that_sags():
    """A teleoperator turns the end effector over many ticks, and the setpoint moves a fraction of a degree
    in each: the turn arrives however many that takes, on an arm holding itself up with a following error."""
    arm = SaggingArm(position=np.append(HOME, 0.0))
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    here = _ee(states)
    asked = geom.Transform3D(
        here.translation, geom.Rotation.from_rotvec(np.array([0.0, np.radians(30), 0.0])) * here.rotation
    )

    for _ in range(1500):  # a teleoperator's stream: the same target, tick after tick
        commands.push(command.CartesianPosition(asked))
        next(loop)

    turned = np.degrees(np.linalg.norm((here.rotation.inv * _ee(states).rotation).as_rotvec))
    assert turned > 25, turned


def test_a_sync_move_to_a_pose_is_answered_like_any_other(world):
    """A pose says which joints reach it once the driver solves for them, so a caller may wait on one.

    Arrival is judged from the joints, within the following error the controller says it allows. The arm
    carries that error holding itself up, so the tolerance cannot be tighter — and at half a metre of
    reach it is centimetres at the end effector, which is why the pose is only checked that closely.
    """
    arm = FakeArm()
    driver, states, loop = _driven(arm)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    # Further than a streamed target is ever paced to, so a move that only stepped once would fall short
    target = geom.Transform3D(_ee(states).translation + np.array([0.0, 0.05, -0.04]), _ee(states).rotation)

    answer = caller(command.CartesianPosition(target))
    for _ in range(400):
        if answer.done():
            break
        next(loop)
    answer.result()

    np.testing.assert_allclose(_ee(states).translation, target.translation, atol=6e-2)


def test_a_pose_far_from_where_the_arm_stands_is_solved_only_when_it_may_change_shape():
    """The same pose is reachable with the arm in more than one shape, and moving between them swings it."""
    kin = trossen_driver._Kinematics()
    here = kin.fk(HOME)
    far = geom.Transform3D(here.translation + np.array([-0.15, 0.15, -0.1]), here.rotation)

    assert kin.ik(far, HOME, max_jump=TOLERANCE) is None
    assert kin.ik(far, HOME) is not None


def test_a_target_the_arm_has_drooped_away_from_is_still_solved():
    """A teleoperator's target starts at what the arm reads, which stands off from what it was asked for."""
    kin = trossen_driver._Kinematics()
    drooped = HOME + np.array([0.0, 0.08, -0.06, -0.05, 0.0, 0.0])  # the following error measured on the arm

    assert kin.ik(kin.fk(drooped), HOME, max_jump=TOLERANCE) is not None


def test_a_stream_of_poses_the_arm_cannot_follow_says_so_once(caplog, monkeypatch):
    """The pose differs every tick and the fault that refuses it does not, so one complaint stands for all."""
    arm = FakeArm()
    clock = MockClock()
    driver, _, loop = _driven(arm, clock)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)
    _at_home(commands, loop)

    monkeypatch.setattr(trossen_driver._Kinematics, 'ik', lambda self, target, current_q, max_jump=None: None)
    for tick in range(5):
        commands.push(command.CartesianPosition(geom.Transform3D(np.array([0.4, 0.001 * tick, 0.2]))))
        next(loop)
        clock.advance(1.0 / trossen_driver._HZ)

    assert caplog.text.count('not applied') == 1


def test_an_arm_reading_outside_a_joint_range_says_so_and_is_driven_anyway(caplog):
    """The controller takes a margin past what it reports, so only it knows whether this one is too far."""
    arm = FakeArm(position=np.append(np.zeros(6), -0.0062))  # the gripper past its own zero

    _, states, loop = _driven(arm)
    next(loop)

    assert 'joint 6 reads' in caplog.text
    assert states.emitted[-1][1].status == RobotStatus.AVAILABLE
