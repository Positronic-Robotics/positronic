"""What the CRON driver commands the controller, and what a caller waiting on a move hears."""

import logging
import math

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm.cron import controller as cron_controller
from positronic.drivers.roboarm.cron import driver as cron
from positronic.drivers.roboarm.cron.controller import JOINTS, MotionMode, PowerState
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

TICK = 1.0 / cron._HZ


class FakeRctl:
    """The vendor handle with an arm behind it: a velocity command moves the joints for one tick.

    The pose is the first three joints and the sixth joint as the yaw. ``compute_ik`` returns the solution
    next to the seed, and one a turn away from it. A hold request takes effect after ``hold_lag`` reads of the
    motion mode.
    """

    def __init__(self):
        self.q = np.zeros(JOINTS)
        self.velocity = np.zeros(JOINTS)
        self.power = PowerState.OFF
        self.mode = MotionMode.NONE
        self.hold_lag = 0
        self._hold_in: int | None = None
        self.mode_at_power_off: MotionMode | None = None
        self.tool = np.zeros(6)
        # A safety stop latches FAILURE: a resume leaves it, an OFF request clears it once the stop is released
        self.safety_latched = False
        self.stop_pressed = False
        self.gravity = np.array([0.0, 0.0, 9.81])
        # The real controller can answer IK with no error and NaN solutions
        self.ik_nan_solutions = 0
        self.ik_only_nan = False
        self.error = False
        self.fail_power_run = False
        self.blocked = False
        self.moving = False
        self.waypoint: np.ndarray | None = None
        self.waypoint_speed = 0.0
        # The real planner travels slower than the speed it is asked for
        self.planner_share = 1.0
        self.calls: list[str] = []
        self.commanded: list[np.ndarray] = []

    def _call(self, name: str) -> None:
        self.calls.append(name)

    def connect(self, ip):
        self._call('connect')

    def disconnect(self):
        self._call('disconnect')

    def get_cycle_count(self):
        return 1

    def sync_robot_model(self):
        self._call('sync_robot_model')

    def is_model_valid(self):
        return True

    def get_tool(self):
        return self.tool.copy()

    def get_gravity(self):
        return self.gravity.copy()

    def take_control(self):
        self._call('take_control')

    def exr_run(self, project_name):
        # The real controller logs "Robot is in error state, program execution is prohibited" and returns
        self._call(f'exr_run {project_name}' + (' refused' if self.error else ''))

    def release_control(self):
        self._call('release_control')

    def cmd_hold(self):
        self._call('hold')
        self._hold_in = self.hold_lag

    def get_motion_mode(self):
        if self._hold_in == 0:
            self.mode, self._hold_in = MotionMode.HOLD, None
        elif self._hold_in is not None:
            self._hold_in -= 1
        return int(self.mode)

    def cmd_resume(self):
        self._call('resume')
        if not self.safety_latched:
            self.error = False

    def cmd_power_standby(self):
        self._call('standby')
        self.power = PowerState.STANDBY

    def cmd_power_run(self):
        self._call('run')
        self.power = PowerState.FAILURE if self.fail_power_run else PowerState.RUN
        if self.power is PowerState.RUN:
            self.mode = MotionMode.HOLD

    def cmd_power_off(self):
        self._call('off')
        self.mode_at_power_off = MotionMode(self.get_motion_mode())
        self.power = PowerState.OFF
        if self.safety_latched and not self.stop_pressed:
            self.safety_latched = self.error = False

    def get_state(self):
        return int(self.power)

    def is_error(self):
        return self.error

    def get_errors_hw(self):
        return 'hardware errors'

    def get_errors_mm(self):
        return 'motion errors'

    def cmd_joint_direct_vel(self, velocities):
        self.velocity = np.array(velocities, dtype=np.float64)
        self.commanded.append(self.velocity.copy())
        if self.power is PowerState.RUN:
            self.mode = MotionMode.JOINT_VEL
            self.q = self.q + self.velocity * TICK

    def cmd_add_waypoint_joint(self, pose, limits, rblend, waypoint_id, main=True):
        self._call('waypoint')
        self.waypoint = np.array(pose, dtype=np.float64)
        self.waypoint_speed = limits[0]

    def cmd_move_run(self, main=True):
        self._call('move_run')
        self.moving = True

    def cmd_move_pause(self, acceleration):
        self._call('move_pause')
        self.moving = False

    def cmd_drop_waypoints(self, main):
        self._call('drop_waypoints')
        self.waypoint = None
        self.moving = False

    def is_moving(self, main=True):
        return self.moving

    def get_joint_angles(self):
        if self.moving and not self.blocked:
            reach = self.waypoint_speed * self.planner_share * TICK
            self.q = self.q + np.clip(self.waypoint - self.q, -reach, reach)
            self.moving = bool(np.max(np.abs(self.waypoint - self.q)) > 1e-9)
        return self.q.copy()

    def get_joint_vels(self):
        return self.velocity.copy()

    def get_cart_pose(self):
        return self.compute_fk(self.q)['pose']

    def rpy2quat(self, rpy):
        return [0.0, 0.0, math.sin(rpy[2] / 2), math.cos(rpy[2] / 2)]

    def quat2rpy(self, quat):
        return [0.0, 0.0, 2 * math.atan2(quat[2], quat[3])]

    def compute_fk(self, q):
        return {'err': False, 'err_code': 'NONE', 'pose': np.array([q[0], q[1], q[2], 0.0, 0.0, q[5]])}

    def compute_ik(self, pose, seed):
        if np.any(np.abs(pose[:3]) > 1.0):
            return {'err': True, 'err_code': 'NO_SOLUTION', 'solutions': []}
        near = np.array(seed, dtype=np.float64)
        near[:3], near[5] = pose[:3], pose[5]
        far = near.copy()
        far[0] += 2 * math.pi
        nans = [np.full(JOINTS, np.nan)] * self.ik_nan_solutions
        solutions = nans if self.ik_only_nan else [*nans, far, near]
        return {'err': False, 'err_code': 'NONE', 'solutions': solutions}


@pytest.fixture
def rctl(monkeypatch):
    fake = FakeRctl()
    monkeypatch.setattr(cron_controller, 'rctl', fake)
    monkeypatch.setattr(cron_controller, '_STARTUP_S', 0.0)
    return fake


def _joints(first: float) -> np.ndarray:
    return np.array([first, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def driven(rctl):
    """Builds a driver with its state recorded, and its loop ready to pump.

    Each loop closes while ``rctl`` is still in place. A loop left to the garbage collector shuts down
    whichever fake is in place then, and that fake can belong to a later test.
    """
    loops = []

    def build(factor: float = 1.0, stop: StopFlag | None = None, tool_length: float = 0.0, **options):
        driver = cron.Robot('controller', relative_dynamics_factor=factor, tool_length=tool_length, **options)
        states = RecordingEmitter()
        clock = MockClock()
        driver.state._bind(states, clock=clock)
        commands = ManualCommandReceiver()
        driver.commands._bind(commands)
        loop = driver.run(stop or StopFlag(), clock)
        loops.append(loop)
        return driver, states, commands, clock, loop

    yield build
    for loop in loops:
        loop.close()


def _pump(loop, clock: MockClock, ticks: int) -> None:
    for _ in range(ticks):
        clock.advance(TICK)
        next(loop)


def _mover(world: pimm.World, driver: cron.Robot) -> pimm.calls.Caller[command.CommandType, None]:
    """A caller on ``driver.sync_move``, for a test that pumps its generator rather than running a World."""
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


def test_the_arm_is_powered_straight_to_run_with_control_taken_first(rctl, driven):
    """RUN from STANDBY fails on an arm whose multi-turn encoder asks for a preset, and RUN from OFF passes."""
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert rctl.calls == ['connect', 'sync_robot_model', 'take_control', 'run']


def test_a_target_beyond_the_reach_is_held_on_the_sphere_of_the_reach(rctl, driven):
    """Near full reach the joints turn fast for a small move of the tool, and the controller stops the arm."""
    _, _, commands, clock, loop = driven(max_reach=0.6)
    far = geom.Transform3D(translation=np.array([0.8, 0.6, 0.0]))
    commands.push(command.CartesianPosition(far))

    _pump(loop, clock, 800)

    np.testing.assert_allclose(rctl.q[:3], [0.48, 0.36, 0.0], atol=0.01)


def test_a_project_of_the_controller_is_started_once_control_is_taken(rctl, driven):
    """The controller starts a project only for a client that has control, such as the bridge to the gripper."""
    _, _, _, clock, loop = driven(project='gripper_bridge')

    _pump(loop, clock, 1)

    assert rctl.calls == ['connect', 'sync_robot_model', 'take_control', 'exr_run gripper_bridge', 'run']


def test_a_project_of_the_controller_is_started_after_a_safety_stop_is_cleared(rctl, driven):
    """The controller refuses to start a project while it holds an error, as after a press of the stop button."""
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _, _, _, clock, loop = driven(project='gripper_bridge')

    _pump(loop, clock, 1)

    connected = ['connect', 'sync_robot_model', 'take_control']
    assert rctl.calls == [*connected, 'resume', 'off', 'exr_run gripper_bridge', 'run']


def test_an_error_left_by_an_earlier_session_is_cleared_before_the_arm_is_powered(rctl, driven):
    rctl.error = True
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert rctl.calls == ['connect', 'sync_robot_model', 'take_control', 'resume', 'run']


def test_a_safety_stop_latched_after_the_stop_button_is_released_is_cleared_and_the_arm_is_powered(rctl, driven):
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert rctl.calls == ['connect', 'sync_robot_model', 'take_control', 'resume', 'off', 'run']
    assert rctl.power is PowerState.RUN


def test_a_safety_stop_with_the_stop_button_still_pressed_fails_the_run_and_is_not_powered(rctl, driven, monkeypatch):
    monkeypatch.setattr(cron_controller, '_CONNECT_TIMEOUT_S', 0.1)
    rctl.error, rctl.safety_latched, rctl.stop_pressed, rctl.power = True, True, True, PowerState.FAILURE
    _, _, _, clock, loop = driven()

    with pytest.raises(TimeoutError, match='stop button'):
        _pump(loop, clock, 1)

    assert 'standby' not in rctl.calls and 'run' not in rctl.calls


def test_the_arm_is_not_commanded_until_something_asks_for_a_target(rctl, driven):
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_a_streamed_target_is_approached_within_the_speed_limit(rctl, driven):
    _, _, commands, clock, loop = driven(factor=0.2)
    commands.push(command.JointPosition(_joints(0.5)))

    _pump(loop, clock, 200)

    speeds = np.abs(np.array(rctl.commanded))
    assert speeds.max() <= 0.2 * cron._MAX_JOINT_SPEED + 1e-9
    assert rctl.commanded[-1][0] > 0.1, 'the arm should be near its speed limit by now'


def test_an_arm_at_full_speed_stops_at_a_streamed_target_without_overshoot_at_a_high_gain(rctl, driven, monkeypatch):
    # Above 2 * max_accel / max_speed (8 1/s) the gain alone asks to brake harder than max_accel allows
    monkeypatch.setattr(cron, '_GAIN', 20.0)
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(1.0)))
    passed = 0.0

    for _ in range(800):
        _pump(loop, clock, 1)
        passed = max(passed, rctl.q[0])

    assert passed <= 1.0 + 0.005, f'overshoot {passed - 1.0:.3f} rad'
    np.testing.assert_allclose(rctl.q, _joints(1.0), atol=0.01)


def test_the_arm_reaches_a_streamed_target(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.3)))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q, _joints(0.3), atol=0.01)


def test_a_sync_move_answers_once_the_arm_reads_back_at_the_target(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    for _ in range(1000):
        if answer.done():
            break
        _pump(loop, clock, 1)

    assert answer.result() is None
    np.testing.assert_allclose(rctl.q, _joints(0.3), atol=cron._ARRIVED_TOL)


def _pump_until(loop, clock: MockClock, answer, ticks: int = 2000) -> None:
    for _ in range(ticks):
        if answer.done():
            return
        _pump(loop, clock, 1)


def test_a_move_that_a_caller_waits_for_is_planned_by_the_controller_and_not_streamed(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    _pump_until(loop, clock, answer)

    assert answer.result() is None
    assert rctl.calls[-2:] == ['waypoint', 'move_run']
    assert rctl.commanded == []


def test_a_move_stops_the_stream_before_the_controller_plans_it(world, rctl, driven):
    driver, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 20)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))

    _pump(loop, clock, 2)

    np.testing.assert_array_equal(rctl.commanded[-1], np.zeros(JOINTS))
    assert rctl.calls[-2:] == ['waypoint', 'move_run']
    assert not answer.done()


def test_a_streamed_target_after_a_move_is_tracked_by_velocity_again(world, rctl, driven):
    driver, _, commands, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, answer)
    commands.push(command.JointPosition(_joints(0.6)))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q, _joints(0.6), atol=0.01)
    assert rctl.commanded


def test_a_move_the_arm_does_not_follow_is_given_up_and_stopped(world, rctl, driven):
    rctl.blocked = True
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    _pump_until(loop, clock, answer)

    with pytest.raises(TimeoutError):
        answer.result()
    assert rctl.calls[-2:] == ['move_pause', 'drop_waypoints']


def test_a_move_asked_for_while_the_controller_holds_an_error_clears_it_and_then_is_planned(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error = True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    _pump_until(loop, clock, answer)

    assert answer.result() is None
    moves = [call for call in rctl.calls if call in ('resume', 'waypoint', 'move_run')]
    assert moves[-3:] == ['resume', 'waypoint', 'move_run']
    np.testing.assert_allclose(rctl.q, _joints(0.3), atol=cron._ARRIVED_TOL)


def test_a_move_asked_for_while_the_error_stays_is_not_planned_and_is_given_up(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    _pump_until(loop, clock, answer, ticks=10000)
    rctl.safety_latched, rctl.error = False, False
    _pump(loop, clock, 20)

    with pytest.raises(TimeoutError):
        answer.result()
    assert 'waypoint' not in rctl.calls, 'a move given up started once the error cleared'


def test_each_change_of_the_error_state_is_logged_with_the_address_of_the_controller(rctl, driven, caplog):
    _, _, _, clock, loop = driven()
    _pump(loop, clock, 2)

    with caplog.at_level(logging.INFO, logger=cron.__name__):
        rctl.error = True
        _pump(loop, clock, 5)
        rctl.error = False
        _pump(loop, clock, 5)

    lines = [record.getMessage() for record in caplog.records]
    assert lines == [
        'controller: RUN, error hardware errors motion errors, joints [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] deg',
        'controller: RUN, no error',
    ]


def test_a_long_move_that_the_planner_travels_slower_than_asked_is_not_given_up(world, rctl, driven):
    rctl.planner_share = 0.7
    driver, _, _, clock, loop = driven(factor=0.1)
    answer = _mover(world, driver)(command.JointPosition(_joints(3.0)))

    _pump_until(loop, clock, answer, ticks=20000)

    assert answer.result() is None


def test_a_move_in_flight_reads_busy(world, rctl, driven):
    driver, states, _, clock, loop = driven(factor=0.2)
    answer = _mover(world, driver)(command.JointPosition(_joints(1.0)))

    _pump(loop, clock, 5)

    assert states.emitted[-1][1].status is RobotStatus.BUSY
    assert not answer.done()


def test_a_cartesian_target_is_reached_through_the_solution_next_to_the_joints(rctl, driven):
    _, _, commands, clock, loop = driven()
    pose = geom.Transform3D(translation=np.array([0.1, 0.2, 0.3]), rotation=geom.Rotation.identity)
    commands.push(command.CartesianPosition(pose))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q[:3], [0.1, 0.2, 0.3], atol=0.01)


def test_the_pose_on_the_wire_is_the_tool_point_along_the_flange_axis(rctl, driven):
    rctl.q = np.array([0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2])
    _, states, _, clock, loop = driven(tool_length=0.195)

    _pump(loop, clock, 1)

    np.testing.assert_allclose(states.emitted[-1][1].ee_pose.translation, [0.1, 0.2, 0.495], atol=1e-6)


def test_a_cartesian_target_is_reached_by_the_tool_point_and_not_the_flange(rctl, driven):
    _, _, commands, clock, loop = driven(tool_length=0.195)
    pose = geom.Transform3D(translation=np.array([0.1, 0.2, 0.5]), rotation=geom.Rotation.identity)
    commands.push(command.CartesianPosition(pose))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q[:3], [0.1, 0.2, 0.305], atol=0.01)


def test_a_controller_with_its_own_tool_is_refused_before_control_is_taken(rctl, driven):
    rctl.tool = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    _, _, _, clock, loop = driven(tool_length=0.195)

    with pytest.raises(RuntimeError, match='tool'):
        _pump(loop, clock, 1)

    assert 'take_control' not in rctl.calls
    assert rctl.calls[-1] == 'disconnect'


_TILTED_UP = np.array([0.0, 1.0, 1.0]) / math.sqrt(2)  # a base mounted 45 deg over about its x axis


def test_a_point_up_from_a_tilted_base_is_straight_above_it_on_the_wire(rctl, driven):
    rctl.gravity = 9.81 * _TILTED_UP
    rctl.q = np.array([*(0.3 * _TILTED_UP), 0.0, 0.0, 0.0])
    _, states, _, clock, loop = driven()

    _pump(loop, clock, 1)

    np.testing.assert_allclose(states.emitted[-1][1].ee_pose.translation, [0.0, 0.0, 0.3], atol=1e-6)


def test_a_cartesian_target_above_a_tilted_base_is_reached_along_its_up_direction(rctl, driven):
    rctl.gravity = 9.81 * _TILTED_UP
    _, _, commands, clock, loop = driven()
    level_turn = geom.Rotation.from_rotvec(np.array([math.pi / 4, 0.0, 0.0]))
    commands.push(command.CartesianPosition(geom.Transform3D(np.array([0.0, 0.0, 0.3]), level_turn)))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q[:3], 0.3 * _TILTED_UP, atol=0.01)


@pytest.mark.parametrize('up', [[0, 0, 9.81], [0, 6.94, 6.94], [0, -6.94, 6.94], [3.0, 1.0, -2.0], [0, 0, -9.81]])
def test_level_puts_the_up_direction_on_the_z_axis(up):
    up = np.array(up, dtype=float)

    np.testing.assert_allclose(cron_controller.level(up)(up / np.linalg.norm(up)), [0.0, 0.0, 1.0], atol=1e-9)


def test_a_controller_with_no_gravity_is_refused_before_control_is_taken(rctl, driven):
    rctl.gravity = np.zeros(3)
    _, _, _, clock, loop = driven()

    with pytest.raises(RuntimeError, match='gravity'):
        _pump(loop, clock, 1)

    assert 'take_control' not in rctl.calls


def test_a_nan_solution_that_the_controller_reports_with_no_error_is_not_picked(rctl, driven):
    rctl.ik_nan_solutions = 2
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.1, 0.2, 0.3]))))

    _pump(loop, clock, 600)

    np.testing.assert_allclose(rctl.q[:3], [0.1, 0.2, 0.3], atol=0.01)


def test_a_cartesian_target_with_only_nan_solutions_is_not_a_setpoint(rctl, driven):
    rctl.ik_nan_solutions, rctl.ik_only_nan = 8, True
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.1, 0.2, 0.3]))))

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_a_joint_target_that_is_not_finite_is_not_a_setpoint(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(np.array([0.1, np.nan, 0.0, 0.0, 0.0, 0.0])))

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_a_cartesian_target_out_of_reach_is_not_a_setpoint(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.full(3, 100.0))))

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_an_arm_that_leaves_run_is_not_commanded_and_does_not_resume_its_target(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 20)
    rctl.power = PowerState.STANDBY
    _pump(loop, clock, 5)
    sent = len(rctl.commanded)

    rctl.power = PowerState.RUN
    _pump(loop, clock, 20)

    assert len(rctl.commanded) == sent


def test_an_error_of_the_controller_reads_as_error_on_the_wire(rctl, driven):
    _, states, _, clock, loop = driven()
    _pump(loop, clock, 2)

    rctl.error = True
    _pump(loop, clock, 2)

    assert states.emitted[-1][1].status is RobotStatus.ERROR


def test_leaving_the_run_stops_the_arm_powers_it_off_and_gives_control_back(rctl, driven):
    stop = StopFlag()
    _, _, commands, clock, loop = driven(stop=stop)
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 20)

    stop.stopped = True
    with pytest.raises(StopIteration):
        _pump(loop, clock, 1)

    np.testing.assert_array_equal(rctl.commanded[-1], np.zeros(JOINTS))
    assert rctl.calls[-4:] == ['hold', 'off', 'release_control', 'disconnect']


def test_a_streamed_arm_is_powered_off_only_once_the_controller_holds(rctl, driven):
    rctl.hold_lag = 2
    stop = StopFlag()
    _, _, commands, clock, loop = driven(stop=stop)
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 20)

    stop.stopped = True
    with pytest.raises(StopIteration):
        _pump(loop, clock, 1)

    assert rctl.mode_at_power_off is MotionMode.HOLD


def test_an_arm_that_fails_to_power_up_is_left_off_and_the_run_fails(rctl, driven):
    rctl.fail_power_run = True
    _, _, _, clock, loop = driven()

    with pytest.raises(RuntimeError, match=r'the controller \S+ failed'):
        _pump(loop, clock, 1)

    assert rctl.power is PowerState.OFF
    assert rctl.calls[-2:] == ['release_control', 'disconnect']
