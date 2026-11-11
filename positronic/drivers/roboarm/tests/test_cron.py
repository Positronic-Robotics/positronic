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
from positronic.drivers.roboarm.cron.controller import JOINTS, MotionMode, PowerState, to_pose
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import MoveRefused
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

TICK = 1.0 / cron._HZ
_WRIST_MAX = float(cron._WRIST_SCALE.max())


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
        # Where the planner of the controller starts a move that does not start from HOLD
        self.planner_start = np.zeros(JOINTS)
        self.hold_lag = 0
        self._hold_in: int | None = None
        self.mode_at_power_off: MotionMode | None = None
        self.mode_at_release: MotionMode | None = None
        self.tool = np.zeros(6)
        # A safety stop latches FAILURE: a resume leaves it, an OFF request clears it once the stop is released
        self.safety_latched = False
        self.stop_pressed = False
        self.gravity = np.array([0.0, 0.0, 9.81])
        # The real controller can answer IK with no error and NaN solutions
        self.ik_nan_solutions = 0
        self.ik_only_nan = False
        # A joint that moves nothing, as at a pose where the jacobian loses a rank
        self.dead_joint: int | None = None
        # The collision model: the clearance of the arm is how far the first joint is under this, if set
        self.wall: float | None = None
        # A second wall, under the first joint
        self.floor: float | None = None
        # A thin part of the arm: the first joint collides between these two angles, and is clear on both sides
        self.band: tuple[float, float] | None = None
        # The model refuses forward kinematics, as it can for a pose
        self.fk_fails = False
        self.tool_capsules: list[dict] = []
        self.error = False
        self.fail_power_run = False
        # Reads of the state that RUN takes after a request, as the real controller takes some seconds
        self.run_lag = 0
        self._run_in: int | None = None
        # Reads of the state that OFF takes after a request; the error flag clears at once
        self.off_lag = 0
        # A request for RUN that stops in STANDBY
        self.run_stalls = False
        self._off_in: int | None = None
        # A joint that a velocity does not move
        self.stuck_joint: int | None = None
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
        self.mode_at_release = MotionMode(self.get_motion_mode())

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
        self._run_in = self.run_lag
        self._update_power()

    def _update_power(self):
        if self._off_in is not None:
            if self._off_in > 0:
                self._off_in -= 1
            else:
                self._off_in, self.power = None, PowerState.OFF
        if self._run_in is None:
            return
        if self._run_in > 0:
            self._run_in -= 1
            return
        self._run_in = None
        if self.run_stalls:
            self.power = PowerState.STANDBY
            return
        self.power = PowerState.FAILURE if self.fail_power_run else PowerState.RUN
        if self.power is PowerState.RUN:
            self.mode = MotionMode.HOLD

    def cmd_power_off(self):
        self._call('off')
        self.mode_at_power_off = MotionMode(self.get_motion_mode())
        self._off_in = self.off_lag
        self._update_power()
        if self.safety_latched and not self.stop_pressed:
            self.safety_latched = self.error = False

    def get_state(self):
        self._update_power()
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
        if self.mode is MotionMode.MOVE and self.moving:
            # The real controller stops the arm with JOINT_VEL: TIMEOUT for a stream that starts while a move runs
            self.error, self.moving = True, False
            return
        if self.power is PowerState.RUN:
            self.mode = MotionMode.JOINT_VEL
            moved = self.velocity * TICK
            if self.stuck_joint is not None:
                moved[self.stuck_joint] = 0.0
            self.q = self.q + moved

    def cmd_add_waypoint_joint(self, pose, limits, rblend, waypoint_id, main=True):
        self._call('waypoint')
        self.waypoint = np.array(pose, dtype=np.float64)
        self.waypoint_speed = limits[0]

    def cmd_move_run(self, main=True):
        self._call('move_run')
        # The planner starts at the joints only from HOLD; else it starts at the end of its last move
        start = self.q if self.mode in (MotionMode.HOLD, MotionMode.NONE) else self.planner_start
        if np.linalg.norm(self.q - start) > 0.034907:
            self.error, self.moving = True, False
            return
        self.planner_start = self.waypoint.copy() if self.waypoint is not None else self.q.copy()
        self.mode = MotionMode.MOVE
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
            assert self.waypoint is not None, 'a move runs only to a waypoint'
            reach = self.waypoint_speed * self.planner_share * TICK
            self.q = self.q + np.clip(self.waypoint - self.q, -reach, reach)
            self.moving = bool(np.max(np.abs(self.waypoint - self.q)) > 1e-9)
        return self.q.copy()

    def get_joint_vels(self):
        return self.velocity.copy()

    def get_cart_pose(self):
        q = self.q.copy()
        return np.array([q[0], q[1], q[2], q[4], 0.0, q[5]])

    def rpy2quat(self, rpy):
        return list(geom.Rotation.from_euler(rpy).as_quat_xyzw)

    def quat2rpy(self, quat):
        return list(geom.Rotation.from_quat_xyzw(list(quat)).as_euler)

    def compute_fk(self, q):
        if self.fk_fails:
            return {'err': True, 'err_code': 'NO_SOLUTION', 'pose': []}
        q = np.array(q, dtype=np.float64)
        if self.dead_joint is not None:
            q[self.dead_joint] = 0.0
        return {'err': False, 'err_code': 'NONE', 'pose': np.array([q[0], q[1], q[2], q[4], 0.0, q[5]])}

    def model_set_tool_capsules(self, capsules):
        self.tool_capsules = [dict(capsule) for capsule in capsules]

    def model_is_collision(self, q):
        clearance = min(
            1.0 if self.wall is None else self.wall - q[0], 1.0 if self.floor is None else q[0] - self.floor
        )
        if self.band is not None:
            low, high = self.band
            clearance = min(clearance, 0.0 if low <= q[0] <= high else min(abs(q[0] - low), abs(q[0] - high)))
        return {'lim': 0, 'coll': int(clearance <= 0), 'min_dist': max(clearance, 0.0), 'threshold': 0.005}

    def jacob(self, q, in_tcp):
        """As the controller gives it: the flange in the base frame, rows vx, vy, vz, wx, wy, wz."""
        start, step, columns = to_pose(self.compute_fk(q)['pose']), 1e-7, []
        for joint in range(JOINTS):
            moved = to_pose(self.compute_fk(np.asarray(q) + step * np.eye(JOINTS)[joint])['pose'])
            turn = (moved.rotation * start.rotation.inv).as_rotvec
            columns.append(np.concatenate([(moved.translation - start.translation) / step, turn / step]))
        return np.stack(columns, axis=1)

    def compute_ik(self, pose, seed):
        if np.any(np.abs(pose[:3]) > 1.0):
            return {'err': True, 'err_code': 'NO_SOLUTION', 'solutions': []}
        near = np.array(seed, dtype=np.float64)
        near[:3], near[4], near[5] = pose[:3], pose[3], pose[5]
        far = near.copy()
        far[0] += 2 * math.pi
        nans = [np.full(JOINTS, np.nan)] * self.ik_nan_solutions
        solutions = nans if self.ik_only_nan else [*nans, far, near]
        return {'err': False, 'err_code': 'NONE', 'solutions': solutions}


@pytest.fixture
def rctl(monkeypatch):
    fake = FakeRctl()
    monkeypatch.setattr(cron_controller, 'rctl', fake)
    monkeypatch.setattr(cron_controller, 'STARTUP_S', 0.0)
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


def _called_in_order(calls: list[str], *names: str) -> bool:
    """Whether each of ``names`` is among ``calls``, the first of each in this order."""
    return all(name in calls for name in names) and sorted(map(calls.index, names)) == list(map(calls.index, names))


def _mover(world: pimm.World, driver: cron.Robot) -> pimm.calls.Caller[command.CommandType, None]:
    """A caller on ``driver.sync_move``, for a test that pumps its generator rather than running a World."""
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


def test_the_arm_is_powered_straight_to_run_with_control_taken_first(rctl, driven):
    """RUN from STANDBY fails on an arm whose multi-turn encoder asks for a preset, and RUN from OFF passes."""
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'run') and 'standby' not in rctl.calls
    assert rctl.power is PowerState.RUN


def test_a_planned_move_after_a_stream_starts_from_the_joints_that_the_arm_has(world, rctl, driven):
    """After a velocity stream the planner starts at the end of its last move, unless the move starts from HOLD."""
    driver, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, 600)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))

    _pump_until(loop, clock, answer)

    assert answer.result() is None
    assert not rctl.error
    np.testing.assert_allclose(rctl.q, _joints(0.0), atol=cron._ARRIVED_TOL)


def test_a_stream_after_a_planned_move_starts_once_the_planner_stops(world, rctl, driven):
    """The planner still finishes a move when the arm reads within the arrival tolerance."""
    driver, _, commands, clock, loop = driven(factor=0.1)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, answer)
    answer.result()

    commands.push(command.JointPosition(_joints(0.0)))
    _pump(loop, clock, 3000)

    assert not rctl.error
    np.testing.assert_allclose(rctl.q, _joints(0.0), atol=cron._ARRIVED_TOL)


def test_each_project_of_the_controller_is_started_once_control_is_taken(rctl, driven):
    """The controller starts a project only for a client that has control, such as the bridge to the gripper."""
    _, _, _, clock, loop = driven(projects=['gripper_bridge', 'logger'])

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'exr_run gripper_bridge', 'run')
    assert _called_in_order(rctl.calls, 'take_control', 'exr_run logger', 'run')


def test_a_project_of_the_controller_is_started_after_a_safety_stop_is_cleared(rctl, driven):
    """The controller refuses to start a project while it holds an error, as after a press of the stop button."""
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _, _, _, clock, loop = driven(projects=['gripper_bridge'])

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'resume', 'off', 'exr_run gripper_bridge', 'run')
    assert 'exr_run gripper_bridge refused' not in rctl.calls


def test_a_safety_stop_whose_error_clears_before_the_arm_is_off_is_powered_once_it_is_off(rctl, driven):
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    rctl.off_lag = 3
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'off', 'run')
    assert rctl.power is PowerState.RUN


def test_an_arm_in_standby_at_start_is_powered_to_run_through_off(rctl, driven):
    """RUN from STANDBY failed on an arm whose multi-turn encoder asks for a preset, and RUN from OFF passed."""
    rctl.power = PowerState.STANDBY
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'off', 'run')
    assert rctl.power is PowerState.RUN


def test_a_failure_left_with_no_error_flag_is_cleared_before_the_arm_is_powered(rctl, driven):
    """An earlier client cleared the error flag, and left before the arm was off."""
    rctl.power = PowerState.FAILURE
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'off', 'run')
    assert rctl.power is PowerState.RUN


def test_a_failure_that_takes_longer_than_the_error_flag_to_leave_is_given_the_power_timeout(rctl, driven, monkeypatch):
    monkeypatch.setattr(cron_controller, '_CONNECT_TIMEOUT_S', 0.1)
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    rctl.off_lag = 10  # reads at 0.05 s: longer than the wait for the error flag
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert rctl.power is PowerState.RUN


def test_an_error_left_by_an_earlier_session_is_cleared_before_the_arm_is_powered(rctl, driven):
    rctl.error = True
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'resume', 'run')
    assert rctl.power is PowerState.RUN


def test_a_safety_stop_latched_after_the_stop_button_is_released_is_cleared_and_the_arm_is_powered(rctl, driven):
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _, _, _, clock, loop = driven()

    _pump(loop, clock, 1)

    assert _called_in_order(rctl.calls, 'take_control', 'resume', 'off', 'run')
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


def test_a_move_brakes_the_stream_to_a_stop_before_the_controller_plans_it(world, rctl, driven):
    """A jump of the velocity to zero stopped the arm on a following error."""
    driver, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(2.0)))
    _pump(loop, clock, 100)
    streamed = len(rctl.commanded)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))

    for _ in range(200):
        _pump(loop, clock, 1)
        if 'move_run' in rctl.calls:
            break

    braking = np.array(rctl.commanded[streamed - 1 :])
    assert len(braking) > 2, 'the stream should brake over a number of ticks'
    assert np.max(np.abs(np.diff(braking, axis=0))) <= cron._MAX_JOINT_ACCEL * _WRIST_MAX * TICK + 1e-9
    np.testing.assert_array_equal(braking[-1], np.zeros(JOINTS))
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


def test_a_move_asked_for_while_the_error_stays_is_not_planned_and_is_refused(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))

    _pump_until(loop, clock, answer, ticks=10000)
    rctl.safety_latched, rctl.error = False, False
    _pump(loop, clock, 20)

    with pytest.raises(MoveRefused, match='motion errors') as refused:
        answer.result()
    assert refused.value.moved_rad == 0.0
    assert 'waypoint' not in rctl.calls, 'a move refused started once the error cleared'


def test_an_error_in_a_stream_is_logged_with_how_far_each_joint_lags_its_commanded_travel(rctl, driven, caplog):
    """The controller does not name the joint of a following error."""
    rctl.stuck_joint = 4
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(np.full(JOINTS, 1.0)))
    _pump(loop, clock, 50)

    with caplog.at_level(logging.ERROR, logger=cron.__name__):
        rctl.error = True
        _pump(loop, clock, 1)

    lag = next(record.getMessage() for record in caplog.records if 'lag' in record.getMessage())
    degrees = [float(value) for value in lag.split('[')[1].split(']')[0].split(',')]
    assert degrees[4] > 1.0
    assert max(abs(value) for i, value in enumerate(degrees) if i != 4) < 0.1


def test_each_change_of_the_error_state_is_logged_with_the_address_of_the_controller(rctl, driven, caplog):
    _, _, _, clock, loop = driven()
    _pump(loop, clock, 2)

    with caplog.at_level(logging.INFO, logger=cron.__name__):
        rctl.error = True
        _pump(loop, clock, 5)
        rctl.error = False
        _pump(loop, clock, 5)

    error, cleared = [record.getMessage() for record in caplog.records]
    assert 'controller' in error and 'hardware errors' in error and 'motion errors' in error
    assert 'controller' in cleared and 'no error' in cleared


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


def test_a_nan_solution_that_the_controller_reports_with_no_error_is_not_picked(world, rctl, driven):
    rctl.ik_nan_solutions = 2
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.CartesianPosition(geom.Transform3D(translation=np.array([0.1, 0.2, 0.3]))))

    _pump_until(loop, clock, answer)

    assert answer.result() is None
    np.testing.assert_allclose(rctl.q[:3], [0.1, 0.2, 0.3], atol=0.01)


def test_a_move_to_a_pose_with_only_nan_solutions_fails_and_is_not_planned(world, rctl, driven):
    rctl.ik_nan_solutions, rctl.ik_only_nan = 8, True
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.CartesianPosition(geom.Transform3D(translation=np.array([0.1, 0.2, 0.3]))))

    _pump_until(loop, clock, answer)

    with pytest.raises(ValueError, match='out of reach'):
        answer.result()
    assert 'waypoint' not in rctl.calls


def test_a_move_to_a_pose_out_of_reach_fails(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.CartesianPosition(geom.Transform3D(translation=np.full(3, 5.0))))

    _pump_until(loop, clock, answer)

    with pytest.raises(ValueError, match='out of reach'):
        answer.result()


def test_a_streamed_cartesian_target_is_reached_after_one_command(rctl, driven):
    """The damped steps go on each tick toward the last Cartesian target, not once per command."""
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.6, -0.4, 0.5]))))

    _pump(loop, clock, 2000)

    np.testing.assert_allclose(rctl.q[:3], [0.6, -0.4, 0.5], atol=0.01)


def test_the_joints_of_the_wrist_turn_faster_than_the_joints_of_the_shoulder(rctl, driven):
    _, _, commands, clock, loop = driven(factor=0.5)
    commands.push(command.JointPosition(np.full(JOINTS, 1.0)))

    _pump(loop, clock, 400)

    fastest = np.max(np.abs(np.array(rctl.commanded)), axis=0)
    np.testing.assert_allclose(fastest[:3], 0.5 * cron._MAX_JOINT_SPEED, rtol=1e-6)
    np.testing.assert_allclose(fastest[3:], 0.5 * cron._MAX_JOINT_SPEED * 2.0, rtol=1e-6)


def test_a_streamed_cartesian_target_keeps_the_elbow_on_its_side(rctl, driven):
    """The fake IK puts the third coordinate on the elbow joint: through a straight elbow the arm turns over."""
    rctl.q = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.2, 0.0, -0.4]))))

    _pump(loop, clock, 3000)

    np.testing.assert_allclose(rctl.q[2], cron._MIN_ELBOW, atol=0.01)
    np.testing.assert_allclose(rctl.q[0], 0.2, atol=0.01)


@pytest.mark.parametrize(
    ('joint', 'start', 'goal', 'limit'),
    [(2, 2.6, 3.0, 167.0), (2, -2.6, -3.0, -167.0), (1, -3.2, -3.6, -190.8)],
    ids=['the elbow folded', 'the elbow folded the other way', 'the shoulder back'],
)
def test_a_streamed_cartesian_target_turns_a_joint_no_further_than_where_the_controller_stops_it(
    rctl, driven, joint, start, goal, limit
):
    """The fake IK puts the first three coordinates on the first three joints."""
    rctl.q = np.zeros(JOINTS)
    rctl.q[joint] = start
    _, _, commands, clock, loop = driven()
    target = np.zeros(3)
    target[joint] = goal
    commands.push(command.CartesianPosition(geom.Transform3D(translation=target)))

    _pump(loop, clock, 3000)

    np.testing.assert_allclose(rctl.q[joint], np.radians(limit) - np.sign(limit) * cron._LIMIT_MARGIN, atol=0.01)


@pytest.mark.parametrize(('joint', 'goal', 'limit'), [(2, 3.0, 167.0), (1, -3.6, -190.8)], ids=['elbow', 'shoulder'])
def test_a_streamed_joint_target_turns_a_joint_no_further_than_where_the_controller_stops_it(
    rctl, driven, joint, goal, limit
):
    _, _, commands, clock, loop = driven()
    target = np.zeros(JOINTS)
    target[joint] = goal
    commands.push(command.JointPosition(target))

    _pump(loop, clock, 3000)

    np.testing.assert_allclose(rctl.q[joint], np.radians(limit) - np.sign(limit) * cron._LIMIT_MARGIN, atol=0.01)


def test_a_streamed_cartesian_target_stops_short_of_the_arm_itself(rctl, driven):
    """The fake collision model puts a wall at 0.5 on the first joint."""
    rctl.wall = 0.5
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.8, 0.0, 0.0]))))

    _pump(loop, clock, 3000)

    assert 0.5 - cron._MIN_CLEARANCE - 0.06 <= rctl.q[0] <= 0.5 - cron._MIN_CLEARANCE + 1e-6


def test_a_streamed_joint_target_stops_short_of_the_arm_itself(rctl, driven):
    """The fake collision model puts a wall at 0.5 on the first joint."""
    rctl.wall = 0.5
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.8)))

    _pump(loop, clock, 3000)

    assert 0.5 - cron._MIN_CLEARANCE - 0.06 <= rctl.q[0] <= 0.5 - cron._MIN_CLEARANCE + 1e-6


def test_a_streamed_step_does_not_jump_over_a_narrow_collision(rctl, driven):
    """A late tick makes a long step: both of its ends can be clear of a thin part of the arm."""
    rctl.band = (0.1355, 0.1395)  # between the ends 0.125 and 0.15 of a step, more than the clearance from each
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.5)))

    for _ in range(300):
        clock.advance(cron._MAX_STEP_S)  # each tick late: a step of 0.025 rad
        next(loop)

    assert rctl.q[0] < 0.1355


def test_a_streamed_target_does_not_move_an_arm_that_is_in_a_collision(rctl, driven):
    """The model gives no depth in a collision: a step in one could go deeper."""
    rctl.wall, rctl.q = 0.5, _joints(0.6)
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.8)))

    _pump(loop, clock, 200)

    np.testing.assert_allclose(rctl.q[0], 0.6, atol=1e-6)


def test_a_planned_move_that_takes_the_arm_into_itself_is_refused(world, rctl, driven):
    """The controller plans the move without the tool spheres of the local model."""
    rctl.wall = 0.5
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.8)))

    _pump_until(loop, clock, answer)

    with pytest.raises(ValueError, match='from itself'):
        answer.result()
    assert 'waypoint' not in rctl.calls


def test_a_planned_move_out_of_a_collision_may_not_go_into_another(world, rctl, driven):
    rctl.wall, rctl.floor, rctl.q = 0.5, -0.2, _joints(0.6)
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(-0.5)))

    _pump_until(loop, clock, answer)

    with pytest.raises(ValueError, match='from itself'):
        answer.result()


def test_a_planned_move_past_a_joint_limit_is_refused(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(np.array([0.0, 0.0, np.radians(169.0), 0.0, 0.0, 0.0])))

    _pump_until(loop, clock, answer)

    with pytest.raises(ValueError, match='joint limits'):
        answer.result()
    assert 'waypoint' not in rctl.calls


def test_a_planned_move_is_checked_again_from_where_the_arm_is_when_it_starts(world, rctl, driven):
    """The arm can move between the call and the start: a brake, a recovery."""
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.8)))
    _pump(loop, clock, 10)

    rctl.wall, rctl.safety_latched = 0.5, False
    _pump_until(loop, clock, answer, ticks=5000)

    with pytest.raises(ValueError, match='from itself'):
        answer.result()
    assert 'waypoint' not in rctl.calls


def test_a_planned_move_out_of_a_collision_is_planned(world, rctl, driven):
    """The stick takes an arm out of a contact."""
    rctl.wall, rctl.q = 0.5, _joints(0.6)
    driver, _, _, clock, loop = driven()
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))

    _pump_until(loop, clock, answer)

    assert answer.result() is None


def test_the_spheres_of_the_tool_go_to_the_collision_model_as_capsules_of_no_length(rctl, driven):
    _, _, _, clock, loop = driven(tool_spheres=[([0.0, 0.01, 0.03], 0.05)])

    _pump(loop, clock, 1)

    assert rctl.tool_capsules == [{'length': 0.0, 'radius': 0.05, 'xyz': [0.0, 0.01, 0.03], 'rpy': [0.0, 0.0, 0.0]}]


def test_a_streamed_cartesian_target_at_a_pose_where_a_joint_does_nothing_keeps_the_joints_finite(rctl, driven):
    """With a column of the jacobian at zero the plain least squares step has no solution; the damping gives one."""
    rctl.dead_joint = 1
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.3, 0.2, 0.1]))))

    _pump(loop, clock, 2000)

    assert all(np.all(np.isfinite(velocity)) for velocity in rctl.commanded)
    np.testing.assert_allclose(rctl.q[[0, 2]], [0.3, 0.1], atol=0.01)
    np.testing.assert_allclose(rctl.q[1], 0.0, atol=1e-6)


def test_a_joint_target_that_is_not_finite_is_not_a_setpoint(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(np.array([0.1, np.nan, 0.0, 0.0, 0.0, 0.0])))

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


def test_leaving_the_run_during_a_planned_move_stops_the_move_and_sends_no_velocity(world, rctl, driven):
    """A velocity that reaches the controller while a planned move runs stops the arm with `JOINT_VEL: TIMEOUT`."""
    stop = StopFlag()
    driver, _, _, clock, loop = driven(stop=stop, factor=0.2)
    _mover(world, driver)(command.JointPosition(_joints(1.0)))
    _pump(loop, clock, 20)
    assert rctl.moving

    stop.stopped = True
    with pytest.raises(StopIteration):
        _pump(loop, clock, 1)

    assert not rctl.error
    assert rctl.commanded == []
    assert _called_in_order(rctl.calls, 'move_pause', 'drop_waypoints', 'hold', 'off')


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


def test_an_arm_left_powered_gives_control_back_only_once_the_controller_holds(rctl, driven):
    """A release that reaches the controller in a stream can end it in a TIMEOUT error."""
    rctl.hold_lag = 2
    stop = StopFlag()
    _, _, commands, clock, loop = driven(stop=stop, power_off_on_exit=False)
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 20)

    stop.stopped = True
    with pytest.raises(StopIteration):
        _pump(loop, clock, 1)

    assert rctl.mode_at_release is MotionMode.HOLD
    assert 'off' not in rctl.calls


def test_an_arm_that_fails_to_power_up_is_left_off_and_the_run_fails(rctl, driven):
    rctl.fail_power_run = True
    _, _, _, clock, loop = driven()

    with pytest.raises(RuntimeError, match=r'the controller \S+ failed'):
        _pump(loop, clock, 1)

    assert rctl.power is PowerState.OFF
    assert rctl.calls[-2:] == ['release_control', 'disconnect']


@pytest.mark.parametrize('positions', [np.array(0.3), np.array([0.3])], ids=['a scalar', 'one joint'])
def test_a_joint_target_of_the_wrong_shape_is_not_a_setpoint(rctl, driven, positions):
    """A target of one value would broadcast to every joint."""
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(positions))

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_a_cartesian_target_that_is_not_finite_is_not_followed(rctl, driven):
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.1, np.nan, 0.3]))))

    _pump(loop, clock, 20)

    assert rctl.commanded == []


def test_a_cartesian_delta_applies_to_the_pose_that_the_arm_reads_and_not_to_the_target(rctl, driven):
    """The arm lags behind a far streamed target when the delta comes."""
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.6, 0.0, 0.0]))))
    _pump(loop, clock, 20)
    lagging = rctl.q.copy()
    assert 0.0 < lagging[0] < 0.3

    commands.push(command.CartesianDelta(geom.Transform3D(translation=np.array([0.0, 0.05, 0.0]))))
    _pump(loop, clock, 2000)

    np.testing.assert_allclose(rctl.q[:3], lagging[:3] + [0.0, 0.05, 0.0], atol=0.01)


def test_a_move_clears_an_error_that_comes_after_it_was_asked_for(world, rctl, driven):
    """The operator can press the stick just as the arm stops on an error."""
    driver, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(2.0)))
    _pump(loop, clock, 100)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))
    _pump(loop, clock, 2)

    rctl.error = True
    _pump_until(loop, clock, answer, ticks=5000)

    assert answer.result() is None
    assert _called_in_order(rctl.calls, 'resume', 'waypoint', 'move_run')


def test_a_move_asks_again_to_clear_an_error_that_stays_through_the_first_request(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, int(2.5 / TICK))
    assert rctl.calls.count('resume') >= 2

    rctl.safety_latched = False
    _pump_until(loop, clock, answer, ticks=5000)

    assert answer.result() is None


def test_a_planned_move_that_stops_on_an_error_goes_on_once_the_error_clears(world, rctl, driven):
    driver, _, _, clock, loop = driven(factor=0.2)
    answer = _mover(world, driver)(command.JointPosition(_joints(1.0)))
    _pump(loop, clock, 200)
    assert 0.0 < rctl.q[0] < 1.0
    del rctl.calls[:]

    rctl.error, rctl.moving = True, False  # the controller stops the planner
    _pump_until(loop, clock, answer, ticks=int(20.0 / TICK))

    assert answer.result() is None
    assert _called_in_order(rctl.calls, 'move_pause', 'resume', 'waypoint', 'move_run')
    np.testing.assert_allclose(rctl.q, _joints(1.0), atol=cron._ARRIVED_TOL)


def test_a_move_whose_hold_does_not_come_waits_and_the_driver_goes_on(world, rctl, driven, monkeypatch):
    """An error on the tick of the start of a move keeps the controller out of HOLD."""
    monkeypatch.setattr(cron_controller, '_HOLD_TIMEOUT_S', 0.01)
    driver, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, 20)
    rctl.hold_lag = 10**6
    answer = _mover(world, driver)(command.JointPosition(_joints(0.0)))
    _pump(loop, clock, 200)
    assert not answer.done()

    rctl.hold_lag = 0
    _pump_until(loop, clock, answer)

    assert answer.result() is None


def test_a_streamed_target_is_not_taken_while_the_arm_holds_an_error(rctl, driven):
    """A HOLD request out of RUN can wait for its timeout on each target."""
    _, _, commands, clock, loop = driven()
    commands.push(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, 20)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    del rctl.calls[:]

    for _ in range(5):
        commands.push(command.JointPosition(_joints(0.3)))
        _pump(loop, clock, 1)

    assert 'hold' not in rctl.calls


def test_a_tick_where_the_model_refuses_a_pose_drops_the_stream_and_the_driver_goes_on(rctl, driven, caplog):
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.3, 0.0, 0.0]))))
    _pump(loop, clock, 20)

    rctl.fk_fails = True
    with caplog.at_level(logging.ERROR, logger=cron.__name__):
        _pump(loop, clock, 100)
    rctl.fk_fails = False
    commands.push(command.JointPosition(_joints(0.5)))
    _pump(loop, clock, 600)

    assert any('dropped' in record.getMessage() for record in caplog.records)
    np.testing.assert_allclose(rctl.q[0], 0.5, atol=0.01)


def test_a_refused_move_reports_how_far_the_arm_moved_while_the_driver_cleared_the_error(world, rctl, driven):
    """The controller moves the joints once just after RUN."""
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, 10)

    rctl.q = rctl.q + _joints(0.02)
    _pump_until(loop, clock, answer, ticks=int(2 * cron._RECOVER_S / TICK))

    with pytest.raises(MoveRefused) as refused:
        answer.result()
    assert refused.value.moved_rad == pytest.approx(0.02)


def test_a_move_to_the_joints_of_the_arm_is_not_answered_while_the_controller_holds_an_error(world, rctl, driven):
    """The move did not start: it cannot arrive."""
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched = True, True
    _pump(loop, clock, 2)
    answer = _mover(world, driver)(command.JointPosition(rctl.q.copy()))

    _pump(loop, clock, 100)

    assert not answer.done()


@pytest.mark.parametrize('power', [PowerState.OFF, PowerState.STANDBY])
def test_an_arm_out_of_run_with_no_error_reads_as_error_on_the_wire(rctl, driven, power):
    """A command sent now would not reach the arm."""
    _, states, _, clock, loop = driven()
    _pump(loop, clock, 2)

    rctl.power = power
    _pump(loop, clock, 2)

    assert states.emitted[-1][1].status is RobotStatus.ERROR


def test_a_move_after_a_move_that_was_given_up_reads_busy(world, rctl, driven):
    """The driver owns the arm while the move travels, and reads no command."""
    rctl.blocked = True
    driver, states, _, clock, loop = driven(factor=0.2)
    move = _mover(world, driver)
    given_up = move(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, given_up, ticks=10000)
    with pytest.raises(TimeoutError):
        given_up.result()
    rctl.blocked = False

    answer = move(command.JointPosition(_joints(1.0)))
    _pump(loop, clock, 20)

    assert states.emitted[-1][1].status is RobotStatus.BUSY
    assert not answer.done()


def test_a_run_that_comes_after_its_move_was_refused_still_holds_commands_for_the_joint_move(world, rctl, driven):
    """The arm moves its joints once just after RUN, and a command in that second stops it on a following error."""
    driver, _, commands, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _pump(loop, clock, 2)
    rctl.safety_latched, rctl.run_lag = False, 10**9
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, answer, ticks=int(2 * cron._RECOVER_S / TICK))
    with pytest.raises(MoveRefused):
        answer.result()
    assert 'run' in rctl.calls

    rctl._run_in = 0  # RUN comes now, and the stream goes on at each tick
    for _ in range(int(0.8 * cron.STARTUP_S / TICK)):
        commands.push(command.JointPosition(_joints(0.5)))
        _pump(loop, clock, 1)
    assert rctl.commanded == [], 'the arm got a command during the joint move after RUN'

    for _ in range(int(0.5 * cron.STARTUP_S / TICK)):
        commands.push(command.JointPosition(_joints(0.5)))
        _pump(loop, clock, 1)
    assert rctl.commanded


def test_a_power_up_that_never_reaches_run_is_asked_for_again(world, rctl, driven):
    """A request for RUN can be lost: the arm stays off, and each later move would be refused."""
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _pump(loop, clock, 2)
    rctl.safety_latched, rctl.run_lag = False, 10**9
    move = _mover(world, driver)
    refused = move(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, refused, ticks=int(2 * cron._RECOVER_S / TICK))
    with pytest.raises(MoveRefused):
        refused.result()

    rctl._run_in, rctl.run_lag = None, 0  # the request was lost
    answer = move(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, answer, ticks=int(2 * cron._RECOVER_S / TICK))

    assert answer.result() is None
    np.testing.assert_allclose(rctl.q, _joints(0.3), atol=cron._ARRIVED_TOL)


def test_a_move_powers_an_arm_in_standby_to_run_through_off(world, rctl, driven):
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.power = PowerState.STANDBY
    _pump(loop, clock, 2)
    del rctl.calls[:]

    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump_until(loop, clock, answer, ticks=5000)

    assert answer.result() is None
    assert _called_in_order(rctl.calls, 'off', 'run', 'waypoint')


def test_a_power_up_that_stops_in_standby_goes_back_to_off_before_it_asks_for_run_again(
    world, rctl, driven, monkeypatch
):
    """RUN from STANDBY failed on an arm whose multi-turn encoder asks for a preset."""
    monkeypatch.setattr(cron, 'POWER_TIMEOUT_S', 2.0)
    driver, _, _, clock, loop = driven()
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _pump(loop, clock, 2)
    rctl.safety_latched, rctl.run_stalls = False, True
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    _pump(loop, clock, int(1.0 / TICK))
    assert rctl.power is PowerState.STANDBY
    del rctl.calls[:]

    rctl.run_stalls = False
    _pump_until(loop, clock, answer, ticks=int(cron._RECOVER_S / TICK))

    assert answer.result() is None
    assert _called_in_order(rctl.calls, 'off', 'run')


def test_a_cartesian_delta_does_not_drive_on_to_an_earlier_target_that_the_arm_has_not_reached(rctl, driven):
    """The seed of the steps is the arm, not the earlier target. The arm still brakes from its speed: 0.125 rad at
    1 rad/s and 4 rad/s^2."""
    rctl.stuck_joint = 0
    _, _, commands, clock, loop = driven()
    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.array([0.9, 0.0, 0.0]))))
    _pump(loop, clock, 200)
    rctl.stuck_joint = None

    commands.push(command.CartesianDelta(geom.Transform3D(translation=np.array([0.0, 0.05, 0.0]))))
    furthest = 0.0
    for _ in range(600):
        _pump(loop, clock, 1)
        furthest = max(furthest, rctl.q[0])

    assert furthest < 0.2, f'the arm drove {furthest:.3f} rad on toward the earlier target'
    np.testing.assert_allclose(rctl.q[1], 0.05, atol=0.01)


def test_a_move_clears_a_failure_by_a_power_off_and_starts_once_the_arm_is_up_again(world, rctl, driven):
    """A FAILURE stays through a resume. The arm moves its joints once just after RUN."""
    driver, _, _, clock, loop = driven(projects=['gripper_bridge'])
    _pump(loop, clock, 2)
    rctl.error, rctl.safety_latched, rctl.power = True, True, PowerState.FAILURE
    _pump(loop, clock, 2)
    del rctl.calls[:]

    rctl.run_lag = 600
    answer = _mover(world, driver)(command.JointPosition(_joints(0.3)))
    for _ in range(2000):
        _pump(loop, clock, 1)
        if rctl.power is PowerState.RUN:
            break
    _pump(loop, clock, int(0.8 * cron.STARTUP_S / TICK))
    assert 'waypoint' not in rctl.calls, 'the move started during the joint move after RUN'
    _pump_until(loop, clock, answer, ticks=5000)

    assert answer.result() is None
    assert _called_in_order(rctl.calls, 'resume', 'off', 'exr_run gripper_bridge', 'run', 'waypoint', 'move_run')
    np.testing.assert_allclose(rctl.q, _joints(0.3), atol=cron._ARRIVED_TOL)


def test_leaving_the_run_brakes_the_stream_to_a_stop(rctl, driven):
    stop = StopFlag()
    _, _, commands, clock, loop = driven(stop=stop)
    commands.push(command.JointPosition(_joints(2.0)))
    _pump(loop, clock, 100)
    streamed = len(rctl.commanded)

    stop.stopped = True
    with pytest.raises(StopIteration):
        _pump(loop, clock, 1)

    braking = np.array(rctl.commanded[streamed - 1 : -cron_controller._STOP_REPEATS])
    assert np.max(np.abs(np.diff(braking, axis=0))) <= cron._MAX_JOINT_ACCEL * _WRIST_MAX * TICK + 1e-9
    np.testing.assert_array_equal(braking[-1], np.zeros(JOINTS))
