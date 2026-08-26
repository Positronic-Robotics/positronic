"""Driver for the Trossen WidowX AI arm — one Ethernet link carrying six joints plus the gripper.

The controller firmware runs the servo loop and solves its own kinematics, so this driver streams joint
setpoints and reads back the end-effector pose the firmware reports, rather than solving FK/IK itself.
A streamed setpoint is applied without interpolation; a synchronous move hands the firmware a goal time
and lets it plan the trajectory.

The gripper is the 7th joint, a prismatic finger drive the controller reports in meters. positronic
speaks a normalized grip where 1 is closed, so grip converts against the travel the arm reports for that
joint instead of a constant.

A link that drops takes the controller's TCP session with it, and the vendor driver does not open a new
one. Telemetry and commands travel separately — the controller streams the first over UDP and takes the
second over TCP — so the arm can be heard from while nothing reaches it. The driver opens a new session
itself once either half stops working, and resumes from wherever it finds the arm.
"""

import contextlib
import logging
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers import vendor_import
from positronic.drivers.utils import DriverRun, MoveStatus, log_failure

from . import RobotStatus, State, command

# trossen_arm lives in the `trossen` extra, which the type-check environment does not install.
with vendor_import(
    'trossen_arm', 'Trossen arm support', hint='Re-run with the trossen extra:\n  uv run --locked --extra trossen ...\n'
):
    import trossen_arm  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_ARM_JOINTS = 6
_GRIPPER_JOINT = 6
_HZ = 100
# Goal time for a streamed setpoint. At or below 0.001 s the firmware applies the goal without interpolation,
# which is what a setpoint that is already one tick away needs.
_STREAM_GOAL_TIME_S = 0.0
# Goal time for a synchronous move, which the firmware plans as a quintic trajectory above 0.2 s.
_MOVE_GOAL_TIME_S = 2.0
_MOVE_TIMEOUT_S = _MOVE_GOAL_TIME_S + 3.0
_ARRIVED_TOL = 0.02  # radians; the firmware reports position, so arrival is judged from the reading
# How long the controller's clock may stand still before the link counts as down. Telemetry arrives at
# over 200 Hz, so this is many missed frames, not a scheduling hiccup.
_STALE_AFTER_S = 0.25
_CONNECT_TIMEOUT_S = 20.0
# A reconnect runs on the control loop, so its timeout is what the loop stands still for on a failed attempt.
_RECONNECT_TIMEOUT_S = 1.0
_RECONNECT_AFTER_S = 0.5  # how long the link stays down before a new session is worth opening
_RECONNECT_EVERY_S = 2.0  # and how often another is tried while it stays down
# How many points along a planned Cartesian trajectory the firmware checks for a solution before it starts.
_TRAJECTORY_CHECK_SAMPLES = 10
# How far ahead of where the arm reads a streamed Cartesian target may sit. At the tick rate this allows
# 1.5 m/s, which is faster than a hand moves, so it does not shape teleoperation.
_MAX_STEP_M = 0.015
_MAX_STEP_RAD = 0.08


class TrossenState(State, pimm.shared_memory.NumpySMAdapter):
    Q_OFFSET = 0
    DQ_OFFSET = Q_OFFSET + _ARM_JOINTS
    EE_POSE_OFFSET = DQ_OFFSET + _ARM_JOINTS
    STATUS_OFFSET = EE_POSE_OFFSET + 7
    TOTAL = STATUS_OFFSET + 1

    def __init__(self):
        super().__init__(shape=(TrossenState.TOTAL,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[TrossenState.Q_OFFSET : TrossenState.Q_OFFSET + _ARM_JOINTS].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[TrossenState.DQ_OFFSET : TrossenState.DQ_OFFSET + _ARM_JOINTS].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[TrossenState.EE_POSE_OFFSET : TrossenState.EE_POSE_OFFSET + 7].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[TrossenState.STATUS_OFFSET]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus) -> None:
        self.array[TrossenState.Q_OFFSET : TrossenState.Q_OFFSET + _ARM_JOINTS] = q
        self.array[TrossenState.DQ_OFFSET : TrossenState.DQ_OFFSET + _ARM_JOINTS] = dq
        self.array[TrossenState.EE_POSE_OFFSET : TrossenState.EE_POSE_OFFSET + 3] = ee_pose.translation
        self.array[TrossenState.EE_POSE_OFFSET + 3 : TrossenState.EE_POSE_OFFSET + 7] = ee_pose.rotation.as_quat
        self.array[TrossenState.STATUS_OFFSET] = status.value


def _configure(driver: Any, ip: str, timeout_s: float) -> None:
    """Open a session with the controller, clearing an error a previous one left behind."""
    end_effector = trossen_arm.StandardEndEffector.wxai_v0_follower
    driver.configure(trossen_arm.Model.wxai_v0, end_effector, ip, True, timeout_s)


def _connect(ip: str) -> Any:
    """Open the arm controller and take ownership of it."""
    driver = trossen_arm.TrossenArmDriver()
    _configure(driver, ip, _CONNECT_TIMEOUT_S)
    return driver


@contextlib.contextmanager
def _opened(connect: Callable[[str], Any], ip: str) -> Iterator[Any]:
    """The arm, left idle and its handle given back however the run ends — including one that never starts."""
    driver = connect(ip)
    try:
        yield driver
    finally:
        try:
            driver.set_all_modes(trossen_arm.Mode.idle)
        # rules-allow: swallowed-error — an arm that cannot be reached cannot be set idle either, and the
        # handle still has to go back
        except trossen_arm.RuntimeError as exc:
            logger.error(f'The arm at {ip} was not set idle: {exc}')
        finally:  # an arm that will not go idle still has a handle to give back
            driver.cleanup()


class _Arm(DriverRun[command.CommandType]):
    """The arm the driver drives: the controller handle, the reading it takes each tick, and the setpoint
    it holds the arm at.

    The controller reports position but not whether it is tracking a goal, so arrival is judged from the
    reading. A setpoint is written only when something has asked for a new one: a goal time re-sent every
    tick restarts the trajectory it plans, and the arm would never arrive.
    """

    def __init__(
        self,
        driver: Any,
        ip: str,
        sync_move: pimm.calls.ControlSystemHandler[command.CommandType, None],
        async_move: pimm.SignalReceiver[command.CommandType],
        out: pimm.SignalEmitter[TrossenState],
        grip_out: pimm.SignalEmitter[float],
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=_HZ)
        self.driver = driver
        self.ip = ip
        self.out = out
        self.grip_out = grip_out
        self.state = TrossenState()
        limits = driver.get_joint_limits()
        self._q_lower = np.array([limits[i].position_min for i in range(_ARM_JOINTS)])
        self._q_upper = np.array([limits[i].position_max for i in range(_ARM_JOINTS)])
        # The gripper joint's own travel, which grip is normalized against
        self._grip_travel = float(limits[_GRIPPER_JOINT].position_max - limits[_GRIPPER_JOINT].position_min)
        self._grip_closed = float(limits[_GRIPPER_JOINT].position_min)
        self._output = driver.get_robot_output()
        self._target: np.ndarray | geom.Transform3D = np.asarray(self._output.joint.arm.positions, dtype=np.float64)
        self._grip_target = self._grip_of(self._output)
        self._goal_time = _STREAM_GOAL_TIME_S
        self._arm_unsent, self._grip_unsent = False, False
        # The two halves of the link, which fail apart. Neither is `Moves.errored`, which says the arm is
        # not where the driver put it: a link that drops says nothing about the move.
        self._stream_stale = False  # the controller's clock stands still, so its telemetry stopped arriving
        self._command_dead = False  # a write was refused, and only a new session takes another
        self._down_since: float | None = None
        self._reconnect_at = -_RECONNECT_EVERY_S
        self._stamp = int(self._output.header.timestamp)
        self._stamp_at = clock.now()

    @property
    def link_down(self) -> bool:
        """Whether the arm is out of reach, either way round."""
        return self._stream_stale or self._command_dead

    def _note_link(self, now: float) -> None:
        """Keep when the link went down, which is what a reconnect waits on."""
        if not self.link_down:
            self._down_since = None
        elif self._down_since is None:
            self._down_since = now

    def _grip_of(self, output: Any) -> float:
        """How closed the fingers are, from the joint position the controller reports.

        The reading sits a little outside the joint range at either end, so it saturates to the 0..1 the
        ``grip`` port carries.
        """
        travelled = (float(output.joint.gripper.position) - self._grip_closed) / self._grip_travel
        return float(np.clip(1.0 - travelled, 0.0, 1.0))

    def _grip_metres(self, grip: float) -> float:
        """The gripper joint position that holds the fingers at ``grip``."""
        return self._grip_closed + (1.0 - grip) * self._grip_travel

    def _take_control(self) -> None:
        """Put the arm in position mode holding where it reads.

        The mode change comes first and the setpoint immediately after, so the servo has a goal from the
        tick it starts servoing. Reading first is what makes a session opened mid-run resume without a jump:
        the arm is wherever it ended up, not where the last session was driving it.
        """
        self._output = self.driver.get_robot_output()
        self._target = self.q
        self._grip_target = self._grip_of(self._output)
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self._arm_unsent, self._grip_unsent = True, True
        self.write()

    def __enter__(self) -> '_Arm':
        self._take_control()
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb) -> None:
        """Answer whatever was waiting on a move; ``_opened`` takes the arm back to idle."""
        self.moves.abandon(exc)

    def read(self) -> None:
        """Take the whole arm off the link, once a tick.

        The controller streams its telemetry and the read hands back the last of it, so a link that drops
        does not raise — it repeats itself. The controller's own clock is what says the stream stopped.
        """
        now = self.clock.now()
        try:
            self._output = self.driver.get_robot_output()
        # rules-allow: swallowed-error — a link that drops reads ERROR; the run outlives it, and a new
        # session clears it
        except trossen_arm.RuntimeError as exc:
            logger.error(f'The arm at {self.ip} did not answer: {exc}')
            self._stream_stale = True
            self._note_link(now)
            return
        stamp = int(self._output.header.timestamp)
        if stamp != self._stamp:
            self._stamp, self._stamp_at = stamp, now
        self._stream_stale = now - self._stamp_at > _STALE_AFTER_S
        self._note_link(now)

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self._output.joint.arm.positions, dtype=np.float64)

    @property
    def ee_pose(self) -> geom.Transform3D:
        """The end-effector pose the firmware reports, in the arm base frame.

        The controller speaks angle-axis where positronic speaks a rotation.
        """
        cartesian = np.asarray(self._output.cartesian.positions, dtype=np.float64)
        return geom.Transform3D(cartesian[:3], geom.Rotation.from_rotvec(cartesian[3:6]))

    def settle(self) -> None:
        """Judge a move in flight against what the controller reports."""
        if not self.moves.active:
            return
        if self.moves.settle(self.q, self.clock.now()) is MoveStatus.GAVE_UP:
            # Holding the target the arm stopped short of would resume the move once whatever blocked it
            # goes away, long after its asker was told it failed.
            self._target, self._goal_time, self._arm_unsent = self.q, _STREAM_GOAL_TIME_S, True

    def hold_grip(self, grip: float) -> None:
        """Hold the fingers at ``grip``."""
        self._grip_target = float(np.clip(grip, 0.0, 1.0))
        self._grip_unsent = True

    def _stepped(self, target: geom.Transform3D) -> geom.Transform3D:
        """``target`` brought within one tick's travel of where the arm reads.

        A teleoperator reaching past what the arm can do produces targets that run away from it. Measured
        against the arm rather than against the last target, the goal cannot outrun an arm that is held up,
        so nothing is stored up for it to lunge through once whatever held it goes away.
        """
        held = self.ee_pose
        step = target.translation - held.translation
        distance = float(np.linalg.norm(step))
        if distance > _MAX_STEP_M:
            step = step * (_MAX_STEP_M / distance)
        turn = (held.rotation.inv * target.rotation).as_rotvec
        angle = float(np.linalg.norm(turn))
        if angle > np.pi:  # `as_rotvec` keeps the way round it was given; the other one is the short way
            turn, angle = turn * (1.0 - 2.0 * np.pi / angle), 2.0 * np.pi - angle
        if angle > _MAX_STEP_RAD:
            turn = turn * (_MAX_STEP_RAD / angle)
        return geom.Transform3D(held.translation + step, held.rotation * geom.Rotation.from_rotvec(turn))

    def _target_of(self, cmd: command.CommandType) -> np.ndarray | geom.Transform3D:
        """What ``cmd`` asks the arm to hold: joints clipped to their range, or a pose one step away."""
        # TODO: accept the modes the arm can run instead of leaving them to what a command omits. Its joints
        # are position-servoed, so `PositionControl` names the law already running.
        command.require_native_mode(cmd, 'Trossen')
        match cmd:
            case command.JointPosition(positions):
                return np.clip(np.asarray(positions, dtype=np.float64), self._q_lower, self._q_upper)
            case command.JointDelta(velocities=delta):
                target = self.q + np.asarray(delta, dtype=np.float64)
                return np.clip(target, self._q_lower, self._q_upper)
            case command.CartesianPosition(pose):
                return self._stepped(pose)
            case command.CartesianDelta() as delta_cmd:
                return self._stepped(delta_cmd.apply(self.ee_pose))
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def track(self, cmd: command.CommandType) -> None:
        """Hold the arm at the setpoint ``cmd`` asks for, with nobody waiting on the arrival."""
        self._target, self._goal_time, self._arm_unsent = self._target_of(cmd), _STREAM_GOAL_TIME_S, True

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Hand the firmware the trajectory ``call`` asks for; ``settle`` answers it once the arm reads
        back there."""
        with pimm.calls.raise_to(call):
            target = self._target_of(call.request)
            if not isinstance(target, np.ndarray):
                # Arrival is judged from the joints the controller reports, and a pose does not say which
                # joints reach it. Solving that here is what host-side kinematics is for.
                raise NotImplementedError('Trossen cannot answer a Cartesian move; ask for one in joint space')
            self._target, self._goal_time, self._arm_unsent = target, _MOVE_GOAL_TIME_S, True
            self.moves.accept(call, target, _ARRIVED_TOL, self.clock.now(), _MOVE_TIMEOUT_S)

    def _put_goal(self) -> None:
        """Hand the controller the goal it is being held at, fingers included where one call carries both.

        All seven joints are in position mode, which ``set_all_positions`` requires. A Cartesian goal names
        the arm alone, so the fingers take their own call. ``num_trajectory_check_samples`` makes the
        firmware refuse a path it cannot solve rather than start one and fail part-way.
        """
        grip_m = self._grip_metres(self._grip_target)
        if isinstance(self._target, np.ndarray):
            self.driver.set_all_positions([*self._target, grip_m], self._goal_time, False)
            self._arm_unsent, self._grip_unsent = False, False
            return
        pose = [*self._target.translation, *self._target.rotation.as_rotvec]
        if self._arm_unsent:
            self.driver.set_cartesian_positions(
                pose,
                trossen_arm.InterpolationSpace.joint,
                self._goal_time,
                False,
                num_trajectory_check_samples=_TRAJECTORY_CHECK_SAMPLES,
            )
            self._arm_unsent = False
        if self._grip_unsent:
            self.driver.set_gripper_position(grip_m, self._goal_time, False)
            self._grip_unsent = False

    def write(self) -> None:
        """Put the setpoint on the link, if anything has asked for one since it was last written."""
        if not (self._arm_unsent or self._grip_unsent):
            return
        try:
            self._put_goal()
        # rules-allow: swallowed-error — a link that refuses a write reads ERROR; the setpoint stays unsent
        # and goes out again on the next session
        except trossen_arm.RuntimeError as exc:
            logger.error(f'The arm at {self.ip} did not take the setpoint: {exc}')
            self._command_dead = True
            self._note_link(self.clock.now())
            return
        self._command_dead = False
        self._note_link(self.clock.now())

    def recover(self) -> None:
        """Open a new session once the link has been down long enough, and no more often than that again.

        A session does not survive the link dropping and the vendor driver does not open another, so the
        arm stays out of reach until this does it. The attempt runs on the control loop, which stands still
        for as long as the connection takes to fail.
        """
        now = self.clock.now()
        if self._down_since is None or now - self._down_since < _RECONNECT_AFTER_S:
            return
        if now - self._reconnect_at < _RECONNECT_EVERY_S:
            return
        self._reconnect_at = now
        logger.info(f'Opening a new session with the arm at {self.ip}')
        try:
            with contextlib.suppress(trossen_arm.RuntimeError):  # the old session is what failed
                self.driver.cleanup()
            _configure(self.driver, self.ip, _RECONNECT_TIMEOUT_S)
            self._take_control()
        # rules-allow: swallowed-error — an arm still out of reach reads ERROR; the next attempt tries again
        except trossen_arm.RuntimeError as exc:
            logger.error(f'The arm at {self.ip} did not take a new session: {exc}')
            return
        self._stamp, self._stamp_at = int(self._output.header.timestamp), now
        self._stream_stale = False
        self._note_link(now)
        logger.info(f'The arm at {self.ip} answers again')

    def publish(self) -> None:
        """Ship the arm as the controller last reported it, arm and fingers."""
        if self.link_down or self.moves.errored:  # not where the driver put it, or out of reach entirely
            status = RobotStatus.ERROR
        elif self.moves.active:  # the driver owns the arm until the move settles
            status = RobotStatus.BUSY
        else:
            status = RobotStatus.AVAILABLE
        velocities = np.asarray(self._output.joint.arm.velocities, dtype=np.float64)
        self.state.encode(self.q, velocities, self.ee_pose, status)
        self.out.emit(self.state)
        self.grip_out.emit(self._grip_of(self._output))


class Robot(pimm.ControlSystem):
    """Drives one Trossen WidowX AI arm over Ethernet, in the arm base frame.

    The gripper shares the arm's controller, so this driver carries the ``grip``/``target_grip`` ports
    (SO-101 precedent).
    """

    def __init__(self, ip: str = '192.168.1.4', *, connect: Callable[[str], Any] = _connect) -> None:
        """
        :param ip: Address of the arm controller.
        :param connect: ``ip -> TrossenArmDriver`` factory; the fake-mode smoke injects ``_FakeTrossen``.
        """
        self._ip = ip
        self._connect = connect

        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.state = pimm.ControlSystemEmitter[TrossenState](self)
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.robot_meta = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        with _opened(self._connect, self._ip) as driver:
            arm = _Arm(driver, self._ip, self.sync_move, self.commands, self.state, self.grip, should_stop, clock)
            with arm:
                # TODO: carry the URDF and the joint names, which live in `trossen_arm_description`.
                self.robot_meta.emit({'robot': 'trossen_wxai'})

                while not should_stop.value:
                    arm.read()
                    if arm.link_down:
                        arm.recover()  # get the arm back first, so the state that goes out says where it is
                        arm.settle()  # a move runs out its deadline on the last reading; nobody waits forever
                        arm.publish()
                        arm.moves.answer()
                        yield arm.limiter.wait()
                        continue

                    if (grip := pimm.value_updated(self.target_grip)) is not None:
                        arm.hold_grip(grip)
                    arm.settle()
                    asked = arm.moves.next_request()
                    if isinstance(asked, pimm.calls.Call):
                        arm.sync_move(asked)
                    elif asked is not None:
                        with log_failure(asked):
                            arm.track(asked)

                    arm.write()
                    arm.publish()
                    arm.moves.answer()  # the state a settled move is answered with is out

                    yield arm.limiter.wait()


class _FakeTrossen:
    """First-order-lag echo of the 7-joint arm, so the ``--fake`` smoke runs without hardware.

    Duck-types the slice of ``TrossenArmDriver`` the driver uses. It models the link and the servo, not
    the kinematics: the Cartesian reading it reports is a constant.
    """

    # What the arm reports for itself, read off a wxai_v0 controller on firmware 1.11.1
    _LIMITS = [
        (-3.141593, 3.141593),
        (0.0, 3.141593),
        (0.0, 2.356194),
        (-1.570796, 1.570796),
        (-1.570796, 1.570796),
        (-3.141593, 3.141593),
        (0.0, 0.04),
    ]
    _CARTESIAN = [0.254, -0.0039, 0.1618, -0.0086, 0.009, -0.0179]

    class _Limit:
        def __init__(self, lower: float, upper: float):
            self.position_min = lower
            self.position_max = upper

    _TICK_US = 5000  # the controller streams faster than the driver reads, so its clock moves every read

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._position = np.zeros(7)  # the arm boots with the fingers closed
        self._velocity = np.zeros(7)
        self._stamp = 0
        self.frozen = False  # the controller stops being heard from, as a dropped link leaves it
        self.sessions = 1
        self.mode: Any = None
        self.goals: list[list[float]] = []
        self.poses: list[list[float]] = []
        self.goal_times: list[float] = []
        self.checked_samples: list[int] = []
        self.gripper_goals: list[float] = []
        self._gripper_goal: float | None = None
        self.cleaned_up = False

    def configure(self, model: Any, end_effector: Any, serv_ip: str, clear_error: bool, timeout: float = 20.0):
        self.mode = None
        self.cleaned_up = False
        self.sessions += 1

    def get_error_information(self) -> str:
        return 'No error'

    def get_joint_limits(self) -> list['_FakeTrossen._Limit']:
        return [_FakeTrossen._Limit(lower, upper) for lower, upper in _FakeTrossen._LIMITS]

    def get_robot_output(self) -> Any:
        if not self.frozen:
            self._servo()
            self._stamp += _FakeTrossen._TICK_US
        arm = SimpleNamespace(positions=self._position[:_ARM_JOINTS].copy(), velocities=self._velocity[:6].copy())
        gripper = SimpleNamespace(position=float(self._position[_GRIPPER_JOINT]))
        return SimpleNamespace(
            joint=SimpleNamespace(arm=arm, gripper=gripper),
            cartesian=SimpleNamespace(positions=list(_FakeTrossen._CARTESIAN)),
            header=SimpleNamespace(timestamp=self._stamp),
        )

    def set_all_modes(self, mode: Any) -> None:
        self.mode = mode

    def set_cartesian_positions(
        self, goal_positions, interpolation_space, goal_time=2.0, blocking=True, num_trajectory_check_samples=0
    ) -> None:
        if self.mode is not trossen_arm.Mode.position:
            raise trossen_arm.RuntimeError(f'a Cartesian goal needs every joint in position mode, not {self.mode}')
        self.poses.append([float(v) for v in goal_positions])
        self.goal_times.append(float(goal_time))
        self.checked_samples.append(int(num_trajectory_check_samples))

    def set_gripper_position(self, goal_position, goal_time=2.0, blocking=True) -> None:
        if self.mode is not trossen_arm.Mode.position:
            raise trossen_arm.RuntimeError(f'a gripper goal needs the joint in position mode, not {self.mode}')
        self._gripper_goal = float(goal_position)
        self.gripper_goals.append(self._gripper_goal)

    def set_all_positions(self, goal_positions, goal_time=2.0, blocking=True) -> None:
        if self.mode is not trossen_arm.Mode.position:
            raise trossen_arm.RuntimeError(f'set_all_positions needs every joint in position mode, not {self.mode}')
        self.goals.append([float(v) for v in goal_positions])
        self.goal_times.append(float(goal_time))

    def _servo(self) -> None:
        """Advance the joints towards the goal they were last given, as the controller's own loop does.

        A Cartesian goal needs kinematics to follow, which this fake does not have; the fingers still move.
        """
        if not self.goals:
            return
        goal = np.asarray(self.goals[-1])
        if self._gripper_goal is not None:
            goal = np.append(goal[:_ARM_JOINTS], self._gripper_goal)
        step = self._alpha * (goal - self._position)
        self._velocity = step * _HZ
        self._position = self._position + step

    def cleanup(self, reboot_controller: bool = False) -> None:
        self.cleaned_up = True


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Trossen driver smoke: joints and gripper round-trip.')
    parser.add_argument('--ip', default='192.168.1.4')
    parser.add_argument('--fake', action='store_true', help='in-process first-order-lag echo; needs no hardware')
    args = parser.parse_args()

    fake = _FakeTrossen() if args.fake else None
    robot = Robot(args.ip, connect=(lambda ip: fake) if args.fake else _connect)

    with pimm.World() as world:
        # `World.pair` cannot express that it returns the counterpart of the port it is given, so the four
        # payload types are named here.
        commands = world.pair(robot.commands)
        sync_move = world.pair(robot.sync_move)
        target_grip = world.pair(robot.target_grip)
        state = world.pair(robot.state)
        grip = world.pair(robot.grip)

        loop = world.start([robot])

        def pump(seconds: float):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and not world.should_stop:
                cmd = next(loop)
                time.sleep(cmd.seconds if isinstance(cmd, pimm.Sleep) else 0)

        pump(0.2)
        assert state.read() is not None, 'the driver published no state'
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        # Grip round-trip: polarity inverted on the way out (goal) and on the way back (reading).
        target_grip.emit(0.0)
        pump(0.5)
        if fake is not None:
            assert abs(fake.goals[-1][_GRIPPER_JOINT] - 0.04) < 1e-6, fake.goals[-1]  # grip 0 open -> full travel
        assert abs(grip.value) < 0.02, grip.value
        target_grip.emit(1.0)
        pump(0.5)
        if fake is not None:
            assert abs(fake.goals[-1][_GRIPPER_JOINT]) < 1e-6, fake.goals[-1]
        assert abs(grip.value - 1.0) < 0.02, grip.value

        # A streamed joint setpoint nobody waits on.
        jog = np.array([0.2, 0.4, 0.3, 0.0, 0.1, 0.0])
        commands.emit(command.JointPosition(jog))
        pump(0.5)
        assert np.allclose(state.value.q, jog, atol=_ARRIVED_TOL), state.value.q

        # A target outside the joint range is clipped, not refused: joint 1 has no negative half.
        commands.emit(command.JointPosition(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
        pump(0.5)
        assert state.value.q[1] > -_ARRIVED_TOL, state.value.q

        # A synchronous move the firmware plans, answered once the arm reads back at the target.
        home = np.zeros(_ARM_JOINTS)
        answer = sync_move(command.JointPosition(home))
        for _ in range(100):
            if answer.done():
                break
            pump(0.1)
        answer.result()
        assert np.allclose(state.value.q, home, atol=_ARRIVED_TOL), state.value.q
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        print(f'ee_pose {state.value.ee_pose}')
        print('Trossen driver smoke passed')
