"""Driver for the Trossen WidowX AI arm — one Ethernet link carrying six joints plus the gripper.

The controller firmware runs the servo loop, and this driver solves FK/IK itself against the vendored MJCF
(``assets/mujoco/trossen_wxai/wxai_follower.xml``) at ``ee_site`` — the frame the controller reports its own
Cartesian position in. So every command reaches the arm as joints, which is what lets the driver hold each
one to its own velocity limit before it goes out: past that limit the controller faults and drops the arm.

The firmware solves Cartesian goals too, but each one on its own, knowing nothing of the last. Near a
workspace boundary — the arm at rest sits on the lower limit of two joints — successive solutions come from
different branches and the arm tears itself between them. Solving here, warm-started from where the arm
stands, is what keeps the joints continuous.

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

import mujoco as mj
import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers import vendor_import
from positronic.drivers.utils import DriverRun, MoveStatus
from positronic.utils import package_assets_path

from . import RobotStatus, State, command
from .ik import qpos_from_site_pose

# trossen_arm lives in the `trossen` extra, which the type-check environment does not install.
with vendor_import(
    'trossen_arm', 'Trossen arm support', hint='Re-run with the trossen extra:\n  uv run --locked --extra trossen ...\n'
):
    import trossen_arm  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_ARM_JOINTS = 6
_GRIPPER_JOINT = 6
_HZ = 100
# Goal time for a streamed setpoint: one tick, so the firmware interpolates linearly across the gap to the
# next one. At or below 0.001 s it applies the goal as a step instead, and asks the servo for whatever
# acceleration closes the distance at once.
_STREAM_GOAL_TIME_S = 1.0 / _HZ
# What a move somebody waits on travels at. Above 0.2 s of goal time the firmware plans the whole move as a
# quintic, which starts and stops the arm gently — so a move hands it the time and lets it do that.
_MOVE_SPEED = 0.6  # rad/s
_MIN_MOVE_TIME_S = 1.0
_MOVE_TIMEOUT_S = 15.0  # the whole range of a joint at that speed, and time to settle after it
# The share of the following error the controller allows within which the arm counts as arrived. It holds
# itself up with that error, so a tolerance tighter than the droop is one no move ever meets.
_ARRIVED_SHARE = 0.5
# How long the controller's clock may stand still before the link counts as down. Telemetry arrives at
# over 200 Hz, so this is many missed frames, not a scheduling hiccup.
_STALE_AFTER_S = 0.25
_CONNECT_TIMEOUT_S = 20.0
# A reconnect runs on the control loop, so its timeout is what the loop stands still for on a failed attempt.
_RECONNECT_TIMEOUT_S = 1.0
_RECONNECT_AFTER_S = 0.5  # how long the link stays down before a new session is worth opening
_RECONNECT_EVERY_S = 2.0  # and how often another is tried while it stays down
# How often a failure that stands is worth saying again. A tick rate of complaints buries every other line.
_COMPLAIN_EVERY_S = 5.0
# A fault the controller latches is not cleared by opening another session, so the attempts back off rather
# than stall the loop every couple of seconds for as long as the arm stays down.
_RECONNECT_MAX_S = 30.0
# The share of a joint's own velocity limit at which the driver stops driving. Past its limit the controller
# faults and drops the arm to idle, so the driver stands down before it gets there.
_VELOCITY_HEADROOM = 0.8
# The share of a joint's velocity limit a streamed setpoint may ask for. Teleoperation is paced by the hand
# it follows, so this only bounds what one wild target can ask for.
_COMMANDED_SHARE = 0.1
_MJCF_PATH = 'assets/mujoco/trossen_wxai/wxai_follower.xml'
_EE_SITE = 'ee_site'
_JOINT_NAMES = ('joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5')
_IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after clamping
_IK_ROT_TOL = 1e-2  # radians
# How far a streamed Cartesian target may move in one tick. At the tick rate this allows 1.5 m/s, which is
# faster than a hand moves, so it does not shape teleoperation.
_MAX_STEP_M = 0.015
_MAX_STEP_RAD = 0.08
# How far that target may sit from where the arm reads. The arm holds itself up with a following error, so
# it stands off from every pose it is asked for; this is what keeps the standoff from becoming a leash the
# target drags. Wide enough for the droop, and narrow enough that a target cannot run away from an arm.
_MAX_STANDOFF_M = 0.08
_MAX_STANDOFF_RAD = 0.4


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


def _reach_postures(x: float, y: float) -> list[np.ndarray]:
    """IK warm-start candidates for reaching toward base-frame point (x, y).

    Joint 0 swung to the target's azimuth, the arm unfolded to two heights. The arm rests on the lower limit
    of joints 1 and 2, where half the directions have no solution at all, so a seed away from there is what
    lets IK find one.
    """
    azimuth = np.arctan2(y, x)
    return [np.array([azimuth, 1.571, 1.178, 0.0, 0.0, 0.0]), np.array([azimuth, 1.2, 1.5, 0.0, 0.4, 0.0])]


class _Kinematics:
    """FK/IK on the vendored MJCF at ``ee_site``, in the arm base frame.

    ``mujoco`` exports every symbol below from a compiled extension, so a type checker cannot see them.
    """

    def __init__(self):
        self._model = mj.MjModel.from_xml_path(package_assets_path(_MJCF_PATH))
        self._data = mj.MjData(self._model)
        self._site_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, _EE_SITE)
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in _JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in _JOINT_NAMES])
        ranges = np.array([self._model.joint(name).range for name in _JOINT_NAMES])
        self.lower, self.upper = ranges[:, 0], ranges[:, 1]

    def fk(self, q: np.ndarray) -> geom.Transform3D:
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, self._data.site_xmat[self._site_id].copy())
        return geom.Transform3D(self._data.site_xpos[self._site_id].copy(), geom.Rotation.from_quat(quat))

    def ik(
        self, target: geom.Transform3D, current_q: np.ndarray, max_jump: float | np.ndarray | None = None
    ) -> np.ndarray | None:
        """LM IK for ``target``, warm-started from where the arm stands.

        ``max_jump`` bounds how far the solution may sit from ``current_q``, per joint or over all of them,
        and searching stops there: the arm keeps the shape it has, and a pose it can only reach in another
        one comes back as nothing.
        Without it the reach postures are tried too, so the arm may change shape to get there — which
        swings the end effector, and is only for a move somebody asked for and waits on.

        Solutions are clamped into joint range and FK-verified before acceptance, so a target the arm cannot
        reach comes back as nothing rather than as the nearest thing the solver stopped at.

        A target that is reached costs about a quarter of a millisecond, and one that is not costs every
        seed's full search — more than a tick.
        """
        seeds = (current_q,) if max_jump is not None else (current_q, *_reach_postures(*target.translation[:2]))
        for start in seeds:
            self._data.qpos[:] = 0.0
            self._data.qpos[self._qpos_ids] = start
            qpos, _, success = qpos_from_site_pose(
                self._model,
                self._data,
                self._site_id,
                self._dof_ids,
                target.translation,
                target.rotation.as_quat,
                rot_weight=0.5,
            )
            if not success:
                continue
            q = np.clip(qpos[self._qpos_ids].copy(), self.lower, self.upper)
            if max_jump is not None and np.any(np.abs(q - current_q) > max_jump):
                continue
            reached = self.fk(q)
            turn = (reached.rotation.inv * target.rotation).as_rotvec
            angle = float(np.linalg.norm(turn))
            angle = min(angle, 2 * np.pi - angle)
            if np.linalg.norm(reached.translation - target.translation) < _IK_POS_TOL and angle < _IK_ROT_TOL:
                return q
        return None


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
        self._grip_open = float(limits[_GRIPPER_JOINT].position_max)
        self._dq_limit = np.array([limits[i].velocity_max for i in range(_ARM_JOINTS)])
        self._dq_max = self._dq_limit * _VELOCITY_HEADROOM
        self._step_max = self._dq_limit * _COMMANDED_SHARE / _HZ  # what a streamed setpoint may move in a tick
        tolerance = np.array([limits[i].position_tolerance for i in range(_ARM_JOINTS)])
        self._arrived_tol = float(np.min(tolerance) * _ARRIVED_SHARE)
        # How far a streamed target's solution may sit from the joints last asked for. Further than this the
        # same pose is reached with the arm in another shape, and moving into one swings the end effector.
        # The arm holds itself up with a following error, so what it reads stands off from what it was asked
        # for by as much as the controller allows — and a teleoperator's target starts at what the arm reads.
        # A cap tighter than that error refuses every one of those targets, so the error the controller
        # reports is the cap.
        self._jump_max = tolerance
        self._output = driver.get_robot_output()
        self._kin = _Kinematics()
        self._target = np.asarray(self._output.joint.arm.positions, dtype=np.float64)
        self._wanted = self._target.copy()
        self._goal_time = _STREAM_GOAL_TIME_S
        self._anchor: geom.Transform3D | None = None  # the pose last asked for, which the next steps on from
        self._grip_target = self._grip_of(self._output)
        self._arm_unsent, self._grip_unsent = False, False
        # The two halves of the link, which fail apart. Neither is `Moves.errored`, which says the arm is
        # not where the driver put it: a link that drops says nothing about the move.
        self._stream_stale = False  # the controller's clock stands still, so its telemetry stopped arriving
        self._command_dead = False  # a write was refused, and only a new session takes another
        self._down_since: float | None = None
        self._reconnect_at = -_RECONNECT_EVERY_S
        self._reconnect_every = _RECONNECT_EVERY_S
        self._stamp = int(self._output.header.timestamp)
        self._stamp_at = clock.now()
        self.overspeed = False
        self._complaint = ''
        self._complained_at = -_COMPLAIN_EVERY_S

    # TODO(#686): a rate limit on a log line belongs in the logging layer, as `log_every_n_sec`, where every
    # driver reaches it.
    def complain(self, message: str, key: str | None = None) -> None:
        """Say what is wrong, but not on every tick of a fault that stands.

        ``key`` names the fault where the message alone cannot: a refused setpoint carries the pose, which
        differs every tick, and the fault that refuses it does not.
        """
        now = self.clock.now()
        key = key if key is not None else message
        if key == self._complaint and now - self._complained_at < _COMPLAIN_EVERY_S:
            return
        self._complaint, self._complained_at = key, now
        logger.error(message)

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

    def limit_violation(self) -> str:
        """What reads outside the range the controller reports for it, if anything.

        The controller takes a margin past what it reports — a gripper reading a millimetre below its zero
        is driven without complaint — but far enough past, and entering position mode faults it and drops
        the arm, whatever it is then told to do. How much further is not something the SDK says, so this
        only says what it sees, next to the fault it may explain.
        """
        readings = [*self._output.joint.arm.positions, self._output.joint.gripper.position]
        limits = [*zip(self._q_lower, self._q_upper, strict=True), (self._grip_closed, self._grip_open)]
        for i, (value, (lower, upper)) in enumerate(zip(readings, limits, strict=True)):
            if not lower <= value <= upper:
                return f'joint {i} reads {value:.4f}, outside [{lower:.4f}, {upper:.4f}]'
        return ''

    def _take_control(self) -> None:
        """Put the arm in position mode holding where it reads.

        The mode change comes first and the setpoint immediately after, so the servo has a goal from the
        tick it starts servoing. Reading first is what makes a session opened mid-run resume without a jump:
        the arm is wherever it ended up, not where the last session was driving it.
        """
        self._output = self.driver.get_robot_output()
        if outside := self.limit_violation():
            self.complain(f'The arm at {self.ip} may refuse position mode: {outside}')
        self._target = self._wanted = self.q
        self._goal_time = _STREAM_GOAL_TIME_S
        self._grip_target = self._grip_of(self._output)
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self._anchor = None  # wherever the arm is now is what a Cartesian target steps on from
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
            self.complain(f'The arm at {self.ip} did not answer: {exc}')
            self._stream_stale = True
            self._note_link(now)
            return
        stamp = int(self._output.header.timestamp)
        if stamp != self._stamp:
            self._stamp, self._stamp_at = stamp, now
        self._stream_stale = now - self._stamp_at > _STALE_AFTER_S
        self._note_link(now)
        dq = np.abs(np.asarray(self._output.joint.arm.velocities, dtype=np.float64))
        was_overspeed, self.overspeed = self.overspeed, bool(np.any(dq > self._dq_max))
        if self.overspeed and not was_overspeed:
            fastest = int(np.argmax(dq / self._dq_max))
            self.complain(f'Joint {fastest} of the arm at {self.ip} runs at {dq[fastest]:.2f} rad/s; standing down')

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self._output.joint.arm.positions, dtype=np.float64)

    @property
    def ee_pose(self) -> geom.Transform3D:
        """Where the joints put the end effector, in the arm base frame.

        Solved here rather than read from the controller so that the pose that goes out and the pose a
        command is solved against are the same frame by construction. The two agree to 0.13 mm anyway.
        """
        return self._kin.fk(self.q)

    def settle(self) -> None:
        """Judge a move in flight against what the controller reports."""
        if not self.moves.active:
            return
        if self.moves.settle(self.q, self.clock.now()) is MoveStatus.GAVE_UP:
            # Holding the target the arm stopped short of would resume the move once whatever blocked it
            # goes away, long after its asker was told it failed.
            self._target = self._wanted = self.q
            self._goal_time, self._arm_unsent = _STREAM_GOAL_TIME_S, True

    def advance(self) -> None:
        """Move the setpoint one tick's travel towards the joints last asked for.

        A move the firmware is planning owns the setpoint until it arrives: it is making its own trajectory
        there, and a setpoint moved under it would be a new move every tick.
        """
        if self._goal_time != _STREAM_GOAL_TIME_S:
            return
        step = np.clip(self._wanted - self._target, -self._step_max, self._step_max)
        if np.any(step):
            self._target, self._arm_unsent = self._target + step, True

    def hold_grip(self, grip: float) -> None:
        """Hold the fingers at ``grip``."""
        self._grip_target = float(np.clip(grip, 0.0, 1.0))
        self._grip_unsent = True

    @property
    def asked_pose(self) -> geom.Transform3D:
        """The pose the arm was last asked to hold.

        Not the pose it reads: it stands off from what it is given. A step measured from the reading and
        solved from the joints last asked for spans that standoff, and asks for a jump a step never needs.
        """
        return self._anchor if self._anchor is not None else self._kin.fk(self._target)

    @staticmethod
    def _towards(frm: geom.Transform3D, to: geom.Transform3D, max_m: float, max_rad: float) -> geom.Transform3D:
        """``to``, brought within ``max_m`` and ``max_rad`` of ``frm``."""
        step = to.translation - frm.translation
        distance = float(np.linalg.norm(step))
        if distance > max_m:
            step = step * (max_m / distance)
        turn = (frm.rotation.inv * to.rotation).as_rotvec
        angle = float(np.linalg.norm(turn))
        if angle > np.pi:  # `as_rotvec` keeps the way round it was given; the other one is the short way
            turn, angle = turn * (1.0 - 2.0 * np.pi / angle), 2.0 * np.pi - angle
        if angle > max_rad:
            turn = turn * (max_rad / angle)
        return geom.Transform3D(frm.translation + step, frm.rotation * geom.Rotation.from_rotvec(turn))

    def _stepped(self, target: geom.Transform3D) -> geom.Transform3D:
        """``target``, one tick's travel on from the pose the arm was last asked for.

        Stepping on from the last pose asked for, rather than from the pose the arm reads, is what keeps
        this from being a loop. The arm stands off from every pose it is given, by the following error it
        holds itself up with, and that standoff grows as the arm reaches further out — so a target measured
        from the reading walks itself outwards, further every tick.

        The standoff is bounded instead: the pose stepped on from is first pulled back to within reach of
        the arm, so a target still cannot run away from one that is held up. The caller keeps what this
        returns, and only once it has solved for it: an anchor that walked on past a pose with no solution
        would leave every pose after it further out of reach than the last.
        """
        anchor = self._towards(self.ee_pose, self.asked_pose, _MAX_STANDOFF_M, _MAX_STANDOFF_RAD)
        return self._towards(anchor, target, _MAX_STEP_M, _MAX_STEP_RAD)

    # TODO(#685): take which kinematics to use as a constructor argument. The controller solves Cartesian
    # goals itself, through `set_cartesian_positions`, and a station may want that path.
    def _ik(self, pose: geom.Transform3D, *, streamed: bool) -> np.ndarray:
        """The joints that reach ``pose``; raises what the arm cannot reach.

        A streamed pose is paced to a step at a time and solved without letting the arm change shape. One
        somebody waits on is solved as it stands, and may change shape to get there.
        """
        if not streamed:
            solution = self._kin.ik(pose, self.q)
            if solution is None:
                raise ValueError(f'{pose} is out of reach')
            return solution
        # Solved from the joints last asked for, not from the ones read back: the arm stands off from what
        # it is given, and measuring against the reading would make the room for a step have to cover that
        # standoff too — which is room enough to change the arm's shape in.
        stepped = self._stepped(pose)
        solution = self._kin.ik(stepped, self._target, max_jump=self._jump_max)
        if solution is None:
            raise ValueError(f'{pose} is out of reach')
        self._anchor = stepped
        return solution

    def _target_of(self, cmd: command.CommandType, *, streamed: bool = True) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold, clipped to the range the controller reports."""
        # TODO: accept the modes the arm can run instead of leaving them to what a command omits. Its joints
        # are position-servoed, so `PositionControl` names the law already running.
        command.require_native_mode(cmd, 'Trossen')
        match cmd:
            case command.JointPosition(positions):
                target = np.asarray(positions, dtype=np.float64)
            case command.JointDelta(velocities=delta):
                target = (self._target if streamed else self.q) + np.asarray(delta, dtype=np.float64)
            case command.CartesianPosition(pose):
                target = self._ik(pose, streamed=streamed)
            case command.CartesianDelta() as delta_cmd:
                from_pose = self.asked_pose if streamed else self.ee_pose
                target = self._ik(delta_cmd.apply(from_pose), streamed=streamed)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')
        return np.clip(target, self._q_lower, self._q_upper)

    def track(self, cmd: command.CommandType) -> None:
        """Hold the arm at the setpoint ``cmd`` asks for, with nobody waiting on the arrival.

        Held to what a joint may travel in a tick. Teleoperation is paced by the hand it follows, so this
        only bounds what one wild target can ask the arm for.
        """
        self._wanted = self._target_of(cmd)
        self._goal_time = _STREAM_GOAL_TIME_S

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Hold the arm where ``call`` asks; ``settle`` answers it once the controller reads back there.

        The whole move goes to the firmware with the time to make it in, so the firmware plans it: above
        0.2 s of goal time that is a quintic, which starts and stops the arm gently. Nothing this driver
        does itself is as smooth, and a move somebody waits on is exactly where that shows.
        """
        with pimm.calls.raise_to(call):
            target = self._target_of(call.request, streamed=False)
            travel = float(np.max(np.abs(target - self.q)))
            self._target = self._wanted = target
            self._goal_time = max(_MIN_MOVE_TIME_S, travel / _MOVE_SPEED)
            self._arm_unsent = True
            self.moves.accept(call, target, self._arrived_tol, self.clock.now(), _MOVE_TIMEOUT_S)

    def _put_goal(self, setpoint: np.ndarray, move_arm: bool) -> None:
        """Hand the controller this tick's setpoint, fingers included where one call carries both.

        All seven joints are in position mode, which ``set_all_positions`` requires; the fingers take their
        own call where the arm is already being held where it is asked to be.
        """
        grip_m = self._grip_metres(self._grip_target)
        if move_arm:
            self.driver.set_all_positions([*setpoint, grip_m], self._goal_time, False)
        else:
            self.driver.set_gripper_position(grip_m, _STREAM_GOAL_TIME_S, False)

    def write(self) -> None:
        """Put the setpoint on the link, if anything has asked for one since it was last written."""
        if not (self._arm_unsent or self._grip_unsent):
            return
        try:
            self._put_goal(self._target, self._arm_unsent)
        # rules-allow: swallowed-error — a link that refuses a write reads ERROR; the setpoint stays unsent
        # and goes out again on the next session
        except trossen_arm.RuntimeError as exc:
            self.complain(f'The arm at {self.ip} did not take the setpoint: {exc}')
            self._command_dead = True
            self._note_link(self.clock.now())
            return
        self._arm_unsent, self._grip_unsent = False, False
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
        if now - self._reconnect_at < self._reconnect_every:
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
            self.complain(f'The arm at {self.ip} did not take a new session: {exc}')
            self._reconnect_every = min(self._reconnect_every * 2, _RECONNECT_MAX_S)
            return
        self._reconnect_every = _RECONNECT_EVERY_S
        self._stamp, self._stamp_at = int(self._output.header.timestamp), now
        self._stream_stale = False
        self._note_link(now)
        logger.info(f'The arm at {self.ip} answers again')

    def stand_down(self) -> None:
        """Hold the arm where it reads, so nothing is driving it while it runs too fast."""
        self._target = self._wanted = self.q
        self._goal_time, self._arm_unsent = _STREAM_GOAL_TIME_S, True
        self.write()

    def publish(self) -> None:
        """Ship the arm as the controller last reported it, arm and fingers."""
        if self.link_down or self.overspeed or self.moves.errored:  # out of reach, running away, or not
            # where the driver put it
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
                self.robot_meta.emit({keys.ROBOT: 'trossen_wxai'})

                while not should_stop.value:
                    arm.read()
                    if arm.link_down:
                        arm.recover()  # get the arm back first, so the state that goes out says where it is
                        arm.settle()  # a move runs out its deadline on the last reading; nobody waits forever
                        arm.publish()
                        arm.moves.answer()
                        yield arm.limiter.wait()
                        continue

                    if arm.overspeed:  # a joint past its limit faults the controller and drops the arm
                        arm.stand_down()
                        arm.settle()
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
                        try:
                            arm.track(asked)
                        # rules-allow: swallowed-error — a command stream cannot end the run; the next
                        # setpoint supersedes this one
                        except Exception as exc:
                            arm.complain(f'{asked} not applied: {exc}', key='setpoint refused')

                    arm.advance()
                    arm.write()
                    arm.publish()
                    arm.moves.answer()  # the state a settled move is answered with is out

                    yield arm.limiter.wait()


class _FakeTrossen:
    """First-order-lag echo of the 7-joint arm, so the ``--fake`` smoke runs without hardware.

    Duck-types the slice of ``TrossenArmDriver`` the driver uses. It models the link and the servo; the
    kinematics are the driver's own, so the joints it reports are the whole of what it says.
    """

    # What the arm reports for itself, read off a wxai_v0 controller on firmware 1.11.1
    _LIMITS = [
        (-3.141593, 3.141593, 6.2832, 0.2),
        (0.0, 3.141593, 6.2832, 0.2),
        (0.0, 2.356194, 6.2832, 0.2),
        (-1.570796, 1.570796, 9.4248, 0.4),
        (-1.570796, 1.570796, 9.4248, 0.4),
        (-3.141593, 3.141593, 9.4248, 0.4),
        (0.0, 0.04, 0.25, 0.004),
    ]

    class _Limit:
        def __init__(self, lower: float, upper: float, velocity_max: float, position_tolerance: float):
            self.position_min = lower
            self.position_max = upper
            self.velocity_max = velocity_max
            self.position_tolerance = position_tolerance

    _TICK_US = 5000  # the controller streams faster than the driver reads, so its clock moves every read

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._position = np.zeros(7)  # the arm boots with the fingers closed
        self._velocity = np.zeros(7)
        self._stamp = 0
        self.frozen = False  # the controller stops being heard from, as a dropped link leaves it
        self.sessions = 1
        self.mode: Any = None
        self.goals: list[list[float]] = []
        self.goal_times: list[float] = []
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
        return [_FakeTrossen._Limit(*limit) for limit in _FakeTrossen._LIMITS]

    def get_robot_output(self) -> Any:
        if not self.frozen:
            self._servo()
            self._stamp += _FakeTrossen._TICK_US
        arm = SimpleNamespace(positions=self._position[:_ARM_JOINTS].copy(), velocities=self._velocity[:6].copy())
        gripper = SimpleNamespace(position=float(self._position[_GRIPPER_JOINT]))
        return SimpleNamespace(
            joint=SimpleNamespace(arm=arm, gripper=gripper), header=SimpleNamespace(timestamp=self._stamp)
        )

    def set_all_modes(self, mode: Any) -> None:
        self.mode = mode

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
        # Half of what each joint may do, which is the headroom a servo keeps when it is not faulting.
        per_tick = np.array([limit[2] for limit in _FakeTrossen._LIMITS]) / (2 * _HZ)
        step = np.clip(step, -per_tick, per_tick)
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

    _ARRIVED_SLACK = 0.1  # what the checks below allow, being the tolerance the wxai_v0 controller reports
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
        if fake is not None:  # grip 0 open -> the joint at the far end of its travel
            assert abs(fake.gripper_goals[-1] - 0.04) < 1e-6, fake.gripper_goals[-1]
        assert abs(grip.value) < 0.02, grip.value
        target_grip.emit(1.0)
        pump(0.5)
        if fake is not None:
            assert abs(fake.gripper_goals[-1]) < 1e-6, fake.gripper_goals[-1]
        assert abs(grip.value - 1.0) < 0.02, grip.value

        # A streamed joint setpoint nobody waits on.
        jog = np.array([0.2, 0.4, 0.3, 0.0, 0.1, 0.0])
        commands.emit(command.JointPosition(jog))
        pump(0.5)
        assert np.allclose(state.value.q, jog, atol=_ARRIVED_SLACK), state.value.q

        # A target outside the joint range is clipped, not refused: joint 1 has no negative half.
        commands.emit(command.JointPosition(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
        pump(0.5)
        assert state.value.q[1] > -_ARRIVED_SLACK, state.value.q

        # A synchronous move the firmware plans, answered once the arm reads back at the target.
        home = np.zeros(_ARM_JOINTS)
        answer = sync_move(command.JointPosition(home))
        for _ in range(100):
            if answer.done():
                break
            pump(0.1)
        answer.result()
        assert np.allclose(state.value.q, home, atol=_ARRIVED_SLACK), state.value.q
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        print(f'ee_pose {state.value.ee_pose}')
        print('Trossen driver smoke passed')
