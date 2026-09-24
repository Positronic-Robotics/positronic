"""Driver for the real i2rt YAM arm — one CAN chain carrying six joints plus the gripper.

i2rt exposes joint-space position-PD with gravity compensation only (its own ~100 Hz control thread), so this
driver solves FK/IK itself against the vendored MJCF (``assets/mujoco/i2rt_yam/yam.xml``) at ``DEFAULT_FRAME`` —
the control frame the training data is expressed in. The upstream MJCF package is vendored whole
(``scene.xml`` and meshes included); the driver itself loads only ``yam.xml``. The gripper is the chain's 7th
DOF, normalized 0=closed/1=open — the inverse of positronic's grip convention — so grip values are inverted in
both directions.

Station bring-up is not verifiable off-hardware and must be re-checked on the rig: CAN interface up
(``ip link set can0 up type can bitrate 1000000``), motor zero calibration, kp/kd gains, the gravity
compensation each joint needs (``gravity_comp_factor``), physical gripper
polarity and joint-range check, mount pose survey (``base_pose``), teleop latency, and the arm going limp
on close (``zero_torque_mode``).
"""

import contextlib
import logging
import math
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import mujoco as mj
import numpy as np

import pimm
from positronic import geom
from positronic.drivers import vendor_import
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.utils import DriverRun, MoveAbandoned, log_failure
from positronic.utils import package_assets_path

from . import RobotStatus, State, command
from .ik import qpos_from_site_pose
from .models import DEFAULT_FRAME
from .settle import MOVE_SETTLE, PARK_SETTLE, SettleTuning

# i2rt lives in the `yam` extra, which the type-check environment does not install.
with vendor_import('i2rt', 'YAM support', hint='Re-run with the yam extra:\n  uv run --locked --extra yam ...\n'):
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface  # pyright: ignore[reportMissingImports]
    from i2rt.robots.get_robot import get_yam_robot  # pyright: ignore[reportMissingImports]
    from i2rt.robots.utils import GripperType  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# The driver solves FK/IK itself, so its joint order and control frame must match the YAM sim's.
# TODO(#517): centralise driver kinematics so driver and sim share one module.
_JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
_MJCF_PATH = 'assets/mujoco/i2rt_yam/yam.xml'
_IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after limit clamping
_IK_ROT_TOL = 1e-2  # radians
# Joints 2 and 3 rest on their lower mechanical stops at zero.
_PARK_JOINTS = np.zeros(6)
_OPEN_GRIP = 0.0  # positronic grip convention: 0 is open
# The vendor's observation contract
_JOINT_POS, _JOINT_VEL, _GRIPPER_POS = 'joint_pos', 'joint_vel', 'gripper_pos'


def _connect(channel: str, sim: bool, gravity_comp_factor: np.ndarray | None):
    """Open the i2rt chain in position-PD mode; ``sim=True`` runs i2rt's own MuJoCo sim instead of hardware."""
    return get_yam_robot(
        channel,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=False,
        sim=sim,
        gravity_comp_factor=gravity_comp_factor,
    )


class YamState(State, pimm.shared_memory.NumpySMAdapter):
    Q_OFFSET = 0
    DQ_OFFSET = Q_OFFSET + 6
    EE_POSE_OFFSET = DQ_OFFSET + 6
    STATUS_OFFSET = EE_POSE_OFFSET + 7
    TOTAL = STATUS_OFFSET + 1

    def __init__(self):
        super().__init__(shape=(YamState.TOTAL,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 7].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[YamState.STATUS_OFFSET]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus):
        self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6] = q
        self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6] = dq
        self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 3] = ee_pose.translation
        self.array[YamState.EE_POSE_OFFSET + 3 : YamState.EE_POSE_OFFSET + 7] = ee_pose.rotation.as_quat
        self.array[YamState.STATUS_OFFSET] = status.value


class _Kinematics:
    """FK/IK on the vendored YAM MJCF at ``DEFAULT_FRAME``, in the arm-base frame.

    ``mujoco`` exports every symbol below from a compiled extension, so a type checker cannot see them.
    """

    def __init__(self):
        model_path = package_assets_path(_MJCF_PATH)
        self._model = mj.MjModel.from_xml_path(model_path)
        self._data = mj.MjData(self._model)
        site = mj.mjtObj.mjOBJ_SITE
        self._site_id = mj.mj_name2id(self._model, site, DEFAULT_FRAME)
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in _JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in _JOINT_NAMES])
        ranges = np.array([self._model.joint(name).range for name in _JOINT_NAMES])
        self._lower, self._upper = ranges[:, 0], ranges[:, 1]

    @property
    def lower(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper(self) -> np.ndarray:
        return self._upper.copy()

    def fk(self, q: np.ndarray) -> geom.Transform3D:
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, self._data.site_xmat[self._site_id].copy())
        return geom.Transform3D(self._data.site_xpos[self._site_id].copy(), geom.Rotation.from_quat(quat))

    @staticmethod
    def _reach_postures(x: float, y: float) -> list[np.ndarray]:
        """IK warm-start candidates for reaching toward arm-base-frame point (x, y): joint1 swung to the target's
        azimuth, elbow folded down at two heights. The 6-DoF wrist gives LM no null space to escape bad basins,
        so seeding near the goal is what makes limit-clamped IK reliable."""
        az = np.arctan2(y, x)
        return [np.array([az, 1.8, 2.2, 0.0, -0.9, 0.0]), np.array([az, 1.2, 1.2, 0.0, 0.6, 0.0])]

    def ik(self, target: geom.Transform3D, current_q: np.ndarray) -> np.ndarray | None:
        """Multi-start LM IK: the live posture first, then the reach postures toward the target's azimuth.
        Solutions are wrapped and clamped into joint range, then FK-verified before acceptance."""
        for start in (current_q, *self._reach_postures(*target.translation[:2])):
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
            q = qpos[self._qpos_ids].copy()
            # A revolute joint at q ± 2π is the same pose; wrap out-of-range entries back in when they fit.
            q = np.where(q > self._upper, q - 2 * np.pi, q)
            q = np.where(q < self._lower, q + 2 * np.pi, q)
            q = np.clip(q, self._lower, self._upper)
            reached = self.fk(q)
            rot_err = (reached.rotation.inv * target.rotation).angle
            rot_err = min(rot_err, 2 * np.pi - rot_err)
            if np.linalg.norm(reached.translation - target.translation) < _IK_POS_TOL and rot_err < _IK_ROT_TOL:
                return q
        return None


class _Arm(DriverRun[command.CommandType]):
    """Control and report one YAM arm and its gripper for a driver run."""

    def __init__(
        self,
        vendor: Any,
        sync_move: pimm.calls.ControlSystemHandler[command.CommandType, None],
        async_move: pimm.SignalReceiver[command.CommandType],
        out: pimm.SignalEmitter[YamState],
        grip_out: pimm.SignalEmitter[float],
        base_pose: geom.Transform3D,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
        park_tuning: SettleTuning,
        move_tuning: SettleTuning,
        kinematics: _Kinematics,
        state: YamState,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=100)
        self.vendor = vendor
        self.park_tuning = park_tuning
        self.move_tuning = move_tuning
        self.out = out
        self.grip_out = grip_out
        self.state = state
        self._base_pose = base_pose
        self._kin = kinematics
        self._shutting_down = False
        self._publish_failure_logged = False

    def observations(self) -> dict[str, np.ndarray]:
        """Read the current joint and gripper measurements."""
        return self.vendor.get_observations()

    @staticmethod
    def _grip(obs: dict[str, np.ndarray]) -> float:
        """Convert measured open width to the closed-fraction grip convention."""
        return 1.0 - float(obs[_GRIPPER_POS][0])

    def publish(self, obs: dict[str, np.ndarray], status: RobotStatus | None = None) -> None:
        """Publish measured state; default to ERROR after a failed move, otherwise AVAILABLE.

        During shutdown a failed publish is logged and dropped, so only the chain can stop the park.
        """
        if status is None:
            status = RobotStatus.ERROR if self.moves.errored else RobotStatus.AVAILABLE
        q = obs[_JOINT_POS]
        self.state.encode(q, obs[_JOINT_VEL], self._base_pose * self._kin.fk(q), status)
        # rules-allow: swallowed-error — in shutdown a report must not decide whether the arm is let go
        try:
            self.out.emit(self.state)
            self.grip_out.emit(self._grip(obs))
        except Exception:
            if not self._shutting_down:
                raise
            if not self._publish_failure_logged:
                self._publish_failure_logged = True
                logger.exception('Publishing the arm state failed during shutdown; the park goes on without it')

    def command_target(self, joints: np.ndarray, grip: float) -> None:
        """Append the gripper target in the vendor's open-width convention."""
        self.vendor.command_joint_pos(np.append(joints, 1.0 - grip))

    def hold_where_it_stopped(self) -> tuple[np.ndarray, float]:
        """Hold the measured joint and gripper positions, publish state, and return the hold target."""
        obs = self.observations()
        self.vendor.command_joint_pos(np.append(obs[_JOINT_POS], obs[_GRIPPER_POS][0]))
        self.publish(obs)
        return np.asarray(obs[_JOINT_POS], dtype=np.float64), self._grip(obs)

    def _ik(self, world_pose: geom.Transform3D, q: np.ndarray) -> np.ndarray:
        """IK in the arm-base frame."""
        solution = self._kin.ik(self._base_pose.inv * world_pose, q)
        if solution is None:
            raise ValueError(f'{world_pose} is out of reach')
        return solution

    def to_joints(self, cmd: command.CommandType, q: np.ndarray) -> np.ndarray:
        """Convert a command to joint targets; reject unsupported modes and unreachable poses."""
        # TODO: accept the modes the arm can run instead of leaving them to what a command omits. Its
        # joints are position-servoed, so `PositionControl` names the rule already running.
        command.require_native_mode(cmd, 'YAM')
        match cmd:
            case command.JointPosition(positions):
                return np.asarray(positions, dtype=np.float64)
            case command.JointDelta(velocities=delta):
                return q + np.asarray(delta, dtype=np.float64)
            case command.CartesianPosition(pose):
                return self._ik(pose, q)
            case command.CartesianDelta() as delta_cmd:
                return self._ik(delta_cmd.apply(self._base_pose * self._kin.fk(q)), q)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def _arrived(self, obs: dict[str, np.ndarray], target: np.ndarray, grip: float, tuning: SettleTuning) -> bool:
        if not np.all(np.abs(obs[_JOINT_POS] - target) < tuning.tolerance_rad):
            return False
        return abs(self._grip(obs) - grip) < tuning.grip_tolerance

    def _move_timeout(
        self, obs: dict[str, np.ndarray], target: np.ndarray, grip: float, timeout_s: float, tolerance_rad: float
    ) -> TimeoutError:
        joint_error = np.max(np.abs(obs[_JOINT_POS] - target))
        joint_speed = np.max(np.abs(obs[_JOINT_VEL]))
        return TimeoutError(
            f'joint error {joint_error:.4f} rad (tolerance {tolerance_rad:.4f}) after '
            f'{timeout_s:g}s; target={target}, measured={obs[_JOINT_POS]}, '
            f'max joint speed={joint_speed:.4f} rad/s; '
            f'grip target={grip:.3f}, measured={self._grip(obs):.3f}'
        )

    def _ramp(
        self, start: np.ndarray, target: np.ndarray, grip: float, fraction: float, obs: dict[str, np.ndarray]
    ) -> None:
        fraction = min(fraction, 1.0)
        self.command_target((1 - fraction) * start + fraction * target, grip)
        self.publish(obs, RobotStatus.BUSY)

    class _Rest(Enum):
        """Where a settle pass left the chain once every joint held still."""

        ON_GOAL = auto()
        SHORT_OF_GOAL = auto()

    def _come_to_rest(
        self, reference: np.ndarray, goal: np.ndarray, grip: float, tuning: SettleTuning, *, interrupt_on_stop: bool
    ) -> Generator[pimm.Command, None, tuple[dict[str, np.ndarray], _Rest] | None]:
        """Ramp to ``reference`` at the tuning's pace, then wait until every joint holds still.

        The chain is still when each joint's measured position spans at most ``tuning.still_position_rad`` over
        the last ``tuning.still_time_s``. The velocity readings play no part: a real chain reports speed spikes
        at rest. Return the reading and where the chain rests. ``ON_GOAL`` needs every reading in that window
        within tolerance. Return None when a stop abandons the move.
        """
        start = np.asarray(self.observations()[_JOINT_POS], dtype=np.float64)
        travel_s = max(tuning.min_ramp_s, float(np.max(np.abs(reference - start))) / tuning.max_speed_rad_s)
        timeout_s = travel_s + tuning.settle_timeout_s
        window: deque[tuple[float, np.ndarray, bool]] = deque()  # (elapsed, joints, on goal) since the ramp ended
        started = self.clock.now()
        while True:
            if self.should_stop.value and interrupt_on_stop:
                return None
            elapsed = self.clock.now() - started
            obs = self.observations()
            if elapsed > timeout_s:
                raise self._move_timeout(obs, goal, grip, timeout_s, tuning.tolerance_rad)

            if elapsed >= travel_s:
                window.append((
                    elapsed,
                    np.asarray(obs[_JOINT_POS], dtype=np.float64),
                    self._arrived(obs, goal, grip, tuning),
                ))
                while len(window) > 1 and window[1][0] <= elapsed - tuning.still_time_s:
                    window.popleft()
                if self._still(window, elapsed, tuning):
                    if all(on_goal for _, _, on_goal in window):
                        return obs, self._Rest.ON_GOAL
                    if not window[-1][2]:
                        return obs, self._Rest.SHORT_OF_GOAL

            self._ramp(start, reference, grip, elapsed / travel_s, obs)
            yield self.limiter.wait()

    @staticmethod
    def _still(window: deque[tuple[float, np.ndarray, bool]], elapsed: float, tuning: SettleTuning) -> bool:
        """Whether the window spans ``still_time_s`` and every joint's position spread stays within bounds."""
        if window[0][0] > elapsed - tuning.still_time_s:
            return False
        joints = np.stack([q for _, q, _ in window])
        return bool(np.all(np.ptp(joints, axis=0) <= tuning.still_position_rad))

    def _settle_onto(
        self, goal: np.ndarray, grip: float, tuning: SettleTuning, *, interrupt_on_stop: bool, within_joint_limits: bool
    ) -> Generator[pimm.Command, None, np.ndarray | None]:
        """Settle the chain onto ``goal`` and return the reference that holds it there.

        The servo holds the chain a steady distance short of its reference, so one ramp leaves the joints
        short of ``goal``. Each pass takes the measured gap off the reference the chain already holds. The
        chain gives back only part of each correction, so the corrections must add up: a reference computed
        from ``goal`` and the latest gap alone swings and does not land. ``within_joint_limits`` keeps the
        reference inside the modeled joint ranges; the park leaves it off, to press joints 2 and 3 onto their
        stops. Return None when a stop abandons the move. Raise ``TimeoutError`` when the passes run out, or
        when the bounds stop a further correction.
        """
        try:
            reference = goal.copy()
            for _ in range(tuning.attempts):
                rest = yield from self._come_to_rest(reference, goal, grip, tuning, interrupt_on_stop=interrupt_on_stop)
                if rest is None:
                    return None
                obs, rests = rest
                if rests is self._Rest.ON_GOAL:
                    self.command_target(reference, grip)
                    self.moves.errored = False
                    return reference
                bound = tuning.max_correction_rad
                corrected = np.clip(reference - (obs[_JOINT_POS] - goal), goal - bound, goal + bound)
                if within_joint_limits:
                    corrected = np.clip(corrected, self._kin.lower, self._kin.upper)
                if np.array_equal(corrected, reference):
                    break
                reference = corrected
            obs = self.observations()
            raise TimeoutError(
                f'the arm rests {np.max(np.abs(obs[_JOINT_POS] - goal)):.4f} rad from {goal} '
                f'(tolerance {tuning.tolerance_rad:.4f}); reference={reference}, measured={obs[_JOINT_POS]}, '
                f'grip target={grip:.3f}, measured={self._grip(obs):.3f}'
            )
        except Exception:
            self.moves.errored = True
            raise

    class _Park(Enum):
        """How a park ended: on the parking pose, or holding wherever the arm stopped."""

        PARKED = auto()
        HELD_WHERE_IT_STOPPED = auto()

    def park(
        self, grip: float, *, interrupt_on_stop: bool = True
    ) -> Generator[pimm.Command, None, tuple[np.ndarray, float, _Park]]:
        """Settle onto the parking pose at bounded speed; return the joints and grip to hold, and how it ended."""
        logger.info('Moving the arm to the parking pose')
        try:
            reference = yield from self._settle_onto(
                _PARK_JOINTS, grip, self.park_tuning, interrupt_on_stop=interrupt_on_stop, within_joint_limits=False
            )
            if reference is not None:
                logger.info('Arm parked')
                self._report_parked()
                return reference, grip, self._Park.PARKED
        # rules-allow: swallowed-error — an arm that will not park reads ERROR; it does not end the run
        except Exception as exc:
            self.moves.errored = True
            logger.error(f'The arm did not reach the parking pose: {exc}')
        return *self.hold_where_it_stopped(), self._Park.HELD_WHERE_IT_STOPPED

    def _report_parked(self) -> None:
        """Publish the parked state. The park is verified already, so a failed report cannot undo it."""
        # rules-allow: swallowed-error — the verdict stands; the report's failure is logged
        try:
            self.publish(self.observations())
        except Exception:
            logger.exception('The arm is parked, but its state could not be published')

    def shutdown(self) -> Generator[pimm.Command, None, None]:
        self._shutting_down = True
        hold_target = None
        try:
            joints, grip, ended = yield from self.park(self._grip(self.observations()), interrupt_on_stop=False)
            if ended is self._Park.PARKED:
                return
            hold_target = joints, grip
        # rules-allow: swallowed-error — any failure before verified parking must block torque release
        except Exception:
            self.moves.errored = True
            logger.exception('Could not verify parking; keeping the arm powered')

        logger.critical('Parking failed; arm still powered. Shutdown blocked: operator assistance required.')
        while True:
            try:
                if hold_target is None:
                    hold_target = self.hold_where_it_stopped()
                q, grip = hold_target
                self.command_target(q, grip)
                self.publish(self.observations())
            # rules-allow: swallowed-error — a failed hold must keep the connection open and shutdown blocked
            except Exception:
                logger.exception('Could not hold the arm; shutdown remains blocked')
            yield self.limiter.wait()

    def sync_move(
        self, call: pimm.calls.Call[command.CommandType, None], q: np.ndarray
    ) -> Generator[pimm.Command, None, tuple[np.ndarray, float]]:
        """Settle onto the move's target with the gripper open, and answer its caller; hold the measured position
        if the move fails or stops. A blocking move is the framework's reset, so an episode starts open-handed."""
        try:
            target = self.to_joints(call.request, q)
            reference = yield from self._settle_onto(
                target, _OPEN_GRIP, self.move_tuning, interrupt_on_stop=True, within_joint_limits=True
            )
            if reference is not None:
                self.publish(self.observations())
                call.set_result(None)
                return reference, _OPEN_GRIP
        except Exception as exc:
            try:
                held = self.hold_where_it_stopped()
            finally:
                call.set_exception(exc)  # Answer even if reading the hold position fails.
            return held
        held = self.hold_where_it_stopped()
        call.set_exception(MoveAbandoned())
        return held


@contextlib.contextmanager
def _opened(
    connect: Callable[[str, bool, np.ndarray | None], Any],
    channel: str,
    sim: bool,
    gravity_comp_factor: np.ndarray | None,
) -> Iterator[Any]:
    """Release torque only after the driver completes its parking shutdown."""
    vendor = connect(channel, sim, gravity_comp_factor)
    try:
        yield vendor
    except BaseException:
        logger.critical('Driver interrupted before verified parking; leaving the motor connection open')
        raise
    else:
        try:
            vendor.zero_torque_mode()
        finally:
            try:
                vendor.close()
            finally:
                # Only after `close()`: it joins i2rt's control thread, which fails on a motor disabled under it.
                _power_off(vendor)


_POWER_OFF_ATTEMPTS = 3  # per motor; a motor can miss the first disable it is sent


def _disable_motors(interface: Any, motor_ids: list[int]) -> list[int]:
    """Send each motor its disable, retried; return the motors that refused every attempt."""
    refused = []
    for motor_id in motor_ids:
        for _ in range(_POWER_OFF_ATTEMPTS):
            # rules-allow: swallowed-error — a motor that will not answer must not leave the rest of them on
            try:
                interface.motor_off(motor_id)
                break
            except Exception:
                pass
        else:
            refused.append(motor_id)
    return refused


def _power_off(vendor: Any) -> None:
    """Disable the chain's motors once ``close()`` has joined i2rt's control thread.

    A chain left limp keeps its motors enabled, so each one reaches its own command timeout and latches an
    error. i2rt enables them one at a time and offers no matching disable, so this sends the per-motor disable
    over an interface of its own: ``close()`` has already shut the chain's.
    """
    chain: Any = getattr(vendor, 'motor_chain', None)
    if chain is None or not getattr(chain, 'motor_list', None):
        return  # i2rt's own sim chain and the fakes carry no motors
    motor_ids = [motor_id for motor_id, _ in chain.motor_list]
    # rules-allow: swallowed-error — the run is over, and a chain that will not answer must not hide what ended it
    try:
        interface = DMSingleMotorCanInterface(
            channel=chain.channel, control_mode=chain.motor_interface.control_mode, name='power-off'
        )
        try:
            refused = _disable_motors(interface, motor_ids)
        finally:
            interface.close()
        if refused:
            logger.warning(f'Motors {refused} stayed enabled, so they will latch their own command timeout')
    except Exception as exc:
        logger.warning(f'The chain kept its motors enabled: {exc}')


@dataclass
class _Serving:
    """What the command loop carries from one tick to the next."""

    q_target: np.ndarray
    grip_target: float
    idle_since: float | None = None
    parking: Generator[pimm.Command, None, tuple[np.ndarray, float, _Arm._Park]] | None = None


class Robot(pimm.ControlSystem):
    """Drives one YAM chain: FK/IK in the driver, joint-space position-PD on the arm.

    ``base_pose`` places the arm base in the world frame (identity = arm-base frame): IK targets are pulled
    back through it and the emitted ``ee_pose`` is pushed forward, so a bimanual embodiment can mount both
    arms in the training world frame. The gripper shares the CAN chain, so the arm driver carries the
    ``grip``/``target_grip`` ports (SO-101 precedent).
    """

    shutdown_policy = pimm.ShutdownPolicy.WAIT_FOR_COMPLETION

    def __init__(
        self,
        channel: str = 'can0',
        *,
        base_pose: geom.Transform3D | None = None,
        sim: bool = False,
        park_after_idle_s: float | None = 60.0,
        park_tuning: SettleTuning = PARK_SETTLE,
        move_tuning: SettleTuning = MOVE_SETTLE,
        gravity_comp_factor: Sequence[float] | None = None,
        connect: Callable = _connect,
    ) -> None:
        """
        :param channel: SocketCAN interface of the arm (e.g. ``can0``). Ignored in sim mode.
        :param base_pose: Arm-base mount pose in the world frame; None keeps everything in the arm-base frame.
        :param sim: Run against i2rt's own MuJoCo sim instead of hardware.
        :param park_after_idle_s: Park after this many seconds without an arm or gripper command.
            For synchronous moves, count from completion.
            None disables idle parking. The driver parks on startup and normal shutdown regardless.
        :param park_tuning: How the park settles onto the parking pose on this arm (``SettleTuning``).
        :param move_tuning: How a blocking ``sync_move`` settles onto its target on this arm. Streamed commands
            go to the chain as they come, with no ramp and no correction.
        :param gravity_comp_factor: One factor per arm joint, scaling the gravity torque i2rt compensates.
            None keeps i2rt's own. A joint that reads a steady offset below where it was sent is under-
            compensated, and the offset is what it carries divided by its position gain.
        :param connect: ``(channel, sim, gravity_comp_factor) -> i2rt Robot`` factory; the fake-mode smoke
            injects ``_FakeYam``.
        """
        if park_after_idle_s is not None and (not math.isfinite(park_after_idle_s) or park_after_idle_s <= 0):
            raise ValueError('park_after_idle_s must be finite and positive, or None')
        self._park_after_idle_s = park_after_idle_s
        self._park_tuning = park_tuning
        self._move_tuning = move_tuning
        self._channel = channel
        self._base_pose = base_pose if base_pose is not None else geom.Transform3D.identity
        self._sim = sim
        self._gravity_comp_factor = None if gravity_comp_factor is None else np.asarray(gravity_comp_factor, float)
        self._connect = connect

        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.state = pimm.ControlSystemEmitter[YamState](self)
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.robot_meta = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def _should_park(self, idle_since: float | None, now: float) -> bool:
        return (
            idle_since is not None
            and self._park_after_idle_s is not None
            and now - idle_since >= self._park_after_idle_s
        )

    @staticmethod
    @contextlib.contextmanager
    def _answer_failed_setup(asked: pimm.calls.Call | command.CommandType | None) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:
            if isinstance(asked, pimm.calls.Call):
                asked.set_exception(exc)
            raise

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        # Built before the chain opens, so nothing between enabling the motors and the protected region can raise.
        kinematics, state = _Kinematics(), YamState()
        meta = {
            'robot': 'i2rt_yam',
            roboarm_keys.JOINT_NAMES: list(_JOINT_NAMES),
            roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME,
        }
        fault = None
        with _opened(self._connect, self._channel, self._sim, self._gravity_comp_factor) as vendor:
            arm = _Arm(
                vendor,
                self.sync_move,
                self.commands,
                self.state,
                self.grip,
                self._base_pose,
                should_stop,
                clock,
                self._park_tuning,
                self._move_tuning,
                kinematics,
                state,
            )
            # rules-allow: swallowed-error — the fault is raised again once the arm is parked and let go
            try:
                self.robot_meta.emit(meta)
                yield from self._serve(arm, should_stop, clock)
            except Exception as exc:
                fault = exc
                logger.exception('The arm driver failed during the run; parking the arm before it lets go')
            yield from arm.shutdown()
        if fault is not None:
            raise fault

    def _serve(
        self, arm: _Arm, should_stop: pimm.SignalReceiver, clock: pimm.Clock
    ) -> Generator[pimm.Command, None, None]:
        """Park on startup, then answer commands and park when idle until ``should_stop``."""
        joints, grip, _ = yield from arm.park(arm._grip(arm.observations()))
        serving = _Serving(joints, grip)
        try:
            while not should_stop.value:
                asked, q = self._take_request(arm, serving, clock)
                yield from self._dispatch(arm, serving, asked, q, clock)
                if (step := self._advance_parking(serving)) is not None:
                    yield step
                    continue
                arm.command_target(serving.q_target, serving.grip_target)
                # Synchronous moves can take seconds; publish a fresh observation.
                arm.publish(arm.observations())
                yield arm.limiter.wait()
        finally:
            if serving.parking is not None:
                serving.parking.close()

    def _take_request(
        self, arm: _Arm, serving: _Serving, clock: pimm.Clock
    ) -> tuple[pimm.calls.Call | command.CommandType | None, np.ndarray]:
        """Read the grip and the next request; a new one interrupts an idle park. Return it with the joints."""
        grip = pimm.value_updated(self.target_grip)
        asked = arm.moves.next_request()
        with self._answer_failed_setup(asked):
            if serving.parking is not None and (grip is not None or asked is not None):
                serving.parking.close()
                serving.parking = None
                serving.q_target, serving.grip_target = arm.hold_where_it_stopped()
            if grip is not None:
                serving.grip_target = float(grip)
                serving.idle_since = clock.now()
            return asked, arm.observations()[_JOINT_POS]

    def _dispatch(
        self,
        arm: _Arm,
        serving: _Serving,
        asked: pimm.calls.Call | command.CommandType | None,
        q: np.ndarray,
        clock: pimm.Clock,
    ) -> Generator[pimm.Command, None, None]:
        """Run a blocking move, take a streamed target, or start an idle park."""
        if isinstance(asked, pimm.calls.Call):
            serving.q_target, serving.grip_target = yield from arm.sync_move(asked, q)
            serving.idle_since = clock.now()
        elif asked is not None:
            with log_failure(asked):
                serving.q_target = arm.to_joints(asked, q)
            serving.idle_since = clock.now()
        elif self._should_park(serving.idle_since, clock.now()):
            serving.parking = arm.park(arm._grip(arm.observations()))
            serving.idle_since = None

    @staticmethod
    def _advance_parking(serving: _Serving) -> pimm.Command | None:
        """Step an idle park; return its command, or None once it has ended or when none is running."""
        if serving.parking is None:
            return None
        try:
            return next(serving.parking)
        except StopIteration as done:
            serving.q_target, serving.grip_target, _ = done.value
            serving.parking = None
            return None


class _FakeYam:
    """First-order-lag echo of the 7-DOF chain (6 joints + normalized gripper, 0=closed/1=open).

    Duck-types the slice of the runtime-checkable ``i2rt.robots.robot.Robot`` protocol the driver uses, so
    the ``--fake`` smoke runs without hardware.
    """

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._pos = np.append(np.zeros(6), 1.0)  # the arm boots with the gripper open
        self._vel = np.zeros(7)
        self.last_command: np.ndarray | None = None

    def num_dofs(self) -> int:
        return 7

    def get_observations(self) -> dict[str, np.ndarray]:
        return {
            _JOINT_POS: self._pos[:6].copy(),
            _JOINT_VEL: self._vel[:6].copy(),
            _GRIPPER_POS: self._pos[6:7].copy(),
            'gripper_vel': self._vel[6:7].copy(),
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.last_command = np.asarray(joint_pos, dtype=np.float64).copy()
        step = self._alpha * (self.last_command - self._pos)
        self._vel = step * 100.0  # commands arrive at the driver's 100 Hz
        self._pos = self._pos + step

    def zero_torque_mode(self) -> None:
        self._vel = np.zeros(7)

    def close(self) -> None:
        pass


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='YAM driver smoke: drives a Cartesian square and checks round-trips.')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--fake', action='store_true', help='in-process first-order-lag echo; needs no hardware')
    parser.add_argument('--sim', action='store_true', help="i2rt's own MuJoCo sim instead of the CAN chain")
    args = parser.parse_args()

    fake = _FakeYam() if args.fake else None
    fake_connect = (lambda channel, sim, gravity_comp_factor: fake) if args.fake else _connect
    robot = Robot(args.channel, sim=args.sim, connect=fake_connect)

    with pimm.World() as world:
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

        pump(0.1)
        while state.read() is None or state.value.status == RobotStatus.BUSY:
            pump(0.1)  # the opening move ramps the arm to the park pose over a couple of seconds
        assert state.value.status == RobotStatus.AVAILABLE, state.value.status

        kin = _Kinematics()

        if fake is not None:
            # State round-trip: the parked chain comes back through the driver's FK.
            assert np.allclose(state.value.q, _PARK_JOINTS, atol=PARK_SETTLE.tolerance_rad), state.value.q
            park_err = np.linalg.norm(state.value.ee_pose.translation - kin.fk(_PARK_JOINTS).translation)
            assert park_err < 0.02, park_err

            # Grip round-trip: polarity inverted on the way out (command) and on the way back (observation).
            target_grip.emit(0.8)
            pump(0.5)
            assert fake.last_command is not None
            assert abs(fake.last_command[6] - 0.2) < 1e-6, fake.last_command  # positronic 0.8 closed -> chain 0.2
            assert abs(grip.value - 0.8) < 0.02, grip.value
            target_grip.emit(0.0)
            pump(0.5)
            assert abs(fake.last_command[6] - 1.0) < 1e-6, fake.last_command
            assert abs(grip.value) < 0.02, grip.value

        reach_q = np.array([0.0, 1.2, 1.2, 0.0, 0.6, 0.0])
        answer = sync_move(command.JointPosition(reach_q))
        for _ in range(100):
            if answer.done():
                break
            pump(0.1)
        answer.result()
        if fake is not None:
            assert np.allclose(state.value.q, reach_q, atol=MOVE_SETTLE.tolerance_rad), state.value.q

        # Then a Cartesian square through the driver's IK, commanded rather than asked for. The square sits
        # well inside the reach envelope, at the unfolded posture's wrist orientation.

        center = geom.Transform3D(np.array([0.30, 0.05, 0.20]), state.value.ee_pose.rotation)
        print(f'Square center: {center}')
        square = [(0.0, 0.05, 0.0), (0.0, 0.05, 0.05), (0.0, -0.05, 0.05), (0.0, -0.05, 0.0), (0.0, 0.0, 0.0)]
        for offset in square:
            target = geom.Transform3D(center.translation + np.asarray(offset), center.rotation)
            solution = kin.ik(target, state.value.q)
            assert solution is not None, f'IK failed for {target}'
            ik_err = np.linalg.norm(kin.fk(solution).translation - target.translation)
            assert ik_err < 5e-3, ik_err  # FK↔IK consistency
            commands.emit(command.CartesianPosition(target))
            pump(0.7)
            reached = np.linalg.norm(state.value.ee_pose.translation - target.translation)
            print(f'Moved to {target.translation}, error {reached * 1000:.2f} mm')
            if fake is not None:
                assert reached < 5e-3, reached

        print('YAM driver smoke passed')
