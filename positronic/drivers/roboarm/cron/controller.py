"""A CRON robot controller, seen through its vendor client ``rctlstd``.

The client is a Linux x86-64 wheel that the controller serves at ``http://<controller>/pypi/``.
"""

import enum
import logging
import time
from typing import NamedTuple

import numpy as np
from rctlstd import rctlext as rctl

from positronic import geom

logger = logging.getLogger(__name__)

JOINTS = 6

_CONNECT_TIMEOUT_S = 5.0
_POWER_TIMEOUT_S = 15.0
_HOLD_TIMEOUT_S = 0.5
_POLL_S = 0.05
# After RUN the controller moves the joints once, to take up the brake slack
_STARTUP_S = 1.0
# Velocity commands that stop the arm, sent more than once so that one lost command does not leave it moving
_STOP_REPEATS = 10
_STOP_ACCEL = 5.0  # rad/s^2


class PowerState(enum.IntEnum):
    """The state of the controller. The numbers are the controller's."""

    IDLE = 0
    OFF = 1
    STANDBY = 2
    RUN = 3
    SERVICE = 4
    FAILURE = 5
    FORCE_EXIT = 6


class MotionMode(enum.IntEnum):
    """The motion mode of the controller. The numbers are the controller's."""

    NONE = 0
    HOLD = 1
    ZG = 2
    JOG = 3
    JOG_JS = 4
    MOVE = 5
    MOVE_AUX = 6
    JOINT_VEL = 7
    JOINT_TRQ = 8
    CART_VEL = 9
    CART_TRQ = 10


class Reading(NamedTuple):
    q: np.ndarray
    dq: np.ndarray
    tcp_pose: geom.Transform3D
    power: PowerState
    in_error: bool


def to_pose(vector: np.ndarray) -> geom.Transform3D:
    """The pose of ``[x, y, z, rx, ry, rz]``, where ``rx, ry, rz`` are the Euler angles of the vendor library."""
    return geom.Transform3D(
        translation=np.asarray(vector[:3], dtype=np.float64),
        rotation=geom.Rotation.from_quat_xyzw(np.asarray(rctl.rpy2quat(list(vector[3:])), dtype=np.float64)),
    )


def to_vector(pose: geom.Transform3D) -> np.ndarray:
    """The ``[x, y, z, rx, ry, rz]`` of ``pose``."""
    euler = np.asarray(rctl.quat2rpy(pose.rotation.as_quat_xyzw), dtype=np.float64)
    return np.concatenate([pose.translation, euler])


def level(up: np.ndarray) -> geom.Transform3D:
    """The smallest turn of the base frame that puts ``up``, given in the base frame, on the z axis."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sin, cos = float(np.linalg.norm(axis)), float(up[2])
    if sin < 1e-9:
        # Upright: no turn. Upside down: any horizontal axis turns z over; x keeps the result definite
        rotvec = np.zeros(3) if cos > 0 else np.array([np.pi, 0.0, 0.0])
    else:
        rotvec = axis / sin * np.arctan2(sin, cos)
    return geom.Transform3D(rotation=geom.Rotation.from_rotvec(rotvec))


class CronController:
    """The connection to one controller. It owns the arm from ``__enter__`` to ``__exit__``.

    ``__enter__`` takes control and powers the arm to RUN, so the local UI must have released control first.
    ``__exit__`` stops the arm, powers it off if ``power_off_on_exit`` is set, and releases control.

    Poses are of the tool point, ``tool_length`` metres along the flange z axis. The tool of the controller must be
    zero, so that the controller computes the flange. Poses are in the base frame turned by ``mount`` so that z points
    up: the gravity setting of the controller gives the up direction of a base mounted at an angle.
    """

    def __init__(self, ip: str, power_off_on_exit: bool = True, tool_length: float = 0.0, project: str | None = None):
        self.ip = ip
        self.power_off_on_exit = power_off_on_exit
        # A project of the controller that runs beside the arm, for example a bridge to the gripper on the wrist. The
        # controller starts it only for a client that has control, and keeps it running after the client leaves. A
        # second start of a running project does nothing
        self.project = project
        self.tool = geom.Transform3D(translation=np.array([0.0, 0.0, tool_length]))
        self.mount = geom.Transform3D.identity
        self._has_control = False
        self._waypoint_id = 0

    def __enter__(self) -> 'CronController':
        rctl.connect(self.ip)
        try:
            self._await(lambda: rctl.get_cycle_count() > 0, _CONNECT_TIMEOUT_S, 'the first data of the controller')
            rctl.sync_robot_model()
            self._await(rctl.is_model_valid, _CONNECT_TIMEOUT_S, 'the model of the robot')
            controller_tool = np.asarray(rctl.get_tool(), dtype=np.float64)
            if np.any(controller_tool != 0):
                raise RuntimeError(f'the controller has the tool {controller_tool}: set it to zero, use tool_length')
            # The controller keeps gravity as the up direction in the base frame, as UR does: [0, 0, 9.81] upright
            up = np.asarray(rctl.get_gravity(), dtype=np.float64)
            if not np.linalg.norm(up) > 0:
                raise RuntimeError(f'the controller has no gravity set ({up}): the up direction of the base is unknown')
            self.mount = level(up)
            rctl.take_control()
            self._has_control = True
            if self.project is not None:
                rctl.exr_run(self.project)
            self._clear_errors()
            # Straight to RUN, as the RCUI does. On an arm whose multi-turn encoder asks for a preset (Hand B, motor 4,
            # 2026-11-06), RUN from STANDBY failed with "Robot error" 4 times in 4, and RUN from OFF passed 3 times in 3
            self._power_to(PowerState.RUN, rctl.cmd_power_run)
            # blocking-sleep-ok: connect-time handshake, before any control loop exists to yield from
            time.sleep(_STARTUP_S)
        # rules-allow: swallowed-error — re-raised after the arm is left safe
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_) -> None:
        # Each step runs on its own: one that fails must not keep the others from running
        for step in (self.stop, self.stop_move, rctl.cmd_hold):
            self._attempt(step)
        if self._has_control and self.power_off_on_exit:
            # A power off that reaches the controller before HOLD can end a velocity stream in a TIMEOUT error
            self._attempt(lambda: self._await(lambda: self.motion is MotionMode.HOLD, _HOLD_TIMEOUT_S, 'HOLD'))
            self._attempt(rctl.cmd_power_off)
            self._attempt(lambda: self._await(lambda: self.power is PowerState.OFF, _POWER_TIMEOUT_S, 'power off'))
        if self._has_control:
            self._attempt(rctl.release_control)
        self._attempt(rctl.disconnect)

    @property
    def power(self) -> PowerState:
        return PowerState(rctl.get_state())

    @property
    def motion(self) -> MotionMode:
        return MotionMode(rctl.get_motion_mode())

    @property
    def errors(self) -> str:
        return f'{rctl.get_errors_hw()} {rctl.get_errors_mm()}'

    def resume(self) -> None:
        """Ask the controller to clear a latched motion error."""
        rctl.cmd_resume()

    def read(self) -> Reading:
        return Reading(
            q=np.asarray(rctl.get_joint_angles(), dtype=np.float64),
            dq=np.asarray(rctl.get_joint_vels(), dtype=np.float64),
            tcp_pose=self.mount * to_pose(np.asarray(rctl.get_cart_pose(), dtype=np.float64)) * self.tool,
            power=self.power,
            in_error=bool(rctl.is_error()),
        )

    def command_velocity(self, velocities: np.ndarray) -> None:
        rctl.cmd_joint_direct_vel(velocities)

    def stop(self) -> None:
        for _ in range(_STOP_REPEATS):
            rctl.cmd_joint_direct_vel(np.zeros(JOINTS))

    def move_to(self, q: np.ndarray, max_speed: float, max_accel: float) -> None:
        """Start a move of the planner of the controller to the joints ``q``."""
        self._waypoint_id += 1
        rctl.cmd_add_waypoint_joint(q, [max_speed, max_accel], 0.0, self._waypoint_id)
        rctl.cmd_move_run()

    def stop_move(self) -> None:
        rctl.cmd_move_pause(_STOP_ACCEL)
        rctl.cmd_drop_waypoints(True)

    def forward(self, q: np.ndarray) -> geom.Transform3D:
        solved = rctl.compute_fk(q)
        if solved['err']:
            raise ValueError(f'no pose for joints {np.round(q, 3)}: {solved["err_code"]}')
        return self.mount * to_pose(np.asarray(solved['pose'], dtype=np.float64)) * self.tool

    def inverse(self, pose: geom.Transform3D, seed: np.ndarray) -> np.ndarray:
        """The joints that put the tool at ``pose``, of the controller's solutions the nearest to ``seed``."""
        solved = rctl.compute_ik(to_vector(self.mount.inv * pose * self.tool.inv), seed)
        candidates = np.asarray(solved['solutions'], dtype=np.float64).reshape(-1, JOINTS)
        # The controller can report no error and still give NaN solutions, and argmin would pick a NaN one
        candidates = candidates[np.all(np.isfinite(candidates), axis=1)]
        if solved['err'] or not len(candidates):
            raise ValueError(f'{pose} is out of reach: {solved["err_code"]}')
        return candidates[np.argmin(np.max(np.abs(candidates - seed), axis=1))]

    def _clear_errors(self) -> None:
        """Clear what an earlier session left: a latched motion error blocks every power request."""
        if rctl.is_error():
            logger.warning(f'{self.ip}: clearing errors: {self.errors}')
            rctl.cmd_resume()
            if self.power is PowerState.FAILURE:
                # A safety stop (RB SFTY) stays latched through a resume. An OFF request clears it once the stop is
                # released
                rctl.cmd_power_off()
        self._await(lambda: not rctl.is_error(), _CONNECT_TIMEOUT_S, 'clear error state (is the stop button released?)')

    def _power_to(self, wanted: PowerState, request) -> None:
        if self.power is not wanted:
            request()
        self._await(lambda: self._is_at(wanted), _POWER_TIMEOUT_S, f'power state {wanted.name}')

    def _is_at(self, wanted: PowerState) -> bool:
        if self.power is PowerState.FAILURE or rctl.is_error():
            raise RuntimeError(f'the controller {self.ip} failed: {self.errors}')
        return self.power is wanted

    @staticmethod
    def _await(done, timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        while not done():
            if time.monotonic() > deadline:
                raise TimeoutError(f'no {what} after {timeout_s} s')
            # blocking-sleep-ok: connect-time handshake, before any control loop exists to yield from
            time.sleep(_POLL_S)

    @staticmethod
    def _attempt(step) -> None:
        try:
            step()
        # rules-allow: swallowed-error — shutdown: every later step must still run, and the first failure is the cause
        except Exception:
            pass
