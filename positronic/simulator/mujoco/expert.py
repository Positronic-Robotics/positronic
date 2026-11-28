"""The scripted expert of ``lift_cube`` (``positronic.simulator.mujoco.lift_cube``): the helper of a run. It lifts the
cube with the Franka arm of ``MujocoSim``.

At each step the expert picks its sub-goal from the state: up from the table, above the cube, down to the grasp height,
close the grip, or up past ``LIFT_HEIGHT`` with the cube in the grip. So it can take over at any pose and grip.

Each element also carries the press fields of a console. With ``takes_over_at_start``, the expert presses Take over at
its second element and drives to the end of the episode, as in round 0 of DAgger. Else it presses Take over when the
source that drives leaves its path, drives to its next sub-goal, and presses Hand back::

    rollout = Rollout(task, policy, writer, LiftCubeExpert(license='...'))
"""

from enum import Enum
from typing import Any

import numpy as np

from positronic import geom, keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.keys import CONTROL, LICENSE, TELEOPERATOR, TYPE
from positronic.policy.takeover import ControlObservation
from positronic.simulator.expert import (
    DOWN,
    Command,
    Console,
    apart,
    arm_commands,
    check_truth,
    grip,
    grip_pose,
    object_pose,
    touched,
    wrap,
    yaw,
)
from positronic.simulator.mujoco.lift_cube import CUBE_HALF, CUBE_NAME, LIFT_HEIGHT, TABLE_TOP

APPROACH_HEIGHT = 0.08  # m, the height of the grip point above the cube centre before the expert goes down
# m, how far past ``LIFT_HEIGHT`` the expert commands the grip point, so that an arm that sags gets there.
LIFT_MARGIN = 0.06
PERIOD_NS = 20_000_000  # the time between two steps of the expert
# How long the grip closes, from the first observed grip past ``OPEN``, before the expert checks that it holds the cube.
CLOSE_NS = 600_000_000
REACH_M = 0.012  # the largest distance from a sub-goal pose of a grip point that is at it
REACH_RAD = 0.15  # the largest angle from a sub-goal pose of a grip point that is at it
KEEP_M = 0.025  # the largest distance from its path that keeps the arm in ``DOWN`` or ``CLOSE``
KEEP_RAD = 0.3  # the largest angle from its path that keeps the arm in ``DOWN`` or ``CLOSE``
HOLD_M = 0.03  # the largest distance from the grip point of the centre of a cube in the grip
OPEN = 0.1  # the largest observed grip of an open grip
HELD_GRIP = (0.15, 0.9)  # the observed grip of a grip that closes on a cube and stops before its end
STILL_M = 0.003  # m, the largest move of a grip point that stays still
STILL_RAD = 0.03  # rad, the largest turn of a grip point that stays still
STUCK_NS = 500_000_000  # how long a grip point stays still, short of the commanded pose, before the expert turns it
# m, how far past the approach height ``RISE`` goes, so that an arm that sags gets to it. An arm in ``ABOVE`` that sags
# by more than this goes back to ``RISE``.
RISE_MARGIN = 0.03
# The frames of the poses that the expert reads and commands: the frames of the one-arm rig.
FRAMES = {signal: EEFrame(site=GRIPPER_SITE) for signal in (keys.EE_POSE, keys.TARGET_EE_POSE)}


class Phase(Enum):
    """A sub-goal of the expert, in the order of its path."""

    RISE = 'rise'  # up to the approach height, with the grip open
    ABOVE = 'above'  # to the approach pose above the cube
    DOWN = 'down'  # to the grasp pose
    CLOSE = 'close'  # close the grip at the grasp pose
    LIFT = 'lift'  # up past the lift height, and hold


# How far each phase is along the path of the expert. A phase of a lower rank after a phase is a step back.
RANK = {phase: rank for rank, phase in enumerate(Phase)}


def _holds(pose: geom.Transform3D, grip: float, cube: np.ndarray, contact: bool) -> bool:
    """Whether a grip point at ``pose`` with the observed ``grip`` holds ``cube``, which a finger touches if
    ``contact``."""
    return bool(
        contact and HELD_GRIP[0] <= grip <= HELD_GRIP[1] and np.linalg.norm(pose.translation - cube[:3]) <= HOLD_M
    )


class LiftCube:
    """The sub-goal of the arm for the cube, which it picks from the state at each step, and the pose that it commands.

    The grasp points down with x along a face of the cube. Of the four such rotations, it takes the one nearest the
    last that it took. When the arm stops short of the commanded pose for ``STUCK_NS``, the IK of the sim finds no
    solution there, most often for a joint at its limit. Then it takes the next rotation the other way.
    """

    def __init__(self, start: geom.Transform3D, now_ns: int):
        """``start`` is the pose of the grip point."""
        self.phase: Phase | None = None
        self.goal = start  # the pose of the sub-goal of the phase
        self._lift_z = TABLE_TOP + CUBE_HALF + LIFT_HEIGHT + LIFT_MARGIN
        self._command = Command(start, now_ns)
        # The time at which the observed grip first got past ``OPEN`` in ``CLOSE``, or ``None``.
        self._closing_ns: int | None = None
        self._yaw = yaw(start.rotation)  # the yaw of the grasp
        self._still = start, now_ns  # a pose of the grip point, and the time from which the grip point stays near it

    def _grasp(self, cube_quat: np.ndarray) -> geom.Rotation:
        cube_yaw = yaw(geom.Rotation.from_quat(cube_quat))
        self._yaw = min((cube_yaw + k * np.pi / 2 for k in range(4)), key=lambda turn: abs(wrap(turn - self._yaw)))
        return geom.Rotation.from_euler([0.0, 0.0, self._yaw]) * DOWN

    def _pick(self, pose: geom.Transform3D, grip: float, cube: np.ndarray, contact: bool, now_ns: int) -> Phase:
        if _holds(pose, grip, cube, contact):
            return Phase.LIFT
        offset = pose.translation - cube[:3]
        keep = self.phase in (Phase.DOWN, Phase.CLOSE)
        limit_m, limit_rad = (KEEP_M, KEEP_RAD) if keep else (REACH_M, REACH_RAD)
        if self.phase == Phase.CLOSE and grip > OPEN and np.linalg.norm(offset) <= limit_m:
            self._closing_ns = now_ns if self._closing_ns is None else self._closing_ns
            if now_ns - self._closing_ns < CLOSE_NS:
                return Phase.CLOSE
        turn = (pose.rotation.inv * self._grasp(cube[3:])).angle
        if grip <= OPEN and np.linalg.norm(offset[:2]) <= limit_m and turn <= limit_rad:
            if abs(offset[2]) <= REACH_M:
                return Phase.CLOSE
            if 0.0 < offset[2] <= APPROACH_HEIGHT + REACH_M:
                return Phase.DOWN
        if offset[2] < APPROACH_HEIGHT - (RISE_MARGIN if self.phase == Phase.ABOVE else REACH_M):
            return Phase.RISE
        return Phase.ABOVE

    def _goal(self, cube: np.ndarray) -> geom.Transform3D:
        centre, here = cube[:3], self._command.pose
        if self.phase == Phase.RISE:
            height = centre[2] + APPROACH_HEIGHT + RISE_MARGIN
            return geom.Transform3D(np.append(here.translation[:2], height), self._grasp(cube[3:]))
        if self.phase == Phase.LIFT:
            return geom.Transform3D(np.append(here.translation[:2], self._lift_z), here.rotation)
        height = APPROACH_HEIGHT if self.phase == Phase.ABOVE else 0.0
        return geom.Transform3D(centre + [0.0, 0.0, height], self._grasp(cube[3:]))

    def command(
        self, pose: geom.Transform3D, grip: float, cube: np.ndarray, contact: bool, now_ns: int, drives: bool
    ) -> tuple[geom.Transform3D, float]:
        """The pose and the grip that the arm takes at ``now_ns``. ``pose`` and ``grip`` are what the arm is at.
        ``cube`` is the pose of the cube: [x, y, z, qw, qx, qy, qz]. ``contact`` is whether a finger touches it. While
        the expert does not drive, the commanded pose stays at ``pose``, so that a takeover starts there."""
        phase = self._pick(pose, grip, cube, contact, now_ns)
        self.phase = phase
        if phase != Phase.CLOSE:
            self._closing_ns = None
        if not drives:
            self._command.restart(pose, now_ns)
            self._still = pose, now_ns
        if apart(pose, self._still[0], STILL_M, STILL_RAD):
            self._still = pose, now_ns
        short = apart(pose, self._command.pose, REACH_M, REACH_RAD)
        if short and now_ns - self._still[1] >= STUCK_NS and phase not in (Phase.CLOSE, Phase.LIFT):
            self._yaw -= (1.0 if wrap(self._yaw - yaw(pose.rotation)) >= 0 else -1.0) * np.pi / 2
            self._command.pose, self._still = pose, (pose, now_ns)
        self._command.follow(pose)
        self.goal = self._goal(cube)
        return self._command.advance(self.goal, now_ns), 1.0 if phase in (Phase.CLOSE, Phase.LIFT) else 0.0


class LiftCubeExpert(Policy):
    """The scripted expert of ``lift_cube``, a helper, which states the license ``license``. ``takes_over_at_start``
    says that the expert takes over at once and never hands back."""

    WIRE_NAME = 'mujoco_lift_cube_expert'

    def __init__(self, license: str, *, takes_over_at_start: bool = False):
        self._license = license
        self._takes_over_at_start = takes_over_at_start

    def meta(self) -> dict[str, Any]:
        return {TYPE: self.WIRE_NAME, LICENSE: self._license, **ee_frames_entry(FRAMES)}

    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        check_truth(obs, CUBE_NAME)
        lift = LiftCube(grip_pose(obs), runtime.time_ns)
        console = Console(RANK, self._takes_over_at_start)
        while True:
            now, control = runtime.time_ns, ControlObservation.model_validate(obs[CONTROL])
            here = grip_pose(obs)
            cube, contact = object_pose(obs, CUBE_NAME), touched(obs, CUBE_NAME)
            pose, closed = lift.command(here, grip(obs), cube, contact, now, control.source == TELEOPERATOR)
            assert lift.phase is not None, 'a command picks a phase'
            distance = float(np.linalg.norm(lift.goal.translation - here.translation))
            element = console.element(control, lift.phase, distance, arm_commands(pose, closed), now)
            obs = yield Step(element, now + PERIOD_NS)
