"""The parts that the scripted experts of the simulated tasks share: the commanded pose that moves toward a goal, the
path that the source that drives keeps to, the press fields of a console, and the readers of an observation.

An expert is the helper of a run. It reads the pose at ``GRIPPER_SITE`` and the grip of the arm, and the ground truth
of the scene under ``GROUND_TRUTH``. It sends a pose at ``GRIPPER_SITE`` and a grip.
"""

from collections.abc import Mapping
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
from eval_vocabulary.episode import GT_STATE_GRIPPER_CONTACT, GT_STATE_OBJECTS, GT_STATE_POSE, GT_STATE_SIGNAL

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy.base import Commands, Obs
from positronic.policy.keys import GROUND_TRUTH, HELPER_HAND_BACK, HELPER_TAKE_OVER, TELEOPERATOR
from positronic.policy.takeover import ControlObservation

SPEED = 0.15  # m/s, the largest speed of the commanded pose
TURN_SPEED = 1.5  # rad/s, the largest turn speed of the commanded pose
TRACK_M = 0.05  # the largest distance of the commanded pose from the grip point before it goes back to the grip point
TRACK_RAD = 0.3  # the largest angle of the commanded pose from the grip point before it goes back to the grip point
# A source that drives leaves the path of the expert when its grip point gets this far past its nearest distance to the
# sub-goal, or when it gets no ``PROGRESS_M`` nearer for ``STALL_NS``.
LEAVE_M = 0.04
PROGRESS_M = 0.005
STALL_NS = 1_500_000_000
# The rotation of a grip point whose approach axis z points down.
DOWN = geom.Rotation.from_euler([np.pi, 0.0, 0.0])
# The phases of the path of an expert.
PhaseT = TypeVar('PhaseT', bound=Enum)


def yaw(rotation: geom.Rotation) -> float:
    """The angle about z of the x axis of ``rotation``."""
    x_axis = rotation.as_rotation_matrix[:, 0]
    return float(np.arctan2(x_axis[1], x_axis[0]))


def wrap(angle: float) -> float:
    """``angle`` in [-pi, pi]."""
    return float(np.angle(np.exp(1j * angle)))


def apart(first: geom.Transform3D, second: geom.Transform3D, limit_m: float, limit_rad: float) -> bool:
    """Whether ``first`` is more than ``limit_m`` or ``limit_rad`` from ``second``."""
    far = np.linalg.norm(first.translation - second.translation) > limit_m
    return bool(far or (first.rotation.inv * second.rotation).angle > limit_rad)


def toward(pose: geom.Transform3D, goal: geom.Transform3D, step_m: float, step_rad: float) -> geom.Transform3D:
    """``pose`` moved toward ``goal`` by not more than ``step_m`` and ``step_rad``."""
    move = np.linalg.norm(goal.translation - pose.translation) / step_m
    steps = max(move, (pose.rotation.inv * goal.rotation).angle / step_rad)
    return goal if steps <= 1.0 else pose.interpolate(goal, float(1.0 / steps))


class Command:
    """The commanded pose of a grip point. It moves toward a goal at ``SPEED`` and ``TURN_SPEED``, and goes back to the
    grip point when the arm does not follow it."""

    def __init__(self, start: geom.Transform3D, now_ns: int):
        self.restart(start, now_ns)

    def restart(self, pose: geom.Transform3D, now_ns: int) -> None:
        """Command ``pose``, the pose of the grip point at ``now_ns``."""
        self.pose = pose
        self._time_ns = now_ns

    def follow(self, pose: geom.Transform3D) -> None:
        """Go back to ``pose``, the pose of the grip point, if the commanded pose is too far from it."""
        if apart(pose, self.pose, TRACK_M, TRACK_RAD):
            self.pose = pose

    def advance(self, goal: geom.Transform3D, now_ns: int) -> geom.Transform3D:
        seconds, self._time_ns = (now_ns - self._time_ns) / 1e9, now_ns
        if seconds > 0:
            self.pose = toward(self.pose, goal, SPEED * seconds, TURN_SPEED * seconds)
        return self.pose


class Path(Generic[PhaseT]):
    """Whether the source that drives keeps to the path of an expert: it moves the arm on toward each sub-goal, and
    never back to an earlier phase. ``rank`` gives how far each phase is along the path."""

    def __init__(self, rank: Mapping[PhaseT, int]):
        self._rank = rank
        self.restart()

    def restart(self) -> None:
        """Watch the path from the next step."""
        self._phase: PhaseT | None = None
        self._nearest = np.inf
        self._since_ns = 0

    def driver_left(self, phase: PhaseT, distance: float, now_ns: int) -> bool:
        """Whether the source that drives left the path at ``now_ns``, with the arm in ``phase`` at ``distance`` from
        its sub-goal."""
        if self._phase is not None and self._rank[phase] < self._rank[self._phase]:
            return True
        if phase != self._phase or distance < self._nearest - PROGRESS_M:
            self._phase, self._nearest, self._since_ns = phase, distance, now_ns
        return distance > self._nearest + LEAVE_M or now_ns - self._since_ns > STALL_NS


class Console(Generic[PhaseT]):
    """The press fields that an expert sends, as the console of a person sends them. ``rank`` gives how far each phase
    is along the path of the expert. ``takes_over_at_start`` says that the expert takes over at its second element and
    never hands back. Else it takes over when the source that drives leaves its path, and presses Hand back at a phase
    of a higher rank than at the takeover."""

    def __init__(self, rank: Mapping[PhaseT, int], takes_over_at_start: bool):
        self._rank = rank
        self._at_start = takes_over_at_start
        self._path = Path(rank)
        self._take_over: int | None = None
        self._hand_back: int | None = None
        self._taken_rank = 0  # the rank of the phase at the last takeover
        self._first = True

    def element(
        self, control: ControlObservation, phase: PhaseT, distance: float, commands: Commands, now_ns: int
    ) -> dict[str, Any]:
        """The element at ``now_ns``: ``commands``, the commands of the expert, and the press fields. ``control`` is
        the ``CONTROL`` observation, and the arm is in ``phase`` at ``distance`` from its sub-goal."""
        drives = control.source == TELEOPERATOR
        # The presses of the first element of a helper are not new, so the expert presses at its second element.
        if self._first:
            self._first = False
        elif not drives and (self._at_start or self._path.driver_left(phase, distance, now_ns)):
            self._take_over, self._taken_rank = now_ns, self._rank[phase]
        elif drives and not self._at_start and self._rank[phase] > self._taken_rank:
            self._hand_back = now_ns
            self._path.restart()
        return {**commands, HELPER_TAKE_OVER: self._take_over, HELPER_HAND_BACK: self._hand_back}


def grip_pose(obs: Obs) -> geom.Transform3D:
    """The pose of the grip point at ``GRIPPER_SITE`` in ``obs``, in the rig's world frame."""
    ee_pose = np.asarray(obs[keys.EE_POSE])
    return geom.Transform3D.from_vector(ee_pose, geom.Rotation.Representation.QUAT)


def grip(obs: Obs) -> float:
    """The observed grip in ``obs``: 0 open, 1 closed."""
    return float(obs[keys.GRIP])


def arm_commands(pose: geom.Transform3D, grip: float) -> dict[str, Any]:
    """The commands that move the grip point to ``pose`` and set the grip."""
    return {keys.ROBOT_COMMAND: CartesianPosition(pose), keys.TARGET_GRIP: grip}


def object_key(name: str, field: str) -> str:
    """The key of ``field`` of the object ``name`` in the ``GROUND_TRUTH`` observation."""
    return f'{GT_STATE_SIGNAL}.{GT_STATE_OBJECTS}.{name}.{field}'


def check_truth(obs: Obs, name: str) -> None:
    """Raise ``ValueError`` when ``obs`` gives no pose of the object ``name`` under ``GROUND_TRUTH``."""
    if object_key(name, GT_STATE_POSE) not in obs.get(GROUND_TRUTH, {}):
        raise ValueError(
            f'The scripted expert gets no {GROUND_TRUTH}. Run it as the helper, and wire the {GT_STATE_SIGNAL} '
            'signal of the sim as a privileged signal of the harness (wire.wire_embodiment(privileged=...))'
        )


def object_pose(obs: Obs, name: str) -> np.ndarray:
    """The pose of the object ``name`` in the ``GROUND_TRUTH`` of ``obs``: [x, y, z, qw, qx, qy, qz]."""
    return np.asarray(obs[GROUND_TRUTH][object_key(name, GT_STATE_POSE)], dtype=np.float64)


def touched(obs: Obs, name: str) -> bool:
    """Whether a finger touches the object ``name`` in ``obs``."""
    return bool(obs[GROUND_TRUTH][object_key(name, GT_STATE_GRIPPER_CONTACT)])
