"""Collection of commands that can be sent to the robot."""

from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

import numpy as np

from positronic import geom


@dataclass
class Reset:
    """Reset the robot to the home position."""

    TYPE = 'reset'

    pass


@dataclass
class CartesianPosition:
    """Move the robot end-effector to the given pose."""

    TYPE = 'cartesian_pos'
    pose: geom.Transform3D


@dataclass
class JointPosition:
    """Move the robot joints to the given positions."""

    TYPE = 'joint_pos'
    positions: np.ndarray


@dataclass
class JointDelta:
    """Move the robot joints with the given velocities."""

    TYPE = 'joint_delta'
    velocities: np.ndarray


def _compose_delta(base: geom.Transform3D, delta: geom.Transform3D) -> geom.Transform3D:
    """Compose a world-frame ``delta`` onto ``base``.

    Translation adds in the world frame and rotation left-multiplies (``goal_ori = R(Δrot) @ ee_ori``), the
    robosuite OSC convention. This is not ``Transform3D.__mul__``, which composes in the body frame and would
    rotate the translation.
    """
    return geom.Transform3D(base.translation + delta.translation, delta.rotation * base.rotation)


@dataclass
class CartesianDelta:
    """Move the end-effector by a world-frame pose delta from its current measured pose.

    A one-shot relative motion: the driver composes ``delta`` onto the pose it measures the moment the
    command is consumed, never re-applying it. Unlike ``JointDelta`` this is end-effector space, not joint
    space.

    A delta has no anchor pose of its own, so it carries the frame it is expressed in instead: ``frame``
    places that frame relative to the receiver's ``default``, and ``apply`` measures there before composing.
    """

    TYPE = 'cartesian_delta'
    delta: geom.Transform3D
    # Per-command, not a shared class-level default: ``Transform3D`` attributes are writable, so one instance
    # across every frameless delta means adjusting one silently relabels the rest.
    frame: geom.Transform3D = field(default_factory=lambda: geom.Transform3D.identity)

    def apply(self, current: geom.Transform3D) -> geom.Transform3D:
        """The absolute target to drive to, given the pose measured at the receiver's ``default`` frame.

        ``frame`` carries the delta into the frame it was expressed in and the result back out, so a policy
        speaking a different end-effector frame moves the arm as it intended.
        """
        return _compose_delta(current * self.frame, self.delta) * self.frame.inv


CommandType = Reset | CartesianPosition | JointPosition | JointDelta | CartesianDelta

_T = TypeVar('_T')

# A schedule the harness plays: waypoints stamped with absolute clock ns, ascending. Command channels
# themselves carry one value -- the command due at the moment it is emitted.
Trajectory: TypeAlias = list[tuple[int, _T]]


def to_wire(command: CommandType) -> dict[str, Any]:
    match command:
        case Reset():
            return {'type': command.TYPE}
        case CartesianPosition(pose):
            return {'type': command.TYPE, 'pose': pose.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}
        case JointPosition(positions):
            return {'type': command.TYPE, 'positions': positions}
        case JointDelta(velocities):
            return {'type': command.TYPE, 'velocities': velocities}
        case CartesianDelta(delta, frame):
            return {
                'type': command.TYPE,
                'delta': delta.as_vector(geom.Rotation.Representation.ROTATION_MATRIX),
                'frame': frame.as_vector(geom.Rotation.Representation.ROTATION_MATRIX),
            }


class TrajectoryPlayer:
    """Plays one channel's schedule: ``set()`` a trajectory, then ``advance(now)`` each round for the value
    to emit."""

    def __init__(self):
        self._trajectory: Trajectory[Any] = []
        self._index: int = 0

    def set(self, trajectory: Trajectory[Any]):
        self._trajectory = trajectory
        self._index = 0

    def next_due(self) -> int | None:
        """Timestamp of the earliest waypoint not yet played, or ``None`` once the schedule is exhausted."""
        return self._trajectory[self._index][0] if self._index < len(self._trajectory) else None

    def advance(self, current_time: int):
        """The single value due at ``current_time``, or ``None`` when no waypoint has come due since the last
        call. Several waypoints due at once collapse to the last: an absolute setpoint supersedes the ones it
        overtook, and only a late round makes it happen.
        """
        value = None
        while self._index < len(self._trajectory) and self._trajectory[self._index][0] <= current_time:
            value = self._trajectory[self._index][1]
            self._index += 1
        return value


def from_wire(wire: dict[str, Any]) -> CommandType:
    match wire['type']:
        case 'reset':
            return Reset()
        case 'cartesian_pos':
            return CartesianPosition(
                pose=geom.Transform3D.from_vector(wire['pose'], geom.Rotation.Representation.ROTATION_MATRIX)
            )
        case 'joint_pos':
            return JointPosition(positions=wire['positions'])
        case 'joint_delta':
            return JointDelta(velocities=wire['velocities'])
        case 'cartesian_delta':
            rep = geom.Rotation.Representation.ROTATION_MATRIX
            return CartesianDelta(
                delta=geom.Transform3D.from_vector(wire['delta'], rep),
                frame=geom.Transform3D.from_vector(wire['frame'], rep),
            )
        case _:
            raise ValueError(f'Unknown command type: {wire["type"]}')
