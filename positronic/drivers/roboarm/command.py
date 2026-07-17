"""Collection of commands that can be sent to the robot."""

from dataclasses import dataclass
from typing import Any

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


@dataclass
class CartesianDelta:
    """Move the end-effector by a world-frame pose delta from its current measured pose.

    A one-shot relative motion: the driver composes ``delta`` onto the pose it measures the moment the
    command is consumed, never re-applying it. Unlike ``JointDelta`` this is end-effector space, not joint
    space.
    """

    TYPE = 'cartesian_delta'
    delta: geom.Transform3D


CommandType = Reset | CartesianPosition | JointPosition | JointDelta | CartesianDelta


def apply_cartesian_delta(current: geom.Transform3D, delta: geom.Transform3D) -> geom.Transform3D:
    """Compose a world-frame ``delta`` onto a measured ``current`` pose for the absolute target a driver drives to.

    Translation adds in the world frame and rotation left-multiplies (``goal_ori = R(Δrot) @ ee_ori``), the
    robosuite OSC convention. This is not ``Transform3D.__mul__``, which composes in the body frame and would
    rotate the translation.
    """
    return geom.Transform3D(current.translation + delta.translation, delta.rotation * current.rotation)


def _combine(acc: CommandType, cmd: CommandType) -> CommandType:
    match (acc, cmd):
        case (CartesianDelta(a), CartesianDelta(b)):
            return CartesianDelta(apply_cartesian_delta(a, b))
        case (JointDelta(a), JointDelta(b)):
            return JointDelta(a + b)
        case (CartesianDelta() | JointDelta(), _) | (_, CartesianDelta() | JointDelta()):
            raise ValueError(f'Cannot reduce {type(acc).__name__} then {type(cmd).__name__} in one tick')
        case _:
            return cmd


def reduce(due: list[tuple[float, CommandType]]) -> CommandType:
    """Collapse the commands due in one control tick into the single command to execute.

    Folds the batch in timestamp order. A run of same-space deltas accumulates (their motion is summed, so a
    missed tick is caught up rather than dropped); a run of absolute commands keeps the last. Mixing an absolute
    with a delta, or two delta spaces, has no faithful single-command form -- a delta binds to the pose measured
    when it is consumed, which an absolute target or a foreign space cannot supply -- and raises.
    """
    result = due[0][1]
    for _, cmd in due[1:]:
        result = _combine(result, cmd)
    return result


def _reduce_last(due: list[tuple[float, Any]]) -> Any:
    """The trailing value wins -- the right collapse for absolute setpoints and gripper targets."""
    return due[-1][1]


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
        case CartesianDelta(delta):
            return {'type': command.TYPE, 'delta': delta.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}


class TrajectoryPlayer:
    """Plays back a timestamped trajectory at the driver's control rate.

    Call ``set()`` when a new trajectory arrives, then ``advance(now)`` each tick to get the single command to
    apply: every waypoint whose timestamp has been reached is collapsed by ``reduce`` into one value (the last
    one by default; the arm channels pass ``command.reduce`` to accumulate due deltas instead of dropping them).
    """

    def __init__(self, reduce=_reduce_last):
        self._trajectory: list[tuple[float, Any]] = []
        self._index: int = 0
        self._reduce = reduce

    def set(self, data):
        if isinstance(data, list):
            self._trajectory = data
        else:
            self._trajectory = [(0.0, data)]
        self._index = 0

    def advance(self, current_time: float):
        """Collapse every waypoint whose timestamp <= current_time into the single value to apply, or None."""
        due = []
        while self._index < len(self._trajectory):
            ts, value = self._trajectory[self._index]
            if ts > current_time:
                break
            self._index += 1
            due.append((ts, value))
        return self._reduce(due) if due else None


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
            return CartesianDelta(
                delta=geom.Transform3D.from_vector(wire['delta'], geom.Rotation.Representation.ROTATION_MATRIX)
            )
        case _:
            raise ValueError(f'Unknown command type: {wire["type"]}')
