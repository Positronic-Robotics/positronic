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
class Recover:
    """Recover the robot from an error state."""

    TYPE = 'recover'


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


CommandType = Reset | Recover | CartesianPosition | JointPosition | JointDelta


def to_wire(command: CommandType) -> dict[str, Any]:
    match command:
        case Reset() | Recover():
            return {'type': command.TYPE}
        case CartesianPosition(pose):
            return {'type': command.TYPE, 'pose': pose.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}
        case JointPosition(positions):
            return {'type': command.TYPE, 'positions': positions}
        case JointDelta(velocities):
            return {'type': command.TYPE, 'velocities': velocities}


class TrajectoryPlayer:
    """Plays back a timestamped trajectory at the driver's control rate.

    Call ``set()`` when a new trajectory arrives, then ``advance(now)`` each tick
    to yield all commands whose timestamp has been reached.
    """

    def __init__(self):
        self._trajectory: list[tuple[float, Any]] = []
        self._index: int = 0

    def set(self, data):
        if isinstance(data, list):
            self._trajectory = data
        else:
            self._trajectory = [(0.0, data)]
        self._index = 0

    def advance(self, current_time: float):
        """Yield all commands whose timestamp <= current_time."""
        while self._index < len(self._trajectory):
            ts, cmd = self._trajectory[self._index]
            if ts > current_time:
                break
            self._index += 1
            yield cmd


def from_wire(wire: dict[str, Any]) -> CommandType:
    match wire['type']:
        case 'reset':
            return Reset()
        case 'recover':
            return Recover()
        case 'cartesian_pos':
            return CartesianPosition(
                pose=geom.Transform3D.from_vector(wire['pose'], geom.Rotation.Representation.ROTATION_MATRIX)
            )
        case 'joint_pos':
            return JointPosition(positions=wire['positions'])
        case 'joint_delta':
            return JointDelta(velocities=wire['velocities'])
        case _:
            raise ValueError(f'Unknown command type: {wire["type"]}')
