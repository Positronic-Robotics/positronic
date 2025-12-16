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


CommandType = Reset | CartesianPosition | JointPosition | JointDelta


def to_wire(command: CommandType) -> dict[str, Any]:
    match command:
        case Reset():
            return {'type': command.TYPE}
        case CartesianPosition(pose):
            return {'type': command.TYPE, 'pose': pose.as_vector(geom.Rotation.Representation.QUAT)}
        case JointPosition(positions):
            return {'type': command.TYPE, 'positions': positions}
        case JointDelta(velocities):
            return {'type': command.TYPE, 'velocities': velocities}


def from_wire(wire: dict[str, Any]) -> CommandType:
    match wire['type']:
        case 'reset':
            return Reset()
        case 'cartesian_pos':
            return CartesianPosition(pose=geom.Transform3D.from_vector(wire['pose']))
        case 'joint_pos':
            return JointPosition(positions=wire['positions'])
        case 'joint_delta':
            return JointDelta(velocities=wire['velocities'])
        case _:
            raise ValueError(f'Unknown command type: {wire["type"]}')
