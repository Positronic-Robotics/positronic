"""Collection of commands that can be sent to the robot."""

from collections.abc import Sequence
from dataclasses import dataclass, field
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


def _combine(acc: CommandType, cmd: CommandType) -> CommandType:
    match (acc, cmd):
        case (CartesianDelta(a, frame_a), CartesianDelta(b, frame_b)):
            if not np.allclose(frame_a.as_matrix, frame_b.as_matrix):
                raise ValueError('Cannot accumulate cartesian deltas expressed in different frames')
            return CartesianDelta(_compose_delta(a, b), frame_a)
        case (JointDelta(a), JointDelta(b)):
            return JointDelta(a + b)
        case (CartesianDelta() | JointDelta(), _) | (_, CartesianDelta() | JointDelta()):
            raise ValueError(f'Cannot reduce {type(acc).__name__} then {type(cmd).__name__} in one tick')
        case _:
            return cmd


def reduce(due: Sequence[CommandType]) -> CommandType:
    """Collapse the commands due in one round into the single command to execute.

    Folds the batch in timestamp order. A run of same-space deltas accumulates (their motion is summed, so a
    round spanning several waypoints catches up rather than dropping them); a run of absolute commands keeps
    the last. Mixing an absolute with a delta, or two delta spaces, has no faithful single-command form — a
    delta binds to the pose measured when it is consumed, which an absolute target or a foreign space cannot
    supply — and raises.
    """
    result = due[0]
    for cmd in due[1:]:
        result = _combine(result, cmd)
    return result
