"""Collection of commands that can be sent to the robot.

A command may pin the control mode it executes under; ``mode=None`` pins nothing, and the arm runs
its native law. What a pinned mode does is the driver's: a simulator runs its own law and ignores it,
a robot driver that cannot execute it raises.

TODO: have an embodiment state the modes it supports, so a policy selects one it works with instead of
pinning a mode the driver may not run.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from positronic import geom


@dataclass
class PositionControl:
    """Arm control mode: a position servo tracking shaped references.

    ``stiffness`` biases the servo's joint-space stiffness on an arm that exposes it; ``None`` names none
    and the arm keeps its own, as damping always does.
    """

    TYPE = 'position_control'
    stiffness: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.stiffness is None:
            return
        self.stiffness = tuple(self.stiffness)
        if not self.stiffness:
            raise ValueError('stiffness must name at least one joint; naming none is None, not an empty vector')
        if not all(math.isfinite(v) and v > 0 for v in self.stiffness):
            raise ValueError('stiffness must be finite and strictly positive')


@dataclass
class Impedance:
    """Arm control mode: the hybrid joint/Cartesian impedance law with instantly-stepped references.

    ``tau = (J^T Kx J + Kq)(q_d - q) - (J^T Kxd J + Kqd) dq + coriolis``.
    """

    TYPE = 'impedance'
    kq: tuple[float, ...]
    kqd: tuple[float, ...]
    kx: tuple[float, ...]
    kxd: tuple[float, ...]

    @staticmethod
    def _validate_gains(name: str, k: tuple[float, ...], kd: tuple[float, ...]) -> None:
        """Check one half's stiffness against its damping."""
        if len(k) != len(kd):
            raise ValueError(f'{name} stiffness and damping must have the same length')
        if not k:
            raise ValueError(f'{name} must name at least one axis; a half is disabled by zeroing it, not emptying it')
        disabled = all(v == 0 for v in k) and all(v == 0 for v in kd)
        active = all(math.isfinite(v) and v > 0 for v in k) and all(math.isfinite(v) and v > 0 for v in kd)
        if not (disabled or active):
            raise ValueError(
                f'{name} stiffness and damping must be either all zero (half disabled) or strictly positive'
            )

    def __post_init__(self):
        self.kq, self.kqd = tuple(self.kq), tuple(self.kqd)
        self.kx, self.kxd = tuple(self.kx), tuple(self.kxd)
        if len(self.kx) != 6:
            raise ValueError('Cartesian (kx/kxd) gains must have length 6')
        self._validate_gains('joint (kq/kqd)', self.kq, self.kqd)
        self._validate_gains('Cartesian (kx/kxd)', self.kx, self.kxd)
        if all(v == 0 for v in self.kq) and all(v == 0 for v in self.kx):
            raise ValueError('at least one of the joint or Cartesian halves must be active')


ControlModeType = PositionControl | Impedance


@dataclass
class CartesianPosition:
    """Move the robot end-effector to the given pose."""

    TYPE = 'cartesian_pos'
    pose: geom.Transform3D
    mode: ControlModeType | None = None


@dataclass
class JointPosition:
    """Move the robot joints to the given positions."""

    TYPE = 'joint_pos'
    positions: np.ndarray
    mode: ControlModeType | None = None


def sampled_joints(nominal: Sequence[float] | np.ndarray, spread: Sequence[float] | np.ndarray) -> JointPosition:
    """A joint target drawn uniformly from ``nominal`` ± ``spread``, per joint.

    An empty ``spread`` names no variation, which is how a rig configured without any asks for its nominal.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    spread = np.zeros_like(nominal) if len(spread) == 0 else np.asarray(spread, dtype=np.float64)
    return JointPosition(nominal + np.random.uniform(-spread, spread))


@dataclass
class JointDelta:
    """Move the robot joints with the given velocities."""

    TYPE = 'joint_delta'
    velocities: np.ndarray
    mode: ControlModeType | None = None


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
    mode: ControlModeType | None = None

    def apply(self, current: geom.Transform3D) -> geom.Transform3D:
        """The absolute target to drive to, given the pose measured at the receiver's ``default`` frame.

        ``frame`` carries the delta into the frame it was expressed in and the result back out, so a policy
        speaking a different end-effector frame moves the arm as it intended.
        """
        return _compose_delta(current * self.frame, self.delta) * self.frame.inv


CommandType = CartesianPosition | JointPosition | JointDelta | CartesianDelta


def to_wire(command: CommandType | ControlModeType) -> dict[str, Any]:
    wire: dict[str, Any]
    rep = geom.Rotation.Representation.ROTATION_MATRIX
    match command:
        case PositionControl(stiffness=stiffness):
            return {'type': command.TYPE} if stiffness is None else {'type': command.TYPE, 'stiffness': list(stiffness)}
        case Impedance(kq=kq, kqd=kqd, kx=kx, kxd=kxd):
            return {'type': command.TYPE, 'kq': list(kq), 'kqd': list(kqd), 'kx': list(kx), 'kxd': list(kxd)}
        case CartesianPosition(pose):
            wire = {'type': command.TYPE, 'pose': pose.as_vector(rep)}
        case JointPosition(positions):
            wire = {'type': command.TYPE, 'positions': positions}
        case JointDelta(velocities):
            wire = {'type': command.TYPE, 'velocities': velocities}
        case CartesianDelta(delta, frame):
            wire = {'type': command.TYPE, 'delta': delta.as_vector(rep), 'frame': frame.as_vector(rep)}
    if command.mode is not None:
        wire['mode'] = to_wire(command.mode)
    return wire


def from_wire(wire: dict[str, Any]) -> CommandType | ControlModeType:
    mode = from_wire(wire['mode']) if 'mode' in wire else None
    assert mode is None or isinstance(mode, PositionControl | Impedance)
    rep = geom.Rotation.Representation.ROTATION_MATRIX
    match wire['type']:
        case PositionControl.TYPE:
            return PositionControl(stiffness=wire.get('stiffness'))
        case Impedance.TYPE:
            return Impedance(kq=wire['kq'], kqd=wire['kqd'], kx=wire['kx'], kxd=wire['kxd'])
        case CartesianPosition.TYPE:
            return CartesianPosition(pose=geom.Transform3D.from_vector(wire['pose'], rep), mode=mode)
        case JointPosition.TYPE:
            return JointPosition(positions=wire['positions'], mode=mode)
        case JointDelta.TYPE:
            return JointDelta(velocities=wire['velocities'], mode=mode)
        case CartesianDelta.TYPE:
            return CartesianDelta(
                delta=geom.Transform3D.from_vector(wire['delta'], rep),
                frame=geom.Transform3D.from_vector(wire['frame'], rep),
                mode=mode,
            )
        case _:
            raise ValueError(f'Unknown command type: {wire["type"]}')


def require_native_mode(cmd: CommandType, embodiment: str) -> None:
    """Raises on a command that pins a control mode: ``embodiment`` runs only its native law."""
    if cmd.mode is not None:
        raise NotImplementedError(f'{embodiment} cannot execute control mode {cmd.mode}')
