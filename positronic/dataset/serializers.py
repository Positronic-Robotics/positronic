from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm.command import (
    CartesianDelta,
    CartesianPosition,
    CommandType,
    ControlModeType,
    Impedance,
    JointDelta,
    JointPosition,
    PositionControl,
)

# Serializer contract for values:
# - Used by `DsWriterAgent.add_signal(name, serializer=None)` (recording) and the
#   Harness observation channels (policy input). In both cases `serializer=None`
#   passes the value through unchanged; a callable is invoked as serializer(value)
#   and can return:
#     * a transformed value -> recorded/keyed under the same name
#     * a dict mapping suffix -> value -> expands into multiple entries named name+suffix
#         - use "" (empty string) to keep the base name as-is
#         - any dict entry with value None is skipped
#     * None -> the sample is dropped
Serializer = Callable[[Any], Any | dict[str, Any]]


class StatefulSerializer:
    """Base for serializers registered with ``DsWriterAgent``.

    ``reset`` is called automatically at the start of each episode.
    The default implementation is a no-op, suitable for pure serializers.
    Subclasses that maintain per-episode state should override ``reset``.
    """

    def reset(self) -> None:
        pass

    def __call__(self, value: Any) -> Any | dict[str, Any]:
        raise NotImplementedError


class _PureSerializer(StatefulSerializer):
    """Wraps a plain callable so every serializer has a uniform interface."""

    def __init__(self, fn: Callable[[Any], Any | dict[str, Any]]):
        self._fn = fn

    def __call__(self, value: Any) -> Any | dict[str, Any]:
        return self._fn(value)


class Serializers:
    """Namespace of built-in, type-keyed serializers.

    Shared by the dataset writer (``agent.add_signal("ee_pose", Serializers.transform_3d)``)
    and the Harness observation assembly. Each method owns a domain type's split into the
    canonical ``name + suffix`` entries.
    """

    @staticmethod
    def transform_3d(x: geom.Transform3D) -> np.ndarray:
        """Serialize a Transform3D into a 7D vector [tx, ty, tz, qw, qx, qy, qz]."""
        return x.as_vector(geom.Rotation.Representation.QUAT)

    class ContinuousTransform3D(StatefulSerializer):
        """Stateful serializer that canonicalises quaternion signs for temporal continuity.

        Each quaternion is flipped to the sign closest to the previous frame,
        avoiding arbitrary sign jumps from the double-cover ambiguity.
        """

        def __init__(self):
            self._prev: geom.Rotation | None = None

        def reset(self):
            self._prev = None

        def __call__(self, x: geom.Transform3D) -> np.ndarray:
            rotation = x.rotation
            if self._prev is not None:
                rotation = geom.quat_closest(rotation, self._prev)
            self._prev = rotation
            return geom.Transform3D(x.translation, rotation).as_vector(geom.Rotation.Representation.QUAT)

    @staticmethod
    def robot_state(state: State) -> dict[str, np.ndarray | RobotStatus]:
        return {
            keys.STATUS_SUFFIX: state.status,
            '.q': state.q,
            '.dq': state.dq,
            '.ee_pose': Serializers.transform_3d(state.ee_pose),
        }

    @staticmethod
    def _mode_entries(mode: ControlModeType) -> dict[str, np.ndarray | int]:
        match mode:
            case PositionControl(stiffness=None):
                return {'.mode.position_control': 1}
            case PositionControl(stiffness=stiffness):
                return {'.mode.position_control.stiffness': np.asarray(stiffness)}
            case Impedance(kq=kq, kqd=kqd, kx=kx, kxd=kxd):
                return {
                    '.mode.impedance.kq': np.asarray(kq),
                    '.mode.impedance.kqd': np.asarray(kqd),
                    '.mode.impedance.kx': np.asarray(kx),
                    '.mode.impedance.kxd': np.asarray(kxd),
                }

    # TODO: a command writes only the entries its own form has, so a signal here is sparse — `.mode.*` has
    # samples at some commands and none at others. Which command a sample belongs to is then a timestamp
    # alignment against a signal every command writes, and a reader taking the last value as still standing
    # ascribes a pinned mode to commands that pinned none. A serializer cannot say which of its entries are
    # sparse, and nothing downstream asks.
    @staticmethod
    def robot_command(command: CommandType) -> dict[str, np.ndarray | int]:
        entries: dict[str, np.ndarray | int]
        match command:
            case CartesianPosition(pose):
                entries = {'.pose': Serializers.transform_3d(pose)}
            case CartesianDelta(delta, frame):
                entries = {
                    '.pose_delta': Serializers.transform_3d(delta),
                    '.pose_delta_frame': Serializers.transform_3d(frame),
                }
            case JointPosition(positions):
                entries = {'.joints': positions}
            case JointDelta(delta):
                entries = {'.joint_deltas': delta}
        if command.mode is not None:
            entries.update(Serializers._mode_entries(command.mode))
        return entries

    @staticmethod
    def camera_images(data: pimm.shared_memory.NumpySMAdapter) -> np.ndarray:
        """Extract array from NumpySMAdapter for storage."""
        return data.array


def expand_suffixed(name: str, value: Any) -> Iterator[tuple[str, Any]]:
    """Unfold a value into ``(full_name, value)`` pairs: a dict expands into ``name + suffix``
    entries (``""`` keeps the base name), anything else yields ``(name, value)``."""
    if isinstance(value, dict):
        for suffix, v in value.items():
            yield name + suffix, v
    else:
        yield name, value
