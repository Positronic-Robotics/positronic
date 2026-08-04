from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm.command import (
    Applied,
    CartesianDelta,
    CartesianPosition,
    CommandType,
    JointDelta,
    JointPosition,
    Reset,
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
#     * a list[Timestamped] -> a self-timestamped stream (recording only): each item is
#       recorded at its own ``ts_ns`` and an empty list records nothing. The per-item
#       ``value`` follows the rules above.
Serializer = Callable[[Any], Any | dict[str, Any]]


# The timeline a played waypoint carries beside its own: the instant it was scheduled for.
SCHEDULED_TIMELINE = 'scheduled'


@dataclass
class Timestamped:
    """A sample paired with its own absolute timestamp (ns), and any further timelines it sits on."""

    ts: int
    value: Any
    extra_ts: dict[str, int] | None = None


class StatefulSerializer:
    """Base for serializers registered with ``DsWriterAgent``.

    ``reset`` is called automatically at the start of each episode.
    The default implementation is a no-op, suitable for pure serializers.
    Subclasses that maintain per-episode state should override ``reset``.
    """

    def reset(self) -> None:
        pass

    def __call__(self, value: Any) -> Any | dict[str, Any] | list['Timestamped']:
        raise NotImplementedError


class _PureSerializer(StatefulSerializer):
    """Wraps a plain callable so every serializer has a uniform interface."""

    def __init__(self, fn: Callable[[Any], Any | dict[str, Any]]):
        self._fn = fn

    def __call__(self, value: Any) -> Any | dict[str, Any]:
        return self._fn(value)


class TrajectorySerializer:
    """Records applied waypoints, each at the instant it was applied, carrying the instant it was
    scheduled for on the ``scheduled`` timeline.

    ``inner`` serializes the waypoint values; ``None`` records them unchanged.
    """

    def __init__(self, inner: Serializer | None):
        self._inner = inner

    def __call__(self, waypoints: list[Applied]) -> list['Timestamped']:
        return [
            Timestamped(
                w.ts, w.value if self._inner is None else self._inner(w.value), {SCHEDULED_TIMELINE: w.scheduled_ts}
            )
            for w in waypoints
        ]


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
    def robot_state(state: State) -> dict[str, np.ndarray] | None:
        # ERROR, like RESETTING, is a not-ready state: drop the frame rather than record or infer on a
        # mid-error pose.
        if state.status in (RobotStatus.RESETTING, RobotStatus.ERROR):
            return None
        return {'.q': state.q, '.dq': state.dq, '.ee_pose': Serializers.transform_3d(state.ee_pose)}

    @staticmethod
    def robot_command(command: CommandType) -> dict[str, np.ndarray | int] | None:
        match command:
            case CartesianPosition(pose):
                return {'.pose': Serializers.transform_3d(pose)}
            case CartesianDelta(delta, frame):
                return {
                    '.pose_delta': Serializers.transform_3d(delta),
                    '.pose_delta_frame': Serializers.transform_3d(frame),
                }
            case JointPosition(positions):
                return {'.joints': positions}
            case JointDelta(delta):
                return {'.joint_deltas': delta}
            case Reset():
                return {'.reset': 1}

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
