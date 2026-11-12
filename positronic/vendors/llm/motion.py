"""Absolute hand waypoints and their bounded Cartesian trajectories."""

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.utils import check_finite_positive

Finite = Annotated[float, Field(allow_inf_nan=False)]


class Waypoint(BaseModel):
    """An absolute hand pose, in metres and radians, and the gripper (0=open, 1=closed) set on the way to it."""

    model_config = ConfigDict(extra='forbid', strict=True)
    x: Finite
    y: Finite
    z: Finite
    roll: Finite
    pitch: Finite
    yaw: Finite
    gripper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @classmethod
    def at(cls, pose: geom.Transform3D, gripper: float) -> 'Waypoint':
        x, y, z = pose.translation
        roll, pitch, yaw = pose.rotation.as_euler
        return cls(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, gripper=gripper)

    @property
    def pose(self) -> geom.Transform3D:
        return geom.Transform3D([self.x, self.y, self.z], geom.Rotation.from_euler([self.roll, self.pitch, self.yaw]))


class MovePath(BaseModel):
    """Move the hand through absolute waypoints in order, without an observation between them."""

    model_config = ConfigDict(extra='forbid', strict=True)
    waypoints: Annotated[list[Waypoint], Field(min_length=1)]
    note: Annotated[str, Field(min_length=1)]


class MoveArmsPath(BaseModel):
    """Move the hands through absolute waypoints in order, without an observation between them. A waypoint maps the
    name of an arm to its hand target; an arm that it leaves out holds its previous target."""

    model_config = ConfigDict(extra='forbid', strict=True)
    waypoints: Annotated[list[Annotated[dict[str, Waypoint], Field(min_length=1)]], Field(min_length=1)]
    note: Annotated[str, Field(min_length=1)]


# The waypoint of each arm, keyed by the arm's name; `None` names the arm of a one-arm embodiment.
Targets = Mapping[str | None, Waypoint]


class Trajectory(NamedTuple):
    """The clamped targets of every arm at each waypoint, the commands of each segment and gripper hold, and
    whether a clamp changed a requested target."""

    bounded: list[Targets]
    parts: list[list[dict]]
    clamped: bool


@dataclass(frozen=True)
class Motion:
    """Bounds on each segment of a requested path, on its waypoint count, and the speed of its sampled reference."""

    max_translation: float = 0.4
    max_rotation: float = math.radians(90)
    linear_speed: float = 0.15
    angular_speed: float = math.radians(30)
    fps: float = 25.0
    grip_hold: float = 0.5
    max_waypoints: int = 5

    def __post_init__(self):
        for name, value in vars(self).items():
            check_finite_positive(value, name)
        if not isinstance(self.max_waypoints, int):
            raise ValueError('max_waypoints must be an integer')
        duration = max(self.max_translation / self.linear_speed, self.max_rotation / self.angular_speed, 1 / self.fps)
        if duration > 10 or math.ceil(duration * self.fps) / self.fps + 1 / self.fps > 10:
            raise ValueError('Motion limits must fit one segment within a 10 second trajectory')

    def _clamp(self, start: geom.Transform3D, target: Waypoint) -> Waypoint:
        end = target.pose
        offset = end.translation - start.translation
        distance = math.hypot(*offset)
        angle = (end.rotation * start.rotation.inv).angle
        bounded = target.model_copy()
        if distance > self.max_translation:
            bounded.x, bounded.y, bounded.z = start.translation + offset / distance * self.max_translation
        if angle > self.max_rotation:
            rotation = start.rotation.interpolate(end.rotation, self.max_rotation / angle)
            bounded.roll, bounded.pitch, bounded.yaw = rotation.as_euler
        return bounded

    @staticmethod
    def _commands(poses: Mapping[str | None, geom.Transform3D], targets: Targets) -> dict:
        commands = {}
        for arm, pose in poses.items():
            commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = CartesianPosition(pose)
            commands[keys.arm_channel(keys.TARGET_GRIP, arm)] = targets[arm].gripper
        return commands

    def _segment(self, starts: Mapping[str | None, geom.Transform3D], targets: Targets) -> list[dict]:
        def span(start: geom.Transform3D, end: geom.Transform3D) -> float:
            distance = math.hypot(*(end.translation - start.translation))
            angle = (end.rotation * start.rotation.inv).angle
            return max(distance / self.linear_speed, angle / self.angular_speed)

        ends = {arm: target.pose for arm, target in targets.items()}
        duration = max(*(span(starts[arm], end) for arm, end in ends.items()), 1 / self.fps)
        steps = math.ceil(duration * self.fps)
        return [
            self._commands({arm: starts[arm].interpolate(end, float(fraction)) for arm, end in ends.items()}, targets)
            for fraction in np.arange(1, steps + 1) / steps
        ]

    def trajectory(
        self, starts: Mapping[str | None, geom.Transform3D], held: Targets, path: list[Targets]
    ) -> Trajectory:
        """The trajectory of ``path`` from ``starts``.

        ``held`` is the target of each arm of ``starts`` before the path. An arm that a waypoint of ``path`` leaves
        out holds its previous clamped target. All arms reach a waypoint together.
        """
        grips = {arm: target.gripper for arm, target in held.items()}
        bounded, parts, clamped = [], [], False
        for waypoint in path:
            requested = {arm: waypoint.get(arm, held[arm]) for arm in starts}
            targets = {arm: self._clamp(start, requested[arm]) for arm, start in starts.items()}
            clamped = clamped or targets != requested
            parts.append(self._segment(starts, targets))
            if any(target.gripper != grips[arm] for arm, target in targets.items()):
                hold = self._commands({arm: target.pose for arm, target in targets.items()}, targets)
                parts.append([hold] * math.ceil(self.grip_hold * self.fps))
            bounded.append(targets)
            starts = {arm: target.pose for arm, target in targets.items()}
            grips = {arm: target.gripper for arm, target in targets.items()}
            held = targets
        return Trajectory(bounded, parts, clamped)


def closest_hand_distances(
    parts: list[list[dict]], arms: tuple[str | None, ...]
) -> dict[tuple[str | None, str | None], float]:
    """The smallest distance between the hands of each pair of ``arms`` over the commands of ``parts``."""
    channels = {arm: keys.arm_channel(keys.ROBOT_COMMAND, arm) for arm in arms}
    return {
        (a, b): min(
            (
                math.dist(command[channels[a]].pose.translation, command[channels[b]].pose.translation)
                for part in parts
                for command in part
            ),
            default=math.inf,
        )
        for a, b in itertools.combinations(arms, 2)
    }


def closest_approach(start: np.ndarray, end: np.ndarray) -> float:
    """The smallest length of a vector that moves in a straight line from ``start`` to ``end``."""
    step = end - start
    span = float(np.dot(step, step))
    fraction = 0.0 if span == 0 else float(np.clip(-np.dot(start, step) / span, 0, 1))
    return float(np.linalg.norm(start + fraction * step))
