"""Absolute hand waypoints and their bounded Cartesian trajectories."""

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition

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

    @property
    def pose(self) -> geom.Transform3D:
        return geom.Transform3D([self.x, self.y, self.z], geom.Rotation.from_euler([self.roll, self.pitch, self.yaw]))


class MovePath(BaseModel):
    """Move the hand through absolute waypoints in order, without an observation between them."""

    model_config = ConfigDict(extra='forbid', strict=True)
    waypoints: Annotated[list[Waypoint], Field(min_length=1)]
    note: Annotated[str, Field(min_length=1)]


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
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
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

    def _segment(self, start: geom.Transform3D, target: Waypoint) -> list[dict]:
        end = target.pose
        distance = math.hypot(*(end.translation - start.translation))
        angle = (end.rotation * start.rotation.inv).angle
        duration = max(distance / self.linear_speed, angle / self.angular_speed, 1 / self.fps)
        steps = math.ceil(duration * self.fps)
        return [
            {
                keys.ROBOT_COMMAND: CartesianPosition(start.interpolate(end, float(fraction))),
                keys.TARGET_GRIP: target.gripper,
            }
            for fraction in np.arange(1, steps + 1) / steps
        ]

    def trajectory(
        self, start: geom.Transform3D, grip: float, path: list[Waypoint]
    ) -> tuple[list[Waypoint], list[dict]]:
        """Return the clamped waypoints and their trajectory from the measured pose and the gripper in force."""
        bounded, trajectory = [], []
        for waypoint in path:
            target = self._clamp(start, waypoint)
            trajectory.extend(self._segment(start, target))
            if target.gripper != grip:
                hold = {keys.ROBOT_COMMAND: CartesianPosition(target.pose), keys.TARGET_GRIP: target.gripper}
                trajectory.extend([hold] * math.ceil(self.grip_hold * self.fps))
            bounded.append(target)
            start, grip = target.pose, target.gripper
        return bounded, trajectory
