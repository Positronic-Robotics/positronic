"""Absolute hand targets and their bounded Cartesian trajectories."""

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition

Finite = Annotated[float, Field(allow_inf_nan=False)]


class MoveTo(BaseModel):
    """Move the hand to an absolute pose, in metres and radians, and set the gripper (0=open, 1=closed)."""

    model_config = ConfigDict(extra='forbid', strict=True)
    x: Finite
    y: Finite
    z: Finite
    roll: Finite
    pitch: Finite
    yaw: Finite
    gripper: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    note: Annotated[str, Field(min_length=1)]

    @property
    def pose(self) -> geom.Transform3D:
        return geom.Transform3D([self.x, self.y, self.z], geom.Rotation.from_euler([self.roll, self.pitch, self.yaw]))


@dataclass(frozen=True)
class Motion:
    """Bounds on one requested move and the speed of its sampled reference."""

    max_translation: float = 0.05
    max_rotation: float = math.radians(20)
    linear_speed: float = 0.05
    angular_speed: float = math.radians(30)
    fps: float = 25.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        duration = max(self.max_translation / self.linear_speed, self.max_rotation / self.angular_speed, 1 / self.fps)
        if duration > 10 or math.ceil(duration * self.fps) / self.fps + 1 / self.fps > 10:
            raise ValueError('Motion limits must fit within a 10 second trajectory')

    def _clamp(self, start: geom.Transform3D, target: MoveTo) -> MoveTo:
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

    def trajectory(self, start: geom.Transform3D, target: MoveTo) -> tuple[MoveTo, list[dict]]:
        """Return the clamped target and its trajectory from the measured starting pose."""
        target = self._clamp(start, target)
        end = target.pose
        distance = math.hypot(*(end.translation - start.translation))
        angle = (end.rotation * start.rotation.inv).angle
        duration = max(distance / self.linear_speed, angle / self.angular_speed, 1 / self.fps)
        steps = math.ceil(duration * self.fps)
        fractions = np.arange(1, steps + 1) / steps
        trajectory = [
            {
                keys.ROBOT_COMMAND: CartesianPosition(start.interpolate(end, float(fraction))),
                keys.TARGET_GRIP: target.gripper,
            }
            for fraction in fractions
        ]
        return target, trajectory
