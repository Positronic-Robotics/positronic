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
        if max(self.max_translation / self.linear_speed, self.max_rotation / self.angular_speed, 1 / self.fps) > 10:
            raise ValueError('Motion limits must fit within a 10 second trajectory')

    def duration(self, start: geom.Transform3D, end: geom.Transform3D) -> float:
        """Validate a displacement and return the minimum duration allowed by the speed bounds."""
        distance = float(np.linalg.norm(end.translation - start.translation))
        angle = (end.rotation * start.rotation.inv).angle
        if distance > self.max_translation + 1e-9:
            raise ValueError(f'Move is {distance:.4f} m; maximum is {self.max_translation:.4f} m. Split the move.')
        if angle > self.max_rotation + 1e-9:
            raise ValueError(f'Rotation is {angle:.4f} rad; maximum is {self.max_rotation:.4f} rad. Split the move.')
        return max(distance / self.linear_speed, angle / self.angular_speed, 1 / self.fps)

    def trajectory(self, start: geom.Transform3D, target: MoveTo) -> list[dict]:
        end = target.pose
        duration = self.duration(start, end)
        steps = math.ceil(duration * self.fps)
        times = np.arange(1, steps + 1) / self.fps
        fractions = times / times[-1]
        return [
            {
                keys.ROBOT_COMMAND: CartesianPosition(start.interpolate(end, float(fraction))),
                keys.TARGET_GRIP: target.gripper,
                keys.ACTION_TIMESTAMP: float(timestamp),
            }
            for fraction, timestamp in zip(fractions, times, strict=True)
        ]
