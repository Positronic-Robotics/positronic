"""How a streamed target moves a YAM arm, and the fixed obstacles that its gripper keeps above."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from positronic import geom


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in meters: a table top, the plate under the foot."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self):
        corners = np.array([self.lower, self.upper], dtype=np.float64)
        if corners.shape != (2, 3) or not np.all(np.isfinite(corners)) or not np.all(corners[0] < corners[1]):
            raise ValueError(f'a box needs finite 3D corners with lower < upper, not {self.lower}, {self.upper}')


@dataclass(frozen=True)
class StreamTuning:
    """How a streamed target moves this arm, past the joint speed cap that every stream has.

    ``max_joint_accel_rad_s2`` caps the joint acceleration of a followed Cartesian target, and a joint brakes in time
    to stop at its target; None caps nothing. A joint target keeps the speed cap alone. A followed step that the
    obstacles raise or hold can pass the cap. ``obstacles`` are boxes in the
    world frame. A Cartesian target, streamed or in a blocking move, keeps the gripper ``obstacle_margin_m`` above
    each obstacle it is over. A followed step that takes the gripper a little lower is raised, and one that takes it
    into an obstacle from its side is held. A joint target, streamed or in a blocking move, is taken as it is.

    A park, a start move or a blocking move ramps in joint space. While its ramp takes the gripper lower over an
    obstacle than ``obstacle_margin_m`` and than both ends, the tool first rises straight up, by ``raise_m`` at most.
    A ramp that does not keep clear within ``raise_timeout_s`` fails the move.
    """

    max_joint_accel_rad_s2: float | None = None
    obstacles: tuple[Box, ...] = ()
    obstacle_margin_m: float = 0.01
    raise_m: float = 0.15
    raise_timeout_s: float = 2.0

    def __post_init__(self):
        accel = self.max_joint_accel_rad_s2
        if accel is not None and not (np.isfinite(accel) and accel > 0):
            raise ValueError(f'max_joint_accel_rad_s2 needs a finite positive value or None, not {accel}')
        for value in (self.raise_m, self.raise_timeout_s):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f'a stream setting needs a finite positive value, not {value}')
        if not (np.isfinite(self.obstacle_margin_m) and self.obstacle_margin_m >= 0):
            raise ValueError(f'obstacle_margin_m needs a finite value >= 0, not {self.obstacle_margin_m}')


# A stream with only the joint speed cap: no acceleration cap, no obstacles
UNLIMITED_STREAM = StreamTuning()


def heights_above(points: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    """How far each of ``points`` (N x 3) is above the top of the highest box that it is over; infinite over none."""
    heights = np.full(len(points), np.inf)
    for box in boxes:
        x, y = points[:, 0], points[:, 1]
        over = (box.lower[0] <= x) & (x <= box.upper[0]) & (box.lower[1] <= y) & (y <= box.upper[1])
        heights = np.where(over, np.minimum(heights, points[:, 2] - box.upper[2]), heights)
    return heights


def points_at(pose: geom.Transform3D, points: np.ndarray) -> np.ndarray:
    """``points`` (N x 3) given in the frame of ``pose``, in the frame that ``pose`` is in."""
    return points @ pose.rotation.as_rotation_matrix.T + pose.translation


def lift_above(pose: geom.Transform3D, points: np.ndarray, boxes: Sequence[Box], margin_m: float) -> geom.Transform3D:
    """``pose`` raised straight up until each of ``points``, given in the frame of ``pose``, is ``margin_m`` above
    each box that it is over."""
    lack = margin_m - float(heights_above(points_at(pose, points), boxes).min(initial=np.inf))
    if lack <= 0:
        return pose
    return geom.Transform3D(pose.translation + np.array([0.0, 0.0, lack]), pose.rotation)
