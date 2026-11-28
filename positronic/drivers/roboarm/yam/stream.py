"""How a streamed target moves a YAM arm, and the fixed obstacles that its gripper keeps above."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

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

    ``max_joint_accel_rad_s2`` caps the joint acceleration of a followed Cartesian target; None caps nothing.
    ``obstacles`` are boxes in the world frame. A Cartesian target keeps the gripper ``obstacle_margin_m`` above them;
    a joint target is taken as it is. Before a joint ramp that dips toward an obstacle, the tool rises ``raise_m`` at
    most, for ``raise_timeout_s`` at most.
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


@dataclass(frozen=True, eq=False)
class Hull:
    """A convex polytope: its ``vertices`` (N x 3), its ``edges`` as pairs of rows of ``vertices``, its ``facets``
    (F x 4), and a ball at ``center`` of ``radius`` that holds it. A point p is inside where
    ``facets[:, :3] @ p + facets[:, 3] <= 0``."""

    vertices: np.ndarray
    edges: np.ndarray
    facets: np.ndarray
    center: np.ndarray
    radius: float

    @classmethod
    def of(cls, points: np.ndarray) -> 'Hull':
        """The convex hull of ``points`` (N x 3)."""
        hull = ConvexHull(points)
        vertices = hull.points[hull.vertices]
        rows = np.empty(len(points), dtype=np.intp)
        rows[hull.vertices] = np.arange(len(vertices))
        corners = rows[hull.simplices]
        edges = np.unique(
            np.sort(np.vstack([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]]), axis=1), axis=0
        )
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        return cls(vertices, edges, hull.equations, center, float(np.linalg.norm(vertices - center, axis=1).max()))


# rules-allow: primitive-type — a `Transform3D` product costs about 73 µs per call, and this runs on each 10 ms tick
def height_above(frame: np.ndarray, hulls: Sequence[Hull], boxes: Sequence[Box]) -> float:
    """How far the lowest point of ``hulls``, placed by the 4 x 4 ``frame``, that is over a box is above the top of
    that box; infinite over none."""
    rotation, shift, height = frame[:3, :3], frame[:3, 3], np.inf
    for hull in hulls:
        (cx, cy, cz), r = rotation @ hull.center + shift, hull.radius
        z = None
        for box in boxes:
            (x0, y0, _), (x1, y1, top) = box.lower, box.upper
            if cz - r - top >= height or cx + r < x0 or x1 < cx - r or cy + r < y0 or y1 < cy - r:
                continue
            if x0 <= cx - r and cx + r <= x1 and y0 <= cy - r and cy + r <= y1:
                z = hull.vertices @ rotation[2] + shift[2] if z is None else z
                height = min(height, float(z.min()) - top)
            else:
                height = min(height, _lowest_over(hull, rotation, shift, box) - top)
    return height


def _lowest_over(hull: Hull, rotation: np.ndarray, shift: np.ndarray, box: Box) -> float:
    """The lowest z of ``hull``, turned by ``rotation`` and moved by ``shift``, over the footprint of ``box``; infinite
    over no part of it."""
    (x0, y0, _), (x1, y1, _) = box.lower, box.upper
    vertices = hull.vertices @ rotation.T + shift
    x, y, z = vertices.T
    low, high = (x.min(), y.min()), (x.max(), y.max())
    if high[0] < x0 or x1 < low[0] or high[1] < y0 or y1 < low[1]:
        return np.inf
    lowest = float(z[(x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)].min(initial=np.inf))
    if x0 <= low[0] and high[0] <= x1 and y0 <= low[1] and high[1] <= y1:
        return lowest
    # Over part of the footprint, the lowest point can also be where an edge crosses a side, or a facet a corner
    for axis, side, (start, end) in ((0, x0, (y0, y1)), (0, x1, (y0, y1)), (1, y0, (x0, x1)), (1, y1, (x0, x1))):
        if low[axis] < side < high[axis]:
            first, second = vertices[hull.edges[:, 0], axis] - side, vertices[hull.edges[:, 1], axis] - side
            crosses = (first < 0) != (second < 0)
            a, b = vertices[hull.edges[crosses, 0]], vertices[hull.edges[crosses, 1]]
            share = first[crosses] / (first[crosses] - second[crosses])
            points = a + share[:, None] * (b - a)
            along = points[:, 1 - axis]
            lowest = min(lowest, float(points[(start <= along) & (along <= end), 2].min(initial=np.inf)))
    corners = [(cx, cy) for cx in (x0, x1) for cy in (y0, y1) if low[0] < cx < high[0] and low[1] < cy < high[1]]
    if corners:
        normals = hull.facets[:, :3] @ rotation.T
        offsets = hull.facets[:, 3] - normals @ shift
        down, up = normals[:, 2] < 0, normals[:, 2] > 0
        for cx, cy in corners:
            # A facet holds the point (cx, cy, z) where the z of its normal times z is at most ``reach``
            reach = -(normals[:, 0] * cx + normals[:, 1] * cy + offsets)
            bottom = (reach[down] / normals[down, 2]).max(initial=-np.inf)
            top = (reach[up] / normals[up, 2]).min(initial=np.inf)
            if bottom <= top and np.all(reach[~down & ~up] >= 0):
                lowest = min(lowest, float(bottom))
    return lowest


def lift_above(
    pose: geom.Transform3D, hulls: Sequence[Hull], boxes: Sequence[Box], margin_m: float
) -> geom.Transform3D:
    """``pose`` raised straight up until ``hulls``, given in the frame of ``pose``, are ``margin_m`` above each box
    that they are over."""
    lack = margin_m - height_above(pose.as_matrix, hulls, boxes)
    if lack <= 0:
        return pose
    return geom.Transform3D(pose.translation + np.array([0.0, 0.0, lack]), pose.rotation)
