"""The gap between the YAM arm and itself or fixed obstacles around it, on the convex hulls of the vendored MJCF."""

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from positronic import geom
from positronic.utils import package_assets_path

from ..models import YAM_JOINT_NAMES, YAM_MJCF_PATH


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

    ``max_joint_accel_rad_s2`` caps the joint acceleration, and a joint brakes in time to stop at its target; None
    caps nothing. ``keep_clear_m`` stops the arm before a step that takes it nearer than this to itself or
    ``obstacles``; None checks nothing. A Cartesian target is kept ``obstacle_margin_m`` above each obstacle it is
    over. The obstacles are in the frame of the targets: the world frame of the arm.
    """

    max_joint_accel_rad_s2: float | None = None
    keep_clear_m: float | None = None
    obstacles: tuple[Box, ...] = ()
    obstacle_margin_m: float = 0.01

    def __post_init__(self):
        for value in (self.max_joint_accel_rad_s2, self.keep_clear_m):
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f'a stream limit needs a finite positive value or None, not {value}')
        if not (np.isfinite(self.obstacle_margin_m) and self.obstacle_margin_m >= 0):
            raise ValueError(f'obstacle_margin_m needs a finite value >= 0, not {self.obstacle_margin_m}')


# A stream with only the joint speed cap: no acceleration cap, no gap check, no obstacles
UNLIMITED_STREAM = StreamTuning()


def lift_above(pose: geom.Transform3D, boxes: Sequence[Box], margin_m: float) -> geom.Transform3D:
    """``pose`` with its point raised to ``margin_m`` above the top of each box that it is over and lower than."""
    x, y, z = pose.translation
    lifted = z
    for box in boxes:
        over = box.lower[0] <= x <= box.upper[0] and box.lower[1] <= y <= box.upper[1]
        if over and lifted < box.upper[2] + margin_m:
            lifted = box.upper[2] + margin_m
    if lifted == z:
        return pose
    return geom.Transform3D(np.array([x, y, lifted]), pose.rotation)


# Pairs the gap leaves out. The foot and link 1 sit on each other, and the hulls of the base and link 2 stay 9 mm apart
# as joint 2 turns. The stops of joint 4 keep links 3 and 5 apart, and their hulls meet 3 mm short of its lower stop.
# An obstacle under the foot touches the foot and link 1.
_ALWAYS_NEAR = {frozenset(pair) for pair in [('base', 'link1'), ('base', 'link2'), ('link3', 'link5')]}
_OBSTACLE = 'obstacle'
_ON_OBSTACLES = {'base', 'link1'}


class Clearance:
    """The smallest gap between the arm and itself or ``boxes``, up to ``reach_m``. The gripper shell counts, which
    the MJCF leaves out of collisions. A hull is larger than its link, so a gap is a lower bound."""

    def __init__(self, boxes: Sequence[Box], reach_m: float):
        spec = mj.MjSpec.from_file(package_assets_path(YAM_MJCF_PATH))
        for index, box in enumerate(boxes):
            lower, upper = np.asarray(box.lower), np.asarray(box.upper)
            spec.worldbody.add_geom(
                name=f'{_OBSTACLE}{index}',
                type=mj.mjtGeom.mjGEOM_BOX,
                size=((upper - lower) / 2).tolist(),
                pos=((upper + lower) / 2).tolist(),
            )
        self._model = spec.compile()
        self._model.geom_margin[:] = reach_m
        self._names: list[str] = []
        for g in range(self._model.ngeom):
            body = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, self._model.geom_bodyid[g])
            if body == 'gripper':
                self._model.geom_contype[g] = self._model.geom_conaffinity[g] = 1
            self._names.append(_OBSTACLE if body == 'world' else body)
        self._data = mj.MjData(self._model)
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in YAM_JOINT_NAMES])
        self.reach_m = reach_m

    @staticmethod
    def _counts(a: str, b: str) -> bool:
        pair = {a, b}
        return frozenset(pair) not in _ALWAYS_NEAR and not (_OBSTACLE in pair and pair & _ON_OBSTACLES)

    def gap(self, q: np.ndarray) -> float:
        """The gap at joints ``q`` in meters, ``reach_m`` when nothing is nearer, negative where hulls overlap."""
        return self.gap_and_slope(q)[0]

    def gap_and_slope(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """The gap at joints ``q``, and how the gap of its nearest pair changes with each joint in m/rad. The slope is
        zero where nothing is nearer than ``reach_m``."""
        model, data = self._model, self._data
        data.qpos[:] = 0.0
        data.qpos[self._qpos_ids] = q
        mj.mj_fwdPosition(model, data)
        nearest = None
        for contact in data.contact[: data.ncon]:
            if self._counts(self._names[contact.geom1], self._names[contact.geom2]):
                if nearest is None or contact.dist < nearest.dist:
                    nearest = contact
        slope = np.zeros(len(self._dof_ids))
        if nearest is None:
            return self.reach_m, slope
        # The normal points from geom1 to geom2: the gap grows as the point on geom2 moves along it, away from geom1
        normal = nearest.frame[:3]
        for side, sign in ((nearest.geom2, 1.0), (nearest.geom1, -1.0)):
            body = model.geom_bodyid[side]
            if body == 0:
                continue
            shift = np.zeros((3, model.nv))
            mj.mj_jac(model, data, shift, None, nearest.pos, body)
            slope += sign * normal @ shift[:, self._dof_ids]
        return float(nearest.dist), slope
