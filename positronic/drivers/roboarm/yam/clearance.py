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
    ``obstacles``; None checks nothing. Near them, the gap closes no faster than a brake of ``approach_decel_m_s2``
    stops it at ``keep_clear_m``. Where a brake at the acceleration cap does not keep the gap, the gap holds over the
    cap: the arm came into the slow zone faster than the cap can stop it there. A Cartesian target is kept
    ``obstacle_margin_m`` above each obstacle it is over. The obstacles are in the frame of the targets: the world
    frame of the arm.

    A park or a start move checks its joint ramp against the obstacles first. Where the ramp does not keep clear,
    the gripper rises, by ``park_lift_m`` at most and for ``park_lift_timeout_s`` at most, or until the arm is
    ``park_clear_m`` clear of them. A ramp that still does not keep clear fails the move.
    """

    max_joint_accel_rad_s2: float | None = None
    keep_clear_m: float | None = None
    obstacles: tuple[Box, ...] = ()
    obstacle_margin_m: float = 0.01
    approach_decel_m_s2: float = 3.0
    park_clear_m: float = 0.06
    park_lift_m: float = 0.08
    park_lift_timeout_s: float = 2.0

    def __post_init__(self):
        for value in (self.max_joint_accel_rad_s2, self.keep_clear_m):
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f'a stream limit needs a finite positive value or None, not {value}')
        for value in (self.approach_decel_m_s2, self.park_clear_m, self.park_lift_m, self.park_lift_timeout_s):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f'a stream setting needs a finite positive value, not {value}')
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
# Joint 6 turns the fingers about an axis of link 5, so their hulls stay 51 mm apart at every pose.
_ALWAYS_NEAR = {
    frozenset(pair)
    for pair in [
        ('base', 'link1'),
        ('base', 'link2'),
        ('link3', 'link5'),
        ('link5', 'tip_left'),
        ('link5', 'tip_right'),
    ]
}
_OBSTACLE = 'obstacle'
_FINGERS = ('joint7', 'joint8')
# The arm slows down from this far past ``keep_clear_m``: a brake of 3 m/s2 stops 0.6 m/s in 6 cm
_SLOW_ZONE_M = 0.08


@dataclass(frozen=True)
class Near:
    """A pair of the arm and itself or an obstacle: the names of the two, its gap in meters, and how the gap changes
    with each joint."""

    pair: frozenset[str]
    gap_m: float
    slope: np.ndarray  # m/rad, one value per arm joint


class Clearance:
    """The gap between the arm and itself or ``obstacles``, for a stream that keeps ``keep_clear_m``. The gripper shell
    counts, which the MJCF leaves out of collisions. A hull is larger than its link, so a gap is a lower bound."""

    def __init__(self, obstacles: Sequence[Box], keep_clear_m: float):
        spec = mj.MjSpec.from_file(package_assets_path(YAM_MJCF_PATH))
        for index, box in enumerate(obstacles):
            lower, upper = np.asarray(box.lower), np.asarray(box.upper)
            spec.worldbody.add_geom(
                name=f'{_OBSTACLE}{index}',
                type=mj.mjtGeom.mjGEOM_BOX,
                size=((upper - lower) / 2).tolist(),
                pos=((upper + lower) / 2).tolist(),
            )
        self.obstacles = tuple(obstacles)
        self.keep_clear_m = keep_clear_m
        self.reach_m = keep_clear_m + _SLOW_ZONE_M  # pairs farther than this read as this gap
        self._model = spec.compile()
        self._model.geom_margin[:] = self.reach_m
        self._names: list[str] = []
        for g in range(self._model.ngeom):
            body = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, self._model.geom_bodyid[g])
            if body == 'gripper':
                self._model.geom_contype[g] = self._model.geom_conaffinity[g] = 1
            # Each obstacle has its name, so a link near two of them is cut for each
            self._names.append(mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_GEOM, g) if body == 'world' else body)
        self._obstacle_names = {name for name in self._names if name.startswith(_OBSTACLE)}
        self._data = mj.MjData(self._model)
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in YAM_JOINT_NAMES])
        self._finger_ids = np.array([self._model.joint(name).qposadr.item() for name in _FINGERS])
        self._finger_travel = np.array([self._model.joint(name).range[1] for name in _FINGERS])

    def nears(self, q: np.ndarray, open_share: float) -> list[Near]:
        """The pairs nearer than ``reach_m`` at joints ``q``, the nearest point of each pair, with the fingers open by
        ``open_share`` of their travel (0 closed, 1 open)."""
        return [
            Near(pair, float(contact.dist), self._slope(contact))
            for pair, contact in self._nearest(q, open_share, obstacles_only=False).items()
        ]

    def gaps(self, q: np.ndarray, open_share: float, *, obstacles_only: bool) -> dict[frozenset[str], float]:
        """The gap of each pair nearer than ``reach_m`` at joints ``q``; with ``obstacles_only``, of the pairs of the
        arm and an obstacle."""
        return {pair: float(contact.dist) for pair, contact in self._nearest(q, open_share, obstacles_only).items()}

    def closes_a_gap(
        self,
        q: np.ndarray,
        before: dict[frozenset[str], float],
        open_share: float,
        tolerance_m: float,
        *,
        obstacles_only: bool,
    ) -> bool:
        """Whether at joints ``q`` a pair is nearer than ``keep_clear_m`` less ``tolerance_m``, and nearer than its gap
        ``before``. Each pair counts: a pair deep in the gap does not hide another that enters it."""
        floor = self.keep_clear_m - tolerance_m
        return any(
            gap < floor and gap < before.get(pair, self.reach_m)
            for pair, gap in self.gaps(q, open_share, obstacles_only=obstacles_only).items()
        )

    def gap(self, q: np.ndarray, open_share: float) -> float:
        """The least gap at joints ``q`` in meters, ``reach_m`` when nothing is nearer, negative where hulls overlap."""
        return min(self.gaps(q, open_share, obstacles_only=False).values(), default=self.reach_m)

    def obstacle_gap(self, q: np.ndarray, open_share: float) -> float:
        """The least gap between the arm and the obstacles at joints ``q``; ``reach_m`` when none is nearer."""
        return min(self.gaps(q, open_share, obstacles_only=True).values(), default=self.reach_m)

    def _nearest(self, q: np.ndarray, open_share: float, obstacles_only: bool) -> dict[frozenset[str], mj.MjContact]:
        """The nearest contact of each counted pair at joints ``q``."""
        self._place(q, open_share)
        data = self._data
        nearest: dict[frozenset[str], mj.MjContact] = {}
        for contact in data.contact[: data.ncon]:
            pair = frozenset((self._names[contact.geom1], self._names[contact.geom2]))
            if pair in _ALWAYS_NEAR or (obstacles_only and not pair & self._obstacle_names):
                continue
            if pair not in nearest or contact.dist < nearest[pair].dist:
                nearest[pair] = contact
        return nearest

    def _place(self, q: np.ndarray, open_share: float) -> None:
        data = self._data
        data.qpos[:] = 0.0
        data.qpos[self._qpos_ids] = q
        data.qpos[self._finger_ids] = float(np.clip(open_share, 0.0, 1.0)) * self._finger_travel
        mj.mj_fwdPosition(self._model, data)

    def _slope(self, contact: mj.MjContact) -> np.ndarray:
        """How the gap of ``contact`` changes with each joint. The normal points from geom1 to geom2: the gap grows as
        the point on geom2 moves along it, away from geom1."""
        model, data = self._model, self._data
        slope = np.zeros(len(self._dof_ids))
        normal = contact.frame[:3]
        for side, sign in ((contact.geom2, 1.0), (contact.geom1, -1.0)):
            body = model.geom_bodyid[side]
            if body == 0:
                continue
            shift = np.zeros((3, model.nv))
            mj.mj_jac(model, data, shift, None, contact.pos, body)
            slope += sign * normal @ shift[:, self._dof_ids]
        return slope
