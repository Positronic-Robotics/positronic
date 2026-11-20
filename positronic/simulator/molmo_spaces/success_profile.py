"""Success profiles: how the MolmoSpaces server judges that an episode succeeded, and when it ends the episode.

This module must work in MolmoSpaces' isolated interpreter without Positronic installed.

``molmo_native`` is MolmoSpaces' own verdict. ``strict_v1`` judges each task family by a predicate of its own, which
must hold for ``HOLD_SEC``; the episode then runs for ``RECORD_AFTER_SEC`` more, so that its video shows the result.
At each step, a ``strict_v1`` episode records each fact of its predicate under ``strict_v1.<fact>``.
"""

import dataclasses
import enum
import math
from typing import Any, ClassVar, NamedTuple, Protocol

import mujoco
import numpy as np


class SuccessProfile(enum.StrEnum):
    MOLMO_NATIVE = 'molmo_native'
    STRICT_V1 = 'strict_v1'


# The `strict_v1` thresholds. A change to one is a new profile version.
HOLD_SEC = 1.0
RECORD_AFTER_SEC = 1.0
MIN_LIFT_M = 0.03
# A body supports an object when its net contact force on the object carries this fraction of the object's weight,
# along a direction within `SUPPORT_MAX_TILT_RAD` of the vertical.
SUPPORT_WEIGHT_FRAC = 0.5
SUPPORT_MAX_TILT_RAD = math.radians(30)
MAX_RECEPTACLE_LIFT_M = 0.01
MIN_OPEN_FRAC = 0.30
MIN_CLOSED_FRAC = 0.95

# The entry of a MolmoSpaces task's info that holds the task's own verdict at that step.
MOLMO_INFO_SUCCESS = 'success'


class Judged(NamedTuple):
    success: bool
    done: bool
    # The task metrics that the profile records beside the task's own, by name.
    metrics: dict[str, Any]


class Judge(Protocol):
    def observe(self, info: dict[str, Any]) -> dict[str, Any]:
        """The metrics of the current state, with no change to the verdict."""
        ...

    def step(self, task: Any, info: dict[str, Any]) -> Judged:
        """The verdict after one step of ``task``, whose info at that step is ``info``."""
        ...


class MolmoNativeJudge:
    """MolmoSpaces' own verdict: ``judge_success`` at each step, and the episode ends at the first success."""

    def observe(self, info: dict[str, Any]) -> dict[str, Any]:
        return {}

    def step(self, task: Any, info: dict[str, Any]) -> Judged:
        success = bool(task.judge_success())
        return Judged(success, success or bool(task.is_done()), {})


def _pose(data: mujoco.MjData, body: int) -> np.ndarray:
    """The world position and wxyz quaternion of ``body``."""
    return np.concatenate([data.xpos[body], data.xquat[body]])


def _root(model: mujoco.MjModel, geom: int) -> int:
    return int(model.body_rootid[model.geom_bodyid[geom]])


def _contact_forces(data: mujoco.MjData, body: int) -> list[tuple[int, np.ndarray]]:
    """The world force of each contact on the tree of root body ``body``, with the root body of the other side."""
    model = data.model
    forces = []
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        root1, root2 = _root(model, contact.geom1), _root(model, contact.geom2)
        if (root1 == body) == (root2 == body):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        # The contact force acts on geom2; the frame's rows are the normal and the two tangents.
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        forces.append((root1, force) if root2 == body else (root2, -force))
    return forces


class _Support(NamedTuple):
    frac: float  # the upward part of the net force, as a fraction of the object's weight
    supports: bool


def _support(data: mujoco.MjData, body: int, by: frozenset[int] | None, exclude: frozenset[int]) -> _Support:
    """The support of the root bodies ``by`` (any but ``exclude`` when None) for the tree of ``body``."""
    net = np.zeros(3)
    for other, force in _contact_forces(data, body):
        if other not in exclude and (by is None or other in by):
            net += force
    weight = float(data.model.body_subtreemass[body] * np.linalg.norm(data.model.opt.gravity))
    frac = float(net[2] / weight)
    norm = float(np.linalg.norm(net))
    vertical = norm > 0.0 and net[2] / norm >= math.cos(SUPPORT_MAX_TILT_RAD)
    return _Support(frac, bool(vertical and frac >= SUPPORT_WEIGHT_FRAC))


def _touches(data: mujoco.MjData, body: int, other: int) -> bool:
    return any(root == other for root, _ in _contact_forces(data, body))


def _aabb(data: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    """The low and high corners of the world box around every geom in the tree of root body ``body``."""
    model = data.model
    geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == body)
    rotations = data.geom_xmat[geoms].reshape(-1, 3, 3)
    centers = data.geom_xpos[geoms] + np.einsum('gij,gj->gi', rotations, model.geom_aabb[geoms, :3])
    halves = np.einsum('gij,gj->gi', np.abs(rotations), model.geom_aabb[geoms, 3:])
    return (centers - halves).min(axis=0), (centers + halves).max(axis=0)


def _opening(data: mujoco.MjData, joint: int) -> float:
    """How far ``joint`` is open, from 0 at its closed position 0 to 1 at the full width of its range."""
    low, high = data.model.jnt_range[joint]
    return float(abs(data.qpos[data.model.jnt_qposadr[joint]]) / abs(high - low))


class Facts(Protocol):
    """The facts of one family's predicate at one step: a dataclass with the verdict in ``holds``."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    @property
    def holds(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class PickFacts:
    holds: bool
    lift_m: float
    robot_contact: bool
    # The upward force of every body but the robot, as a fraction of the object's weight.
    scene_support_frac: float
    scene_supports: bool
    object_pose: np.ndarray


class Pick:
    """The object is up ``MIN_LIFT_M`` from its start, the robot touches it, and the scene does not support it."""

    def __init__(self, data: mujoco.MjData, obj: int, robot: int):
        self._data, self._obj, self._robot = data, obj, robot
        self._start_z = float(data.xpos[obj][2])

    def facts(self, info: dict[str, Any]) -> PickFacts:
        lift = float(self._data.xpos[self._obj][2]) - self._start_z
        contact = _touches(self._data, self._obj, self._robot)
        scene = _support(self._data, self._obj, None, exclude=frozenset({self._robot}))
        return PickFacts(
            holds=lift >= MIN_LIFT_M and contact and not scene.supports,
            lift_m=lift,
            robot_contact=contact,
            scene_support_frac=scene.frac,
            scene_supports=scene.supports,
            object_pose=_pose(self._data, self._obj),
        )


class _Receptacle:
    """The pose of a receptacle against its start: how far it rose and how far its up axis tilted."""

    def __init__(self, data: mujoco.MjData, body: int):
        self._data, self.body = data, body
        self._start_z = float(data.xpos[body][2])
        self._start_up = data.xmat[body].reshape(3, 3)[:, 2].copy()

    def lift_m(self) -> float:
        return float(self._data.xpos[self.body][2]) - self._start_z

    def tilt_rad(self) -> float:
        up = self._data.xmat[self.body].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(up @ self._start_up, -1.0, 1.0)))


@dataclasses.dataclass(frozen=True)
class PlaceFacts:
    holds: bool
    # The upward force of the receptacle, as a fraction of the object's weight.
    support_frac: float
    receptacle_supports: bool
    # Whether the center of the object's box is inside the receptacle's box, seen from above.
    in_footprint: bool
    robot_contact: bool
    receptacle_lift_m: float
    receptacle_tilt_rad: float
    object_pose: np.ndarray
    receptacle_pose: np.ndarray


class Place:
    """The receptacle supports the released object, whose center is over the receptacle, and the receptacle rests.

    The receptacle rests when it rose at most ``MAX_RECEPTACLE_LIFT_M`` and tilted at most ``max_tilt_rad``. It can
    slide any distance.
    """

    def __init__(self, data: mujoco.MjData, obj: int, receptacle: int, robot: int, max_tilt_rad: float):
        self._data, self._obj, self._robot = data, obj, robot
        self._receptacle = _Receptacle(data, receptacle)
        self._max_tilt_rad = max_tilt_rad

    def facts(self, info: dict[str, Any]) -> PlaceFacts:
        receptacle = self._receptacle.body
        support = _support(self._data, self._obj, frozenset({receptacle}), exclude=frozenset())
        low, high = _aabb(self._data, receptacle)
        obj_low, obj_high = _aabb(self._data, self._obj)
        center = (obj_low + obj_high)[:2] / 2
        in_footprint = bool(np.all(low[:2] <= center) and np.all(center <= high[:2]))
        contact = _touches(self._data, self._obj, self._robot)
        lift, tilt = self._receptacle.lift_m(), self._receptacle.tilt_rad()
        return PlaceFacts(
            holds=support.supports
            and in_footprint
            and not contact
            and lift <= MAX_RECEPTACLE_LIFT_M
            and tilt <= self._max_tilt_rad,
            support_frac=support.frac,
            receptacle_supports=support.supports,
            in_footprint=in_footprint,
            robot_contact=contact,
            receptacle_lift_m=lift,
            receptacle_tilt_rad=tilt,
            object_pose=_pose(self._data, self._obj),
            receptacle_pose=_pose(self._data, receptacle),
        )


@dataclasses.dataclass(frozen=True)
class PlaceNextToFacts:
    holds: bool
    molmo_success: bool
    robot_contact: bool
    object_pose: np.ndarray
    receptacle_pose: np.ndarray


class PlaceNextTo:
    """MolmoSpaces' own next-to verdict, and the robot does not touch the object."""

    def __init__(self, data: mujoco.MjData, obj: int, receptacle: int, robot: int):
        self._data, self._obj, self._receptacle, self._robot = data, obj, receptacle, robot

    def facts(self, info: dict[str, Any]) -> PlaceNextToFacts:
        molmo_success = bool(info[MOLMO_INFO_SUCCESS])
        contact = _touches(self._data, self._obj, self._robot)
        return PlaceNextToFacts(
            holds=molmo_success and not contact,
            molmo_success=molmo_success,
            robot_contact=contact,
            object_pose=_pose(self._data, self._obj),
            receptacle_pose=_pose(self._data, self._receptacle),
        )


@dataclasses.dataclass(frozen=True)
class JointFacts:
    holds: bool
    joint: int  # the MuJoCo id of the joint that the task names
    opening: float


class Open:
    """The joint that the task names is open at least ``MIN_OPEN_FRAC`` of its range."""

    def __init__(self, data: mujoco.MjData, joint: int):
        self._data, self._joint = data, joint

    def facts(self, info: dict[str, Any]) -> JointFacts:
        opening = _opening(self._data, self._joint)
        return JointFacts(holds=opening >= MIN_OPEN_FRAC, joint=self._joint, opening=opening)


class Close:
    """The joint that the task names is closed at least ``MIN_CLOSED_FRAC`` of its range."""

    def __init__(self, data: mujoco.MjData, joint: int):
        self._data, self._joint = data, joint

    def facts(self, info: dict[str, Any]) -> JointFacts:
        opening = _opening(self._data, self._joint)
        return JointFacts(holds=1.0 - opening >= MIN_CLOSED_FRAC, joint=self._joint, opening=opening)


class Check(Protocol):
    def facts(self, info: dict[str, Any]) -> Facts: ...


def _steps(seconds: float, control_dt: float) -> int:
    """The fewest control steps that span ``seconds``."""
    return math.ceil(round(seconds / control_dt, 9))


class StrictJudge:
    """The ``strict_v1`` verdict of one episode.

    The episode succeeds at the step where ``check`` held for ``HOLD_SEC`` of consecutive steps, and the success
    stays. It ends ``RECORD_AFTER_SEC`` after that step, or when the task is done.
    """

    def __init__(self, check: Check, control_dt: float):
        self._check = check
        self._hold_steps = _steps(HOLD_SEC, control_dt)
        self._after_steps = _steps(RECORD_AFTER_SEC, control_dt)
        self._held_steps = 0
        self._steps_since_success: int | None = None

    def _metrics(self, facts: Facts) -> dict[str, Any]:
        fields = {
            **dataclasses.asdict(facts),
            'held_steps': self._held_steps,
            'success': self._steps_since_success is not None,
        }
        return {f'{SuccessProfile.STRICT_V1}.{name}': value for name, value in fields.items()}

    def observe(self, info: dict[str, Any]) -> dict[str, Any]:
        """The metrics of the current state, with no step of the hold window."""
        return self._metrics(self._check.facts(info))

    def step(self, task: Any, info: dict[str, Any]) -> Judged:
        facts = self._check.facts(info)
        self._held_steps = self._held_steps + 1 if facts.holds else 0
        if self._steps_since_success is not None:
            self._steps_since_success += 1
        elif self._held_steps >= self._hold_steps:
            self._steps_since_success = 0
        since = self._steps_since_success
        recorded_after = since is not None and since >= self._after_steps
        return Judged(since is not None, recorded_after or bool(task.is_done()), self._metrics(facts))
