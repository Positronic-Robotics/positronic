"""The ASPIRE CapApi value types and environment boundary that a program sees, for the ``yam_real`` embodiment.

Ported from NVlabs/ASPIRE ``src/cap_harness/{contracts,errors,protocols}.py`` at commit ``fa00f0e``, under the Apache
License 2.0 (see ``NOTICE``); the field names, defaults and result semantics are ASPIRE's. A camera observation holds
RGB only: its depth, intrinsics and pose are ``None``, because the rig sends no calibrated depth.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

EMBODIMENT = 'yam_real'
JOINT_DIMENSION = 6
PRIMARY_ARM = 'primary'
SECONDARY_ARM = 'secondary'


class ErrorCode(StrEnum):
    """Stable machine-readable categories for recoverable API failures."""

    INVALID_REQUEST = 'invalid_request'
    NOT_FOUND = 'not_found'
    PERCEPTION_FAILED = 'perception_failed'
    NO_SEGMENTATION = 'no_segmentation'
    POINT_CLOUD_FAILED = 'point_cloud_failed'
    GRASP_FAILED = 'grasp_failed'
    NO_GRASP = 'no_grasp'
    IK_FAILED = 'ik_failed'
    IK_UNREACHABLE = 'ik_unreachable'
    PLANNING_FAILED = 'planning_failed'
    EXECUTION_FAILED = 'execution_failed'
    CONTROLLER_FAILED = 'controller_failed'
    PROVIDER_ERROR = 'provider_error'
    PROVIDER_UNAVAILABLE = 'provider_unavailable'
    ADAPTER_FAILED = 'adapter_failed'
    TIMEOUT = 'timeout'
    CANCELLED = 'cancelled'
    TERMINATED = 'terminated'
    TRUNCATED = 'truncated'
    STALE_PLAN = 'stale_plan'
    SAFETY_INTERLOCK = 'safety_interlock'
    UNSUPPORTED = 'unsupported'
    INTERNAL = 'internal'


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    """An operational failure that a result carries across the API boundary."""

    code: ErrorCode
    message: str
    recoverable: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'code', ErrorCode(self.code))
        object.__setattr__(self, 'details', MappingProxyType(dict(self.details)))


class ExecutionStatus(StrEnum):
    """Stable terminal classification for a motion or control attempt."""

    SUCCESS = 'success'
    TERMINATED = 'terminated'
    TRUNCATED = 'truncated'
    TIMEOUT = 'timeout'
    CANCELLED = 'cancelled'
    STALE_PLAN = 'stale_plan'
    PLANNER_FAILED = 'planner_failed'
    CONTROLLER_FAILED = 'controller_failed'
    INVALID_REQUEST = 'invalid_request'
    UNSUPPORTED = 'unsupported'
    FAILED = 'failed'


def _vector(value: object, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.array(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        want is not None and got != want for got, want in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f'{name} must have shape {shape}, got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    array.setflags(write=False)
    return array


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return number


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid pose expressed in ``frame`` using a wxyz unit quaternion."""

    position: np.ndarray
    quaternion_wxyz: np.ndarray
    frame: str

    def __post_init__(self) -> None:
        quaternion = _vector(self.quaternion_wxyz, 'quaternion_wxyz', (4,))
        if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=0.0, atol=1e-5):
            raise ValueError('quaternion_wxyz must be normalized to unit length')
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError('frame must be a non-empty string')
        object.__setattr__(self, 'position', _vector(self.position, 'position', (3,)))
        object.__setattr__(self, 'quaternion_wxyz', quaternion)

    @property
    def position_m(self) -> np.ndarray:
        return self.position

    def as_matrix(self) -> np.ndarray:
        """Return a read-only 4x4 homogeneous transform in ``frame``."""
        w, x, y, z = self.quaternion_wxyz
        transform = np.eye(4)
        transform[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        transform[:3, 3] = self.position
        transform.setflags(write=False)
        return transform


@dataclass(frozen=True, slots=True)
class MotionStrategy:
    """Per-call selection of IK and trajectory planning. The rig solves IK, so the bridge reads none of it."""

    ik_solver: str = 'pyroki'
    trajectory_planner: str = 'interpolation'
    pose_planner: str = 'composed'
    time_dilation_factor: float | None = None
    interpolation_dt_s: float | None = None
    maximum_trajectory_dt_s: float | None = None


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """One camera view: RGB with height, width and channels. Depth, intrinsics and pose are ``None`` on this rig."""

    rgb: np.ndarray
    frame: str
    depth_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    camera_pose: Pose | None = None
    timestamp_s: float | None = None


@dataclass(frozen=True, slots=True)
class RobotState:
    """Per-arm robot state with end-effector poses expressed in ``base_frame``.

    ``gripper_positions`` run from 0 (closed) to 1 (open).
    """

    joint_positions: Mapping[str, np.ndarray]
    joint_velocities: Mapping[str, np.ndarray]
    end_effector_poses: Mapping[str, Pose]
    gripper_positions: Mapping[str, float]
    base_frame: str
    joint_names: Mapping[str, Sequence[str]] | None = None
    timestamp_s: float | None = None
    embodiment: str = EMBODIMENT

    def __post_init__(self) -> None:
        arms = set(self.joint_positions)
        if not arms or {*self.joint_velocities} != arms or {*self.end_effector_poses} != arms:
            raise ValueError('all RobotState per-arm mappings must have identical, non-empty arm names')
        if {*self.gripper_positions} != arms:
            raise ValueError('all RobotState per-arm mappings must have identical, non-empty arm names')
        for arm, pose in self.end_effector_poses.items():
            if pose.frame != self.base_frame:
                raise ValueError(
                    f'end_effector_poses[{arm!r}] is in frame {pose.frame!r}, expected {self.base_frame!r}'
                )
        names = self.joint_names or {arm: tuple(f'joint_{i}' for i in range(JOINT_DIMENSION)) for arm in arms}
        shape = (JOINT_DIMENSION,)
        positions = {
            arm: _vector(value, f'joint_positions[{arm!r}]', shape) for arm, value in self.joint_positions.items()
        }
        velocities = {
            arm: _vector(value, f'joint_velocities[{arm!r}]', shape) for arm, value in self.joint_velocities.items()
        }
        grippers = {
            arm: _unit_interval(value, f'gripper_positions[{arm!r}]') for arm, value in self.gripper_positions.items()
        }
        object.__setattr__(self, 'joint_positions', MappingProxyType(positions))
        object.__setattr__(self, 'joint_velocities', MappingProxyType(velocities))
        object.__setattr__(self, 'end_effector_poses', MappingProxyType(dict(self.end_effector_poses)))
        object.__setattr__(self, 'gripper_positions', MappingProxyType(grippers))
        object.__setattr__(self, 'joint_names', MappingProxyType({arm: tuple(names[arm]) for arm in arms}))

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.joint_positions))


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Task identity and natural-language instruction."""

    suite: str
    task_id: int
    task_name: str
    language: str
    family: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    """Synchronized public observation assembled by an environment adapter."""

    cameras: Mapping[str, CameraObservation]
    robot_state: RobotState
    task_context: TaskContext
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'cameras', MappingProxyType(dict(self.cameras)))


@dataclass(frozen=True, slots=True)
class ArmCommand:
    """One control-period joint command for one arm. ``gripper_position`` runs from 0 (closed) to 1 (open)."""

    mode: str
    target: np.ndarray
    gripper_position: float | None = None
    embodiment: str = EMBODIMENT

    def __post_init__(self) -> None:
        if self.mode != 'joint_position':
            raise ValueError("mode must be 'joint_position'")
        if self.embodiment != EMBODIMENT:
            raise ValueError(f'unsupported embodiment {self.embodiment!r}; expected {EMBODIMENT!r}')
        object.__setattr__(self, 'target', _vector(self.target, 'target', (JOINT_DIMENSION,)))
        if self.gripper_position is not None:
            object.__setattr__(self, 'gripper_position', _unit_interval(self.gripper_position, 'gripper_position'))


@dataclass(frozen=True, slots=True)
class RobotAction:
    """A mapping from arm names to synchronized commands."""

    arms: Mapping[str, ArmCommand]

    def __post_init__(self) -> None:
        if not self.arms or not all(isinstance(command, ArmCommand) for command in self.arms.values()):
            raise ValueError('arms must map arm names to ArmCommand values')
        object.__setattr__(self, 'arms', MappingProxyType(dict(self.arms)))

    @property
    def embodiment(self) -> str:
        return next(iter(self.arms.values())).embodiment


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A fixed-rate joint trajectory of one arm, bound to the measured start state."""

    joint_positions: np.ndarray
    dt_s: float
    joint_names: Sequence[str]
    planner: str
    collision_aware: bool
    expected_start: RobotState
    arm: str = PRIMARY_ARM
    gripper_positions: np.ndarray | None = None
    embodiment: str = EMBODIMENT

    def __post_init__(self) -> None:
        positions = _vector(self.joint_positions, 'joint_positions', (None, JOINT_DIMENSION))
        if positions.shape[0] == 0:
            raise ValueError('joint_positions must contain at least one waypoint')
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError('dt_s must be finite and positive')
        object.__setattr__(self, 'joint_positions', positions)
        if self.gripper_positions is not None:
            grippers = _vector(self.gripper_positions, 'gripper_positions', (positions.shape[0],))
            if np.any((grippers < 0.0) | (grippers > 1.0)):
                raise ValueError('gripper_positions values must be in [0, 1]')
            object.__setattr__(self, 'gripper_positions', grippers)

    @property
    def waypoints(self) -> np.ndarray:
        return self.joint_positions


@dataclass(frozen=True, slots=True)
class SynchronizedTrajectory:
    """Equal-length per-arm joint waypoints executed in the same control ticks."""

    joint_positions: Mapping[str, np.ndarray]
    dt_s: float
    joint_names: Mapping[str, Sequence[str]]
    planner: str
    collision_aware: bool
    expected_start: RobotState
    gripper_positions: Mapping[str, np.ndarray] | None = None
    embodiment: str = EMBODIMENT

    def __post_init__(self) -> None:
        positions = {
            arm: _vector(value, f'joint_positions[{arm!r}]', (None, JOINT_DIMENSION))
            for arm, value in self.joint_positions.items()
        }
        lengths = {value.shape[0] for value in positions.values()}
        if len(lengths) != 1 or lengths == {0}:
            raise ValueError('all synchronized arms must have the same nonzero waypoint count')
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError('dt_s must be finite and positive')
        object.__setattr__(self, 'joint_positions', MappingProxyType(positions))
        if self.gripper_positions is not None:
            if set(self.gripper_positions) != set(positions):
                raise ValueError('gripper_positions must have the same arms as joint_positions')
            count = lengths.pop()
            grippers = {
                arm: _vector(value, f'gripper_positions[{arm!r}]', (count,))
                for arm, value in self.gripper_positions.items()
            }
            if any(np.any((value < 0.0) | (value > 1.0)) for value in grippers.values()):
                raise ValueError('gripper_positions values must be in [0, 1]')
            object.__setattr__(self, 'gripper_positions', MappingProxyType(grippers))

    @property
    def waypoint_count(self) -> int:
        return next(iter(self.joint_positions.values())).shape[0]


def _require_error_unless_ok(ok: bool, error: ApiError | None) -> None:
    if ok and error is not None:
        raise ValueError('a successful result must not carry an error')
    if not ok and error is None:
        raise ValueError('an unsuccessful result must carry an error')


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of exactly one adapter control period."""

    ok: bool
    observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    reward: float | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_error_unless_ok(self.ok, self.error)
        if self.ok and self.observation is None:
            raise ValueError('a successful StepResult must carry an observation')
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))


_STATUS_OF_ERROR = {
    ErrorCode.TERMINATED: ExecutionStatus.TERMINATED,
    ErrorCode.TRUNCATED: ExecutionStatus.TRUNCATED,
    ErrorCode.TIMEOUT: ExecutionStatus.TIMEOUT,
    ErrorCode.CANCELLED: ExecutionStatus.CANCELLED,
    ErrorCode.STALE_PLAN: ExecutionStatus.STALE_PLAN,
    ErrorCode.PLANNING_FAILED: ExecutionStatus.PLANNER_FAILED,
    ErrorCode.IK_FAILED: ExecutionStatus.PLANNER_FAILED,
    ErrorCode.IK_UNREACHABLE: ExecutionStatus.PLANNER_FAILED,
    ErrorCode.EXECUTION_FAILED: ExecutionStatus.CONTROLLER_FAILED,
    ErrorCode.CONTROLLER_FAILED: ExecutionStatus.CONTROLLER_FAILED,
    ErrorCode.ADAPTER_FAILED: ExecutionStatus.CONTROLLER_FAILED,
    ErrorCode.INVALID_REQUEST: ExecutionStatus.INVALID_REQUEST,
    ErrorCode.UNSUPPORTED: ExecutionStatus.UNSUPPORTED,
}


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Recoverable result of a motion or control attempt. ``status`` follows from ``ok`` and ``error``."""

    ok: bool
    steps_executed: int
    final_observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    error: ApiError | None = None
    final_errors: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_error_unless_ok(self.ok, self.error)
        if self.steps_executed < 0:
            raise ValueError('steps_executed must be a non-negative integer')
        object.__setattr__(self, 'final_errors', MappingProxyType({k: float(v) for k, v in self.final_errors.items()}))
        object.__setattr__(self, 'diagnostics', MappingProxyType(dict(self.diagnostics)))

    @property
    def status(self) -> ExecutionStatus:
        if self.ok:
            return ExecutionStatus.SUCCESS
        if self.terminated:
            return ExecutionStatus.TERMINATED
        if self.truncated:
            return ExecutionStatus.TRUNCATED
        assert self.error is not None
        return _STATUS_OF_ERROR.get(self.error.code, ExecutionStatus.FAILED)


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Embodiment boundary with one-control-period ``step`` semantics."""

    @property
    def embodiment(self) -> str: ...

    def reset(self, task_ref: object, seed: int) -> Observation: ...

    def get_task_context(self) -> TaskContext: ...

    def get_observation(self) -> Observation: ...

    def get_robot_state(self) -> RobotState: ...

    def step(self, action: RobotAction) -> StepResult: ...

    def execute_trajectory(self, trajectory: Trajectory | SynchronizedTrajectory) -> ExecutionResult: ...

    def set_gripper(self, position: float, *, arm: str = PRIMARY_ARM) -> ExecutionResult: ...

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult: ...

    def close(self) -> None: ...
