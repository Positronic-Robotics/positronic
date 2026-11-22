"""Run an ASPIRE CapApi program against a stream of positronic observations.

A call that moves the robot puts its target in a queue and blocks until ``CapBridge.answer`` reads an observation that
ends the move. The rig plays the targets as action chunks: its driver solves IK and limits speed. The vendor README
states how a move ends and what each result field means.
"""

import hashlib
import importlib
import logging
import math
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from functools import cached_property, partial, wraps
from types import MappingProxyType
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, JsonValue, model_validator
from pydantic_core import to_jsonable_python

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition
from positronic.policy.base import Obs, PolicyEnd
from positronic.policy.keys import END
from positronic.utils import check_finite_positive

from . import geometry
from .contracts import (
    DEFAULT_EMBODIMENT,
    PRIMARY_ARM,
    SECONDARY_ARM,
    ApiError,
    ArmCommand,
    CameraObservation,
    ErrorCode,
    ExecutionResult,
    MotionStrategy,
    Observation,
    PointCloud,
    Pose,
    RobotAction,
    RobotState,
    Segmentation,
    SegmentationProvider,
    SegmentationSet,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)

logger = logging.getLogger(__name__)

# ASPIRE's CapApi judges the orientation of a pose move against this bound, whatever the call's tolerance.
POSE_ORIENTATION_TOLERANCE_RAD = math.radians(10.0)
# ASPIRE's CapApi refuses a trajectory whose start is farther than this from the measured joints.
TRAJECTORY_START_TOLERANCE_RAD = 0.02
# The TaskContext fields that a positronic observation does not carry.
TASK_SUITE = TASK_FAMILY = 'positronic'


# Absorbs float error in `round_s * fps`, so 1.1 s at 50 Hz is 55 ticks and not 56.
_TICK_EPSILON = 1e-9


@dataclass(frozen=True)
class BridgeSettings:
    """How the bridge paces the rig and decides that a move ended.

    A chunk lasts at least ``round_s``; the rig asks for the next one when it ends. A move ends when the robot comes
    within the tolerances of its target, when its joints and grippers stay still for ``still_rounds`` chunks short
    of it, or ``call_timeout_s`` after its path played. The bridge counts time in the chunks it answered, so a slow
    simulator times out as a fast one does. ``think_timeout_s`` is wall time: how long one answer waits for the
    program's next call before it holds the targets for a chunk.

    ``arms`` names the rig's arms as its channels do: ``keys.ONE_ARM`` for a rig whose channels carry no arm name.
    A program names that arm ``primary``. ``aliases`` gives an arm more names. ``embodiment`` is the name each
    state and command of a program carries.
    """

    fps: float = 30.0
    round_s: float = 0.2
    position_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = 0.05
    joint_tolerance_rad: float = 0.02
    grip_tolerance: float = 0.05
    still_joint_rad: float = 0.002
    still_grip: float = 0.005
    still_rounds: int = 3
    call_timeout_s: float = 10.0
    think_timeout_s: float = 5.0
    base_frame: str = 'world'
    arms: Sequence[str | None] = keys.BIMANUAL_ARMS
    aliases: Mapping[str, str] = field(
        default_factory=lambda: {PRIMARY_ARM: keys.RIGHT_ARM, SECONDARY_ARM: keys.LEFT_ARM}
    )
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        for name in ('fps', 'round_s', 'call_timeout_s', 'think_timeout_s'):
            check_finite_positive(getattr(self, name), name)
        if self.still_rounds < 1:
            raise ValueError('still_rounds must be at least 1')
        if not self.arms or len(self.arm_names) != len(self.arms):
            raise ValueError('arms must name at least one arm, and each arm once')

    @property
    def arm_names(self) -> dict[str, str | None]:
        """The name of each arm in a program, with its name on the rig."""
        return {PRIMARY_ARM if arm is None else arm: arm for arm in self.arms}

    def program_arm(self, name: str) -> str | None:
        """The key in ``arm_names`` of the arm a program names ``name``, aliases included; ``None`` for no arm."""
        arm = name if name in self.arm_names else self.aliases.get(name)
        return arm if arm in self.arm_names else None

    @property
    def round_ticks(self) -> int:
        return max(1, math.ceil(self.round_s * self.fps - _TICK_EPSILON))


_Row = tuple[float, float, float]


class CameraCalibration(BaseModel, frozen=True, extra='forbid'):
    """The camera matrix of one camera and the pose of its optical frame, as a calibration file holds them.

    ``pose`` is x, y, z, qw, qx, qy, qz in the base frame. With ``hand`` it is in the hand frame of that arm, as a
    program names the arm. The optical frame has x to the right of the image, y down and z forward.
    """

    intrinsics: tuple[_Row, _Row, _Row]
    pose: tuple[float, float, float, float, float, float, float]
    hand: str | None = None

    @model_validator(mode='after')
    def _validate(self) -> Self:
        if not np.all(np.isfinite(self.matrix)) or not np.all(np.isfinite(self.pose)):
            raise ValueError('intrinsics and pose must be finite')
        if self.intrinsics[2] != (0.0, 0.0, 1.0) or self.intrinsics[0][0] <= 0.0 or self.intrinsics[1][1] <= 0.0:
            raise ValueError('intrinsics must have positive focal lengths and the last row [0, 0, 1]')
        if not math.isclose(np.linalg.norm(self.pose[3:]), 1.0, abs_tol=1e-3):
            raise ValueError('the quaternion of pose must have unit length')
        return self

    @cached_property
    def matrix(self) -> np.ndarray:
        matrix = np.array(self.intrinsics, dtype=np.float64)
        matrix.setflags(write=False)
        return matrix

    @cached_property
    def transform(self) -> geom.Transform3D:
        return geom.Transform3D.from_vector(np.array(self.pose, dtype=np.float64), geom.Rotation.Representation.QUAT)


def camera_hands(calibration: Mapping[str, CameraCalibration], settings: BridgeSettings) -> dict[str, str]:
    """The arm whose hand carries each calibrated camera that has a ``hand``, by camera name."""
    hands = {}
    for name, camera in calibration.items():
        if camera.hand is None:
            continue
        arm = settings.program_arm(camera.hand)
        if arm is None:
            raise ValueError(f'The calibration of camera {name!r} names the hand of no arm: {camera.hand!r}')
        hands[name] = arm
    return hands


class MoveEnd(StrEnum):
    """Why the bridge ended a call."""

    ARRIVED = 'arrived'
    STOPPED = 'stopped'
    TIMEOUT = 'timeout'
    PERIOD = 'period'


# The keys of a call result's ``diagnostics``, which a program reads.
DIAGNOSTIC_ACCEPTED = 'accepted'
DIAGNOSTIC_ARRIVED = 'arrived'
DIAGNOSTIC_MOVE_END = 'move_end'


@dataclass(frozen=True)
class Program:
    """A CapApi program: its name, as the session asked for it, and its source."""

    name: str
    source: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


class ArraySummary(BaseModel):
    """What a call trace holds of an array with two or more dimensions, such as a mask."""

    shape: tuple[int, ...]
    dtype: str
    nonzero: int


class PointCloudSummary(BaseModel):
    """What a call trace holds of a point cloud."""

    frame: str
    point_count: int


class CallTrace(BaseModel):
    """One CapApi call that a program made: its arguments, its result or what it raised, and when it ran.

    ``*_wall_s`` are seconds of wall time since the program started. ``*_robot_s`` are seconds of the bridge's
    clock: the time of the chunks it answered. An observation in ``args``, ``kwargs`` or ``result`` carries no camera
    images. A mask is an ``ArraySummary`` there, and a point cloud a ``PointCloudSummary``.
    """

    name: str
    args: list[JsonValue]
    kwargs: dict[str, JsonValue]
    result: JsonValue = None
    raised: str | None = None
    started_wall_s: float
    ended_wall_s: float
    started_robot_s: float
    ended_robot_s: float


@dataclass(frozen=True)
class ProgramOutcome:
    """How a program ended: ``COMPLETE`` when it ran to its end, ``GIVE_UP`` with the traceback when it raised."""

    end: PolicyEnd
    error: str | None = None


class ProgramStopped(BaseException):
    """Unwinds the program thread when its session closes; a program's ``except Exception`` does not catch it."""


@dataclass(frozen=True)
class _ArmTarget:
    command: CartesianPosition | JointPosition
    # Positronic closure: 0 open, 1 closed.
    grip: float


@dataclass(frozen=True)
class _ChunkPlan:
    """One answer, decided under the lock and built after it: the path, then `holds` copies of `hold`."""

    path: Sequence[Mapping[str, _ArmTarget]]
    hold: Mapping[str, _ArmTarget]
    holds: int
    end: PolicyEnd | None

    def chunk(self, arm_names: Mapping[str, str | None]) -> list[dict[str, Any]]:
        """The answer on the rig's channels; ``arm_names`` gives each arm's name on the rig."""

        def commands(targets: Mapping[str, _ArmTarget]) -> dict[str, Any]:
            entry: dict[str, Any] = {}
            for arm, target in targets.items():
                entry[keys.arm_channel(keys.ROBOT_COMMAND, arm_names[arm])] = target.command
                entry[keys.arm_channel(keys.TARGET_GRIP, arm_names[arm])] = target.grip
            return entry

        chunk = [commands(targets) for targets in self.path] + [commands(self.hold)] * self.holds
        if self.end is not None:
            chunk[-1] = {**chunk[-1], END: self.end.value}
        return chunk


@dataclass(frozen=True)
class _Goal:
    """What a call waits for on one arm: its command, its grip, or both."""

    command: CartesianPosition | JointPosition | None
    grip: float | None = None


@dataclass
class _Call:
    path: list[dict[str, _ArmTarget]]
    goals: dict[str, _Goal]
    one_period: bool = False
    deadline_s: float | None = None
    ticks: int = 0
    still: int = 0
    end: MoveEnd | None = None
    final: Observation | None = None


@dataclass
class _Errors:
    """How far one arm is from its goal, in the units of each part the goal names."""

    position_m: float | None = None
    rotation_rad: float | None = None
    joint_rad: float | None = None
    grip: float | None = None

    def within(self, settings: BridgeSettings) -> bool:
        bounds = (
            (self.position_m, settings.position_tolerance_m),
            (self.rotation_rad, settings.rotation_tolerance_rad),
            (self.joint_rad, settings.joint_tolerance_rad),
            (self.grip, settings.grip_tolerance),
        )
        return all(error <= bound for error, bound in bounds if error is not None)

    def within_call(self, tolerance: float) -> bool:
        """ASPIRE's verdict on the call's ``tolerance``: metres for a pose, radians for joints."""
        bounds = (
            (self.position_m, tolerance),
            (self.rotation_rad, POSE_ORIENTATION_TOLERANCE_RAD),
            (self.joint_rad, tolerance),
        )
        return all(error <= bound for error, bound in bounds if error is not None)

    def headline(self) -> float:
        """The error ASPIRE reports in ``final_errors``: position, else joints, else grip."""
        return next(error for error in (self.position_m, self.joint_rad, self.grip) if error is not None)


def _refusal_is_a_result(method: Callable[..., ExecutionResult]) -> Callable[..., ExecutionResult]:
    """Answer an ``ApiError`` that ``method`` raises before it posts its call with a result that was not accepted."""

    @wraps(method)
    def call(*args: Any, **kwargs: Any) -> ExecutionResult:
        try:
            return method(*args, **kwargs)
        except ApiError as refusal:
            return ExecutionResult(ok=False, steps_executed=0, error=refusal, diagnostics={DIAGNOSTIC_ACCEPTED: False})

    return call


def _motion_options(tolerance: float, max_steps: int) -> None:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ApiError(ErrorCode.INVALID_REQUEST, 'tolerance must be finite and positive')
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ApiError(ErrorCode.INVALID_REQUEST, 'max_steps must be a positive integer')


def pose_from_mapping(value: Mapping[str, Any]) -> Pose:
    """ASPIRE's pose mapping: ``position`` and ``frame``, with exactly one of ``quaternion_wxyz`` or fixed-axis XYZ
    ``rpy_deg``."""
    if set(value) - {'position', 'quaternion_wxyz', 'rpy_deg', 'frame'}:
        raise ValueError('unknown pose target field')
    if not {'position', 'frame'} <= set(value):
        raise ValueError('pose target requires position and frame')
    if ('rpy_deg' in value) == ('quaternion_wxyz' in value):
        raise ValueError('specify exactly one of rpy_deg or quaternion_wxyz')
    if 'rpy_deg' in value:
        quaternion = geom.Rotation.from_euler(np.radians(np.asarray(value['rpy_deg'], dtype=np.float64))).as_quat
    else:
        quaternion = value['quaternion_wxyz']
    return Pose(position=value['position'], quaternion_wxyz=quaternion, frame=value['frame'])


class CapBridge:
    """An ASPIRE ``EnvironmentAdapter`` and the CapApi call surface of one program run, fed by ``answer``.

    ``answer`` takes one positronic observation and returns the next action chunk. The first observation starts the
    program and fixes each arm's home at its measured joints. Every later answer carries ``END`` once the program
    ended. ``calibration`` gives a camera view its intrinsics and its pose, by camera name. ``segmentation`` answers
    ``segment_text``; with no provider, ``segment_text`` raises ``NotImplementedError``, and so does each other
    perception call and each planning call.
    """

    def __init__(
        self,
        program: Program,
        settings: BridgeSettings,
        *,
        calibration: Mapping[str, CameraCalibration] = MappingProxyType({}),
        segmentation: SegmentationProvider | None = None,
    ):
        self._program = program
        self._settings = settings
        self._arms = settings.arm_names
        self._calibration = calibration
        self._hands = camera_hands(calibration, settings)
        self._segmentation = segmentation
        self._calls: list[CallTrace] = []
        self._started_wall_s = 0.0
        self._turn = threading.Condition()
        self._thread: threading.Thread | None = None
        self._observation: Observation | None = None
        self._clock_s = 0.0
        self._targets: dict[str, _ArmTarget] = {}
        self._home: dict[str, np.ndarray] = {}
        self._call: _Call | None = None
        self._outcome: ProgramOutcome | None = None
        self._closed = False

    @property
    def embodiment(self) -> str:
        return self._settings.embodiment

    @property
    def outcome(self) -> ProgramOutcome | None:
        return self._outcome

    @property
    def program(self) -> Program:
        return self._program

    @property
    def calls(self) -> list[CallTrace]:
        """The CapApi calls the program made so far, in order."""
        with self._turn:
            return list(self._calls)

    def answer(self, obs: Obs) -> list[dict[str, Any]]:
        observation = self._observe(obs)
        # Each step under the lock is bounded by the arm count; `wait_for` releases it.
        with self._turn:
            previous, self._observation = self._observation, observation
            if self._thread is None:
                self._start(observation.robot_state)
            elif self._call is not None and self._call.deadline_s is not None and self._call.end is None:
                assert previous is not None
                self._judge(self._call, previous.robot_state, observation)
            self._turn.notify_all()
            self._turn.wait_for(self._program_is_waiting, timeout=self._settings.think_timeout_s)
            plan = self._advance()
        return plan.chunk(self._arms)

    def close(self) -> None:
        with self._turn:
            self._closed = True
            self._turn.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._settings.think_timeout_s)
            if self._thread.is_alive():
                logger.error('The program thread did not stop within %s s of the close', self._settings.think_timeout_s)

    def reset(self, task_ref: object, seed: int) -> Observation:
        raise NotImplementedError('The positronic harness owns resets and episode boundaries')

    # ---- The rig side: observations in, chunks out ----

    def _observe(self, obs: Obs) -> Observation:
        joints, velocities, hands, grippers = {}, {}, {}, {}
        for arm, rig_arm in self._arms.items():
            state = keys.arm_channel(keys.ROBOT_STATE, rig_arm)
            joints[arm] = obs[f'{state}{keys.JOINTS_SUFFIX}']
            velocities[arm] = obs[f'{state}{keys.JOINT_VEL_SUFFIX}']
            hands[arm] = geom.Transform3D.from_vector(
                np.asarray(obs[f'{state}{keys.EE_POSE_SUFFIX}'], dtype=np.float64), geom.Rotation.Representation.QUAT
            )
            # ASPIRE's gripper runs from 0 (closed) to 1 (open); positronic's grip from 0 (open) to 1 (closed).
            grippers[arm] = float(np.clip(1.0 - float(obs[keys.arm_channel(keys.GRIP, rig_arm)]), 0.0, 1.0))
        poses = {arm: self._pose(hand) for arm, hand in hands.items()}
        state = RobotState(
            joints, velocities, poses, grippers, self._settings.base_frame, embodiment=self._settings.embodiment
        )
        names = [key.removeprefix(keys.IMAGE_PREFIX) for key in obs if key.startswith(keys.IMAGE_PREFIX)]
        cameras = {name: self._camera(name, obs, hands) for name in names}
        language = str(obs[keys.TASK])
        task = TaskContext(TASK_SUITE, 0, language, language, TASK_FAMILY)
        return Observation(cameras, state, task)

    def _pose(self, transform: geom.Transform3D) -> Pose:
        return Pose(transform.translation, transform.rotation.as_quat, self._settings.base_frame)

    def _camera(self, name: str, obs: Obs, hands: Mapping[str, geom.Transform3D]) -> CameraObservation:
        """The view of camera ``name``; ``hands`` holds the measured hand pose of each arm."""
        image = np.asarray(obs[f'{keys.IMAGE_PREFIX}{name}'])
        depth = obs.get(f'{keys.DEPTH_PREFIX}{name}')
        depth_m = None if depth is None else self._depth_m(depth)
        calibration = self._calibration.get(name)
        if calibration is None:
            return CameraObservation(image, name, depth_m)
        hand = self._hands.get(name)
        mount = geom.Transform3D.identity if hand is None else hands[hand]
        return CameraObservation(image, name, depth_m, calibration.matrix, self._pose(mount * calibration.transform))

    @staticmethod
    def _depth_m(depth: Any) -> np.ndarray:
        """Depth in metres from ``uint16`` millimetres or floating metres; 0 where the camera measured none."""
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f'Depth must hold uint16 millimetres or floating metres, got {depth.dtype}')
        return np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)

    def _start(self, state: RobotState) -> None:
        for arm in self._arms:
            self._home[arm] = state.joint_positions[arm]
            self._targets[arm] = _ArmTarget(
                JointPosition(state.joint_positions[arm]), 1.0 - state.gripper_positions[arm]
            )
        self._thread = threading.Thread(target=self._run_program, name='aspire-program', daemon=True)
        self._thread.start()

    def _program_is_waiting(self) -> bool:
        return self._outcome is not None or (self._call is not None and self._call.end is None)

    def _advance(self) -> _ChunkPlan:
        if self._outcome is not None:
            return _ChunkPlan([], self._targets, 1, self._outcome.end)
        call = self._call
        sending = call is not None and call.end is None and call.deadline_s is None
        path = call.path if sending else []
        ticks = max(len(path), self._settings.round_ticks)
        if path:
            self._targets = path[-1]
        self._clock_s += ticks / self._settings.fps
        if call is not None and call.end is None:
            call.ticks += ticks
            if sending:
                call.deadline_s = self._clock_s + self._settings.call_timeout_s
        return _ChunkPlan(path, self._targets, ticks - len(path), None)

    def _judge(self, call: _Call, previous: RobotState, observation: Observation) -> None:
        settings, state = self._settings, observation.robot_state
        if call.one_period:
            end = MoveEnd.PERIOD
        elif all(self._goal_errors(goal, state, arm).within(settings) for arm, goal in call.goals.items()):
            end = MoveEnd.ARRIVED
        else:
            still = all(
                np.max(np.abs(state.joint_positions[arm] - previous.joint_positions[arm])) < settings.still_joint_rad
                and abs(state.gripper_positions[arm] - previous.gripper_positions[arm]) < settings.still_grip
                for arm in call.goals
            )
            call.still = call.still + 1 if still else 0
            assert call.deadline_s is not None
            if call.still >= settings.still_rounds:
                end = MoveEnd.STOPPED
            elif self._clock_s >= call.deadline_s:
                end = MoveEnd.TIMEOUT
            else:
                return
        call.end, call.final = end, observation

    @staticmethod
    def _goal_errors(goal: _Goal, state: RobotState, arm: str) -> _Errors:
        errors = _Errors()
        match goal.command:
            case CartesianPosition(pose=target):
                measured = state.end_effector_poses[arm]
                errors.position_m = float(np.linalg.norm(measured.position - target.translation))
                rotation = geom.Rotation.from_quat(measured.quaternion_wxyz)
                errors.rotation_rad = (rotation.inv * target.rotation).angle
            case JointPosition(positions=target):
                errors.joint_rad = float(np.max(np.abs(state.joint_positions[arm] - target)))
        if goal.grip is not None:
            errors.grip = abs((1.0 - state.gripper_positions[arm]) - goal.grip)
        return errors

    # ---- The program side ----

    @staticmethod
    def _import_math(name: str, _globals=None, _locals=None, _fromlist=(), level=0):
        if level != 0 or name != 'math':
            raise ImportError(f'import of {name!r} is not allowed in generated programs')
        return importlib.import_module(name)

    def _run_program(self) -> None:
        # ASPIRE's program builtins and constructors. They keep a program to its contract; they are not a sandbox.
        builtins: dict[str, Any] = {
            builtin.__name__: builtin
            for builtin in (Exception, ValueError, TypeError, abs, all, any, bool, dict, enumerate, float, int, len)
            + (list, max, min, print, range, reversed, round, set, sorted, str, sum, tuple, zip)
        }
        builtins['__import__'] = self._import_math
        constructors: dict[str, Any] = {t.__name__: t for t in (MotionStrategy, Pose, RobotAction)}
        for of_embodiment in (ArmCommand, SynchronizedTrajectory, Trajectory):
            constructors[of_embodiment.__name__] = partial(of_embodiment, embodiment=self.embodiment)
        tools = {name: self._traced(name, tool) for name, tool in self._tools().items()}
        namespace: dict[str, Any] = {'__builtins__': builtins, **constructors, **tools}
        self._started_wall_s = time.monotonic()
        try:
            exec(compile(self._program.source, self._program.name, 'exec'), namespace)
            outcome = ProgramOutcome(PolicyEnd.COMPLETE)
        except ProgramStopped:
            return
        except Exception:
            # The episode gives up and records the traceback; the server keeps serving other sessions.
            logger.exception('The program raised')
            outcome = ProgramOutcome(PolicyEnd.GIVE_UP, traceback.format_exc())
        with self._turn:
            self._outcome = outcome
            self._turn.notify_all()

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """The JSON form of what pydantic cannot dump itself: numpy values and read-only mappings."""
        match value:
            case np.ndarray():
                return value.tolist()
            case np.generic():
                return value.item()
            case MappingProxyType():
                return dict(value)
        # A record must not change what the call gives the program, so a value with no JSON form is its repr.
        logger.error('A call trace holds the repr of a value of type %s, which has no JSON form', type(value).__name__)
        return repr(value)

    @staticmethod
    def _trace_form(value: Any) -> Any:
        """``value`` as a call trace holds it: no camera view, and a summary of each mask and point cloud."""
        match value:
            case Observation():
                return replace(value, cameras={})
            case ExecutionResult(final_observation=Observation() as observation):
                return replace(value, final_observation=replace(observation, cameras={}))
            case StepResult(observation=Observation() as observation):
                return replace(value, observation=replace(observation, cameras={}))
            case np.ndarray() if value.ndim >= 2:
                return ArraySummary(shape=value.shape, dtype=str(value.dtype), nonzero=np.count_nonzero(value))
            case PointCloud():
                return PointCloudSummary(frame=value.frame, point_count=len(value.points))
            case _ if is_dataclass(value) and not isinstance(value, type):
                return {item.name: CapBridge._trace_form(getattr(value, item.name)) for item in fields(value)}
            case tuple() | list():
                return [CapBridge._trace_form(item) for item in value]
            case Mapping():
                return {key: CapBridge._trace_form(item) for key, item in value.items()}
        return value

    def _traced(self, name: str, tool: Callable[..., Any]) -> Callable[..., Any]:
        """``tool``, which also records each call the program makes to it."""

        @wraps(tool)
        def call(*args: Any, **kwargs: Any) -> Any:
            started_wall_s, started_robot_s = time.monotonic() - self._started_wall_s, self._robot_clock_s()
            result, raised = None, None
            try:
                result = tool(*args, **kwargs)
                return result
            except Exception as error:
                raised = f'{type(error).__name__}: {error}'
                raise
            finally:
                trace = CallTrace(
                    name=name,
                    args=to_jsonable_python(self._trace_form(args), fallback=self._jsonable),
                    kwargs=to_jsonable_python(self._trace_form(kwargs), fallback=self._jsonable),
                    result=to_jsonable_python(self._trace_form(result), fallback=self._jsonable),
                    raised=raised,
                    started_wall_s=started_wall_s,
                    ended_wall_s=time.monotonic() - self._started_wall_s,
                    started_robot_s=started_robot_s,
                    ended_robot_s=self._robot_clock_s(),
                )
                with self._turn:
                    self._calls.append(trace)

        return call

    def _robot_clock_s(self) -> float:
        with self._turn:
            return self._clock_s

    @staticmethod
    def _unsupported(name: str) -> Callable[..., Any]:
        def call(*args: object, **kwargs: object) -> Any:
            raise NotImplementedError(f'{name} is not available: this bridge does not serve it')

        call.__name__ = name
        return call

    def _tools(self) -> dict[str, Callable[..., Any]]:
        perception = (
            'segment_points',
            'crop_point_cloud',
            'estimate_geometry',
            'generate_grasps',
            'solve_ik',
            'plan_motion',
            'plan_synchronized_motion',
            'localize_object',
            'select_grasp',
        )
        tools: dict[str, Callable[..., Any]] = {name: self._unsupported(name) for name in perception}
        tools.update(
            get_task_context=self.get_task_context,
            get_observation=self.get_observation,
            get_robot_state=self.get_robot_state,
            segment_text=self.segment_text,
            mask_to_point_cloud=self.mask_to_point_cloud,
            step=self.step,
            execute_trajectory=self.execute_trajectory,
            set_gripper=self.set_gripper,
            set_grippers=self.set_grippers,
            move_to_joints=self.move_to_joints,
            move_to_pose=self.move_to_pose,
            move_synchronized=self.move_synchronized,
            open_gripper=self.open_gripper,
            close_gripper=self.close_gripper,
            go_home=self.go_home,
        )
        return tools

    def _latest(self) -> Observation:
        with self._turn:
            if self._closed:
                raise ProgramStopped
            assert self._observation is not None
            return self._observation

    def _await(self, call: _Call) -> _Call:
        with self._turn:
            if self._closed:
                raise ProgramStopped
            self._call = call
            self._turn.notify_all()
            self._turn.wait_for(lambda: call.end is not None or self._closed)
            if call.end is None:
                raise ProgramStopped
            return call

    def _arm(self, name: str) -> str:
        arm = self._settings.program_arm(name)
        if arm is None:
            raise ApiError(ErrorCode.NOT_FOUND, f'unknown arm {name!r}')
        return arm

    def _settle(self, goals: dict[str, _Goal], tolerance: float | None) -> ExecutionResult:
        """Post a move of the arms in ``goals``, wait for it to end, and judge it against ``tolerance``.

        ``tolerance`` is metres for a pose and radians for joints; ``None`` judges the move by its arrival.
        """
        with self._turn:
            targets = dict(self._targets)
        for arm, goal in goals.items():
            held = targets[arm]
            targets[arm] = _ArmTarget(goal.command or held.command, held.grip if goal.grip is None else goal.grip)
        return self._result(self._await(_Call([targets], goals)), tolerance)

    def _result(self, call: _Call, tolerance: float | None) -> ExecutionResult:
        assert call.final is not None and call.end is not None
        state = call.final.robot_state
        arrived = call.end is MoveEnd.ARRIVED
        final_errors, failed = {}, []
        for arm, goal in call.goals.items():
            errors = self._goal_errors(goal, state, arm)
            final_errors[arm] = errors.headline()
            if not (arrived if tolerance is None else errors.within_call(tolerance)):
                failed.append(arm)
        diagnostics = {DIAGNOSTIC_ACCEPTED: True, DIAGNOSTIC_ARRIVED: arrived, DIAGNOSTIC_MOVE_END: call.end.value}
        if not failed:
            error = None
        elif call.end is MoveEnd.TIMEOUT:
            error = ApiError(ErrorCode.TIMEOUT, f'the move did not end within {self._settings.call_timeout_s} s')
        else:
            error = ApiError(
                ErrorCode.CONTROLLER_FAILED, f'the final state of {failed} is outside the target tolerance'
            )
        return ExecutionResult(
            ok=not failed,
            steps_executed=call.ticks,
            final_observation=call.final,
            error=error,
            final_errors=final_errors,
            diagnostics=diagnostics,
        )

    # ---- The CapApi and EnvironmentAdapter calls ----

    def get_task_context(self) -> TaskContext:
        return self._latest().task_context

    def get_observation(self, camera_names: str | list[str] | tuple[str, ...] | None = None) -> Observation:
        observation = self._latest()
        if camera_names is None:
            return observation
        names = (camera_names,) if isinstance(camera_names, str) else tuple(camera_names)
        if not names or any(name not in observation.cameras for name in names):
            raise ValueError('camera_names must name cameras present in the observation')
        return replace(observation, cameras={name: observation.cameras[name] for name in names})

    def get_robot_state(self) -> RobotState:
        return self._latest().robot_state

    def segment_text(self, camera_name: str, text: str) -> SegmentationSet:
        """The segmentation provider's answer for the view of ``camera_name``, each segmentation under that name."""
        if self._segmentation is None:
            raise NotImplementedError('segment_text is not available: this bridge has no segmentation provider')
        view = self._view(self._latest(), camera_name)
        try:
            found = self._segmentation.segment_text(view, text)
        except ApiError as error:
            return SegmentationSet(ok=False, error=error)
        except Exception as error:
            # One provider failure must not end the program: it gets the failure as a result, and the log holds it.
            logger.error('The segmentation provider raised', exc_info=error)
            return SegmentationSet(
                ok=False, error=ApiError(ErrorCode.PROVIDER_ERROR, f'{type(error).__name__}: {error}')
            )
        if not found.ok:
            return found
        # `mask_to_point_cloud` takes a mask only in the frame of the view it came from.
        frames = {item.frame for item in found.segmentations} - {view.frame}
        if frames:
            message = f'the segmentation provider answered in frame {frames.pop()!r}, not {view.frame!r}'
            return SegmentationSet(ok=False, error=ApiError(ErrorCode.PROVIDER_ERROR, message))
        return replace(found, segmentations=[replace(item, camera_name=camera_name) for item in found.segmentations])

    def mask_to_point_cloud(
        self, mask: Segmentation | np.ndarray, camera_name: str, target_frame: str | None = None
    ) -> PointCloud:
        """The points of ``mask`` in ``target_frame``: the base frame when it is ``None``."""
        observation = self._latest()
        view = self._view(observation, camera_name)
        if not isinstance(mask, Segmentation):
            mask = Segmentation(mask=mask, label='object', score=1.0, camera_name=camera_name, frame=view.frame)
        frame = target_frame or observation.robot_state.base_frame
        return geometry.mask_to_point_cloud(view, mask, target_frame=frame)

    @staticmethod
    def _view(observation: Observation, camera_name: str) -> CameraObservation:
        if camera_name not in observation.cameras:
            raise ValueError(f'observation has no camera {camera_name!r}')
        return observation.cameras[camera_name]

    def step(self, action: RobotAction) -> StepResult:
        try:
            if action.embodiment != self.embodiment:
                raise ApiError(
                    ErrorCode.INVALID_REQUEST, f'action embodiment {action.embodiment!r} is not {self.embodiment!r}'
                )
            commands = {self._arm(name): command for name, command in action.arms.items()}
            joints = {arm: self._joint_target(command.target, arm) for arm, command in commands.items()}
        except ApiError as refusal:
            return StepResult(ok=False, error=refusal, diagnostics={DIAGNOSTIC_ACCEPTED: False})
        with self._turn:
            targets = dict(self._targets)
        for arm, command in commands.items():
            grip = targets[arm].grip if command.gripper_position is None else 1.0 - command.gripper_position
            targets[arm] = _ArmTarget(joints[arm], grip)
        call = self._await(_Call([targets], {}, one_period=True))
        assert call.end is not None
        diagnostics = {DIAGNOSTIC_ACCEPTED: True, DIAGNOSTIC_MOVE_END: call.end.value}
        return StepResult(ok=True, observation=call.final, diagnostics=diagnostics)

    @_refusal_is_a_result
    def execute_trajectory(self, trajectory: Trajectory | SynchronizedTrajectory) -> ExecutionResult:
        if trajectory.embodiment != self.embodiment:
            raise ApiError(
                ErrorCode.INVALID_REQUEST, f'trajectory embodiment {trajectory.embodiment!r} is not {self.embodiment!r}'
            )
        if isinstance(trajectory, Trajectory):
            arm = self._arm(trajectory.arm)
            waypoints = {arm: trajectory.joint_positions}
            grippers = None if trajectory.gripper_positions is None else {arm: trajectory.gripper_positions}
        else:
            waypoints = {self._arm(name): joints for name, joints in trajectory.joint_positions.items()}
            grippers = None
            if trajectory.gripper_positions is not None:
                grippers = {self._arm(name): values for name, values in trajectory.gripper_positions.items()}
        goals = {arm: _Goal(self._joint_target(joints[-1], arm)) for arm, joints in waypoints.items()}
        measured = self.get_robot_state().joint_positions
        start = trajectory.expected_start.joint_positions
        for arm in waypoints:
            planned = start.get(arm)
            if planned is None or np.max(np.abs(measured[arm] - planned)) > TRAJECTORY_START_TOLERANCE_RAD:
                raise ApiError(ErrorCode.STALE_PLAN, f'the trajectory of {arm!r} does not start at the measured joints')
        count = len(next(iter(waypoints.values())))
        ticks = max(1, round(count * trajectory.dt_s * self._settings.fps))
        indices = [min(count - 1, int(tick * count / ticks)) for tick in range(ticks - 1)] + [count - 1]
        with self._turn:
            held = dict(self._targets)
        path = []
        for index in indices:
            targets = dict(held)
            for arm, joints in waypoints.items():
                grip = held[arm].grip if grippers is None else 1.0 - float(grippers[arm][index])
                targets[arm] = _ArmTarget(JointPosition(joints[index]), grip)
            path.append(targets)
        return self._result(self._await(_Call(path, goals)), self._settings.joint_tolerance_rad)

    @_refusal_is_a_result
    def set_gripper(self, position: float, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        return self._settle(self._grip_goals({arm: position}), None)

    @_refusal_is_a_result
    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        goals = self._grip_goals(positions)
        if set(goals) != set(self._arms):
            raise ApiError(ErrorCode.INVALID_REQUEST, 'positions must name every robot arm')
        return self._settle(goals, None)

    def _grip_goals(self, positions: Mapping[str, float]) -> dict[str, _Goal]:
        goals = {}
        for name, position in positions.items():
            try:
                value = float(position)
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ApiError(ErrorCode.INVALID_REQUEST, f'gripper position for {name!r} must be finite and in [0, 1]')
            goals[self._arm(name)] = _Goal(None, 1.0 - value)
        return goals

    @_refusal_is_a_result
    def move_to_joints(
        self, target: np.ndarray, tolerance: float = 0.01, max_steps: int = 120, *, arm: str = PRIMARY_ARM
    ) -> ExecutionResult:
        _motion_options(tolerance, max_steps)
        arm = self._arm(arm)
        return self._settle({arm: _Goal(self._joint_target(target, arm))}, tolerance)

    @_refusal_is_a_result
    def move_to_pose(
        self,
        target_pose: Pose | Mapping[str, Any],
        tolerance: float = 0.01,
        max_steps: int = 120,
        *,
        arm: str = PRIMARY_ARM,
        strategy: MotionStrategy | None = None,
    ) -> ExecutionResult:
        _motion_options(tolerance, max_steps)
        return self._settle({self._arm(arm): _Goal(self._pose_target(target_pose))}, tolerance)

    @_refusal_is_a_result
    def move_synchronized(
        self,
        targets: Mapping[str, Pose | np.ndarray],
        tolerance: float = 0.01,
        max_steps: int = 120,
        strategy: MotionStrategy | None = None,
    ) -> ExecutionResult:
        _motion_options(tolerance, max_steps)
        if not targets:
            raise ApiError(ErrorCode.INVALID_REQUEST, 'targets must name at least one arm')
        goals = {}
        for name, target in targets.items():
            arm = self._arm(name)
            pose = isinstance(target, Pose | Mapping)
            goals[arm] = _Goal(self._pose_target(target) if pose else self._joint_target(target, arm))
        return self._settle(goals, tolerance)

    def open_gripper(self, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        return self.set_gripper(1.0, arm=arm)

    def close_gripper(self, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        return self.set_gripper(0.0, arm=arm)

    @_refusal_is_a_result
    def go_home(self, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        """Move ``arm`` to the joints it had at the first observation."""
        return self.move_to_joints(self._home[self._arm(arm)], arm=arm)

    def _pose_target(self, target: Pose | Mapping[str, Any]) -> CartesianPosition:
        try:
            pose = pose_from_mapping(target) if isinstance(target, Mapping) else target
        except (TypeError, ValueError) as exc:
            raise ApiError(ErrorCode.INVALID_REQUEST, str(exc)) from exc
        if not isinstance(pose, Pose):
            raise ApiError(ErrorCode.INVALID_REQUEST, 'target_pose must be a Pose or pose mapping')
        if pose.frame != self._settings.base_frame:
            raise ApiError(ErrorCode.INVALID_REQUEST, f'pose frame {pose.frame!r} is not {self._settings.base_frame!r}')
        return CartesianPosition(geom.Transform3D(pose.position, geom.Rotation.from_quat(pose.quaternion_wxyz)))

    def _joint_target(self, target: object, arm: str) -> JointPosition:
        try:
            joints = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ApiError(ErrorCode.INVALID_REQUEST, f'joint target must be a finite joint vector: {exc}') from exc
        # The measured joints of the arm give its joint count.
        count = len(self._home[arm])
        if joints.shape != (count,) or not np.all(np.isfinite(joints)):
            raise ApiError(ErrorCode.INVALID_REQUEST, f'joint target must be a finite {count}-joint vector')
        return JointPosition(joints)
