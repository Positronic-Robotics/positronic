"""Run an ASPIRE CapApi program against a stream of positronic observations.

A call that moves the robot puts its target in a queue and blocks until ``CapBridge.answer`` reads an observation that
ends the move. The rig plays the targets as action chunks: its driver solves IK and limits speed. The vendor README
states how a move ends and what each result field means.
"""

import importlib
import logging
import math
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import wraps
from types import CodeType
from typing import Any

import numpy as np

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition
from positronic.policy.base import Obs, PolicyEnd
from positronic.policy.keys import END
from positronic.utils import check_finite_positive

from .contracts import (
    EMBODIMENT,
    JOINT_DIMENSION,
    PRIMARY_ARM,
    SECONDARY_ARM,
    ApiError,
    ArmCommand,
    CameraObservation,
    ErrorCode,
    ExecutionResult,
    MotionStrategy,
    Observation,
    Pose,
    RobotAction,
    RobotState,
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
    program's next call before it holds the targets for a chunk. ``aliases`` maps ASPIRE's arm roles to arm names.
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
    aliases: Mapping[str, str] = field(
        default_factory=lambda: {PRIMARY_ARM: keys.RIGHT_ARM, SECONDARY_ARM: keys.LEFT_ARM}
    )

    def __post_init__(self) -> None:
        for name in (
            'fps',
            'round_s',
            'call_timeout_s',
            'think_timeout_s',
            'position_tolerance_m',
            'rotation_tolerance_rad',
            'joint_tolerance_rad',
            'grip_tolerance',
            'still_joint_rad',
            'still_grip',
        ):
            check_finite_positive(getattr(self, name), name)
        if self.still_rounds < 1:
            raise ValueError('still_rounds must be at least 1')

    @property
    def round_ticks(self) -> int:
        return max(1, math.ceil(self.round_s * self.fps - _TICK_EPSILON))


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

    def chunk(self) -> list[dict[str, Any]]:
        chunk = [_commands(targets) for targets in self.path] + [_commands(self.hold)] * self.holds
        if self.end is not None:
            chunk[-1] = {**chunk[-1], END: self.end.value}
        return chunk


def _commands(targets: Mapping[str, _ArmTarget]) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for arm, target in targets.items():
        commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = target.command
        commands[keys.arm_channel(keys.TARGET_GRIP, arm)] = target.grip
    return commands


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
    ended. Perception and planning calls raise ``NotImplementedError``.
    """

    embodiment = EMBODIMENT

    def __init__(self, program: CodeType, settings: BridgeSettings):
        self._program = program
        self._settings = settings
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
    def outcome(self) -> ProgramOutcome | None:
        return self._outcome

    def answer(self, obs: Obs) -> list[dict[str, Any]]:
        observation = self._observe(obs)
        if self._thread is None:
            self._start(observation)
        # Each step under the lock is bounded by the arm count; `wait_for` releases it.
        with self._turn:
            previous, self._observation = self._observation, observation
            if self._call is not None and self._call.deadline_s is not None and self._call.end is None:
                assert previous is not None
                self._judge(self._call, previous.robot_state, observation)
            self._turn.notify_all()
            self._turn.wait_for(self._program_is_waiting, timeout=self._settings.think_timeout_s)
            plan = self._advance()
        return plan.chunk()

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
        joints, velocities, poses, grippers = {}, {}, {}, {}
        for arm in keys.BIMANUAL_ARMS:
            state = keys.arm_channel(keys.ROBOT_STATE, arm)
            joints[arm] = obs[f'{state}{keys.JOINTS_SUFFIX}']
            velocities[arm] = obs[f'{state}{keys.JOINT_VEL_SUFFIX}']
            pose = geom.Transform3D.from_vector(
                np.asarray(obs[f'{state}{keys.EE_POSE_SUFFIX}'], dtype=np.float64), geom.Rotation.Representation.QUAT
            )
            poses[arm] = Pose(pose.translation, pose.rotation.as_quat, self._settings.base_frame)
            # ASPIRE's gripper runs from 0 (closed) to 1 (open); positronic's grip from 0 (open) to 1 (closed).
            grippers[arm] = float(np.clip(1.0 - float(obs[keys.arm_channel(keys.GRIP, arm)]), 0.0, 1.0))
        state = RobotState(joints, velocities, poses, grippers, self._settings.base_frame)
        cameras = {
            name.removeprefix(keys.IMAGE_PREFIX): CameraObservation(
                np.asarray(value), name.removeprefix(keys.IMAGE_PREFIX)
            )
            for name, value in obs.items()
            if name.startswith(keys.IMAGE_PREFIX)
        }
        language = str(obs[keys.TASK])
        task = TaskContext(TASK_SUITE, 0, language, language, TASK_FAMILY)
        return Observation(cameras, state, task)

    def _start(self, observation: Observation) -> None:
        # No program thread exists yet, so nothing else reads this state until `start` publishes it.
        self._observation, state = observation, observation.robot_state
        for arm in keys.BIMANUAL_ARMS:
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
        constructors = {
            t.__name__: t for t in (ArmCommand, MotionStrategy, Pose, RobotAction, SynchronizedTrajectory, Trajectory)
        }
        namespace: dict[str, Any] = {'__builtins__': builtins, **constructors, **self._tools()}
        try:
            exec(self._program, namespace)
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
    def _unsupported(name: str) -> Callable[..., Any]:
        def call(*args: object, **kwargs: object) -> Any:
            raise NotImplementedError(f'{name} is not available: this bridge serves no perception or planning')

        call.__name__ = name
        return call

    def _tools(self) -> dict[str, Callable[..., Any]]:
        perception = (
            'segment_text',
            'segment_points',
            'mask_to_point_cloud',
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
        arm = self._settings.aliases.get(name, name)
        if arm not in keys.BIMANUAL_ARMS:
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

    def step(self, action: RobotAction) -> StepResult:
        try:
            if action.embodiment != EMBODIMENT:
                raise ApiError(
                    ErrorCode.INVALID_REQUEST, f'action embodiment {action.embodiment!r} is not {EMBODIMENT!r}'
                )
            commands = {self._arm(name): command for name, command in action.arms.items()}
        except ApiError as refusal:
            return StepResult(ok=False, error=refusal, diagnostics={DIAGNOSTIC_ACCEPTED: False})
        with self._turn:
            targets = dict(self._targets)
        for arm, command in commands.items():
            grip = targets[arm].grip if command.gripper_position is None else 1.0 - command.gripper_position
            targets[arm] = _ArmTarget(JointPosition(command.target), grip)
        call = self._await(_Call([targets], {}, one_period=True))
        return StepResult(ok=True, observation=call.final, diagnostics={DIAGNOSTIC_ACCEPTED: True})

    @_refusal_is_a_result
    def execute_trajectory(self, trajectory: Trajectory | SynchronizedTrajectory) -> ExecutionResult:
        if trajectory.embodiment != EMBODIMENT:
            raise ApiError(
                ErrorCode.INVALID_REQUEST, f'trajectory embodiment {trajectory.embodiment!r} is not {EMBODIMENT!r}'
            )
        if isinstance(trajectory, Trajectory):
            arm = self._arm(trajectory.arm)
            waypoints = {arm: trajectory.joint_positions}
            names = {arm: trajectory.joint_names}
            grippers = None if trajectory.gripper_positions is None else {arm: trajectory.gripper_positions}
        else:
            waypoints = {self._arm(name): joints for name, joints in trajectory.joint_positions.items()}
            names = {self._arm(name): joint_names for name, joint_names in trajectory.joint_names.items()}
            grippers = None
            if trajectory.gripper_positions is not None:
                grippers = {self._arm(name): values for name, values in trajectory.gripper_positions.items()}
        state = self.get_robot_state()
        start = trajectory.expected_start.joint_positions
        for arm in waypoints:
            if arm not in names or tuple(names[arm]) != state.joint_names[arm]:
                raise ApiError(
                    ErrorCode.INVALID_REQUEST, f'the joint names of {arm!r} are not the measured joint names'
                )
            planned = start.get(arm)
            if planned is None or np.max(np.abs(state.joint_positions[arm] - planned)) > TRAJECTORY_START_TOLERANCE_RAD:
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
        goals = {
            arm: _Goal(JointPosition(joints[-1]), None if grippers is None else 1.0 - float(grippers[arm][-1]))
            for arm, joints in waypoints.items()
        }
        return self._result(self._await(_Call(path, goals)), self._settings.joint_tolerance_rad)

    @_refusal_is_a_result
    def set_gripper(self, position: float, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        return self._settle(self._grip_goals({arm: position}), None)

    @_refusal_is_a_result
    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        goals = self._grip_goals(positions)
        if set(goals) != set(keys.BIMANUAL_ARMS):
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
        return self._settle({self._arm(arm): _Goal(self._joint_target(target))}, tolerance)

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
            command = self._pose_target(target) if isinstance(target, Pose | Mapping) else self._joint_target(target)
            goals[self._arm(name)] = _Goal(command)
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

    @staticmethod
    def _joint_target(target: object) -> JointPosition:
        try:
            joints = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ApiError(ErrorCode.INVALID_REQUEST, f'joint target must be a finite joint vector: {exc}') from exc
        if joints.shape != (JOINT_DIMENSION,) or not np.all(np.isfinite(joints)):
            raise ApiError(ErrorCode.INVALID_REQUEST, f'joint target must be a finite {JOINT_DIMENSION}-joint vector')
        return JointPosition(joints)
