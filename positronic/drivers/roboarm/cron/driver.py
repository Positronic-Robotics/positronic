import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State, command
from positronic.drivers.roboarm.cron.controller import JOINTS, CronController, PowerState
from positronic.drivers.utils import DriverRun, MoveStatus, log_failure
from positronic.eval import NO_MODEL_ROBOT_META

logger = logging.getLogger(__name__)

_HZ = 200
# At relative_dynamics_factor 1.0. TODO: read the limits of the arm from the controller
_MAX_JOINT_SPEED = 1.0  # rad/s
_MAX_JOINT_ACCEL = 4.0  # rad/s^2
# The joints of the wrist (4 to 6) carry only the tool, and a turn of the flange is slow at the speed of the shoulder
_WRIST_SCALE = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
_GAIN = 8.0  # 1/s
# A loop that runs late must not step the velocity by more than a few ticks
_MAX_STEP_S = 5.0 / _HZ
# Radians; the arm reports joints but no goal, so arrival is judged from the joints it reads
_ARRIVED_TOL = 0.01
# On top of the travel itself: the arm slows down when it nears the target
_MOVE_GRACE_S = 3.0
# The planner of the controller travels at about 0.75 of the speed it is asked for (2026-11-05: 164 deg in 35 s at
# 0.1 rad/s), so the time a move gets is its travel at the asked speed, times this
_MOVE_SLACK = 1.5
# A streamed Cartesian target is followed by damped least squares steps on each tick, as on the Franka: the damping
# grows as the smallest singular value of the jacobian falls under _DLS_SINGULAR, so near a pose where joints turn with
# no limit the arm slows down and follows less exactly, and does not stop
_DLS_ITERATIONS = 3
_DLS_DAMPING = 0.05
_DLS_SINGULAR = 0.08
# Radians: the largest turn of a joint in one step of the solver
_DLS_MAX_STEP = 0.05
# The elbow keeps its side and this bend, as a joint limit of the solver: through a straight elbow the arm turns over
_ELBOW, _MIN_ELBOW = 2, np.radians(5.0)
# At the edge of the reach the steps move the tool sideways and turn it, and the tool drifts from where the operator
# points (0.5 m and 60 degrees on Hand B). Such a step takes the tool further from the target along an axis or in its
# turn, and brings it nearer by less than _MIN_PROGRESS of how far it moves it; it is not taken. A step that does only
# one of the two is taken: on a curved way a straight step leaves the way a little, and where a part of the target is
# out of reach a step that serves the rest brings the tool nearer by little
_MIN_PROGRESS = 0.5
_DRIFT_M, _DRIFT_RAD = 1e-4, 1e-3
# Metres: a step that takes the arm nearer than this to itself, by the collision model of the controller, is not taken
_MIN_CLEARANCE = 0.01


def _pose_error(target: geom.Transform3D, pose: geom.Transform3D) -> float:
    """How far ``pose`` is from ``target``: metres of the tool point plus radians of the turn."""
    turn = float(np.linalg.norm((target.rotation * pose.rotation.inv).as_rotvec))
    return float(np.linalg.norm(np.asarray(target.translation) - np.asarray(pose.translation))) + turn


def _drifts(target: geom.Transform3D, before: geom.Transform3D, after: geom.Transform3D) -> bool:
    """Whether the step from ``before`` to ``after`` takes the tool further from ``target`` along an axis or in its
    turn, and brings it nearer by less than ``_MIN_PROGRESS`` of how far it moves it."""
    shift_before = np.abs(np.asarray(target.translation) - np.asarray(before.translation))
    shift_after = np.abs(np.asarray(target.translation) - np.asarray(after.translation))
    turn_before = float(np.linalg.norm((target.rotation * before.rotation.inv).as_rotvec))
    turn_after = float(np.linalg.norm((target.rotation * after.rotation.inv).as_rotvec))
    away = np.any(shift_after - shift_before > _DRIFT_M) or turn_after - turn_before > _DRIFT_RAD
    gain = _pose_error(target, before) - _pose_error(target, after)
    return bool(away and gain < _MIN_PROGRESS * _pose_error(after, before))


def _finite(cmd: command.CommandType, joints: np.ndarray) -> np.ndarray:
    # The controller refuses a velocity that is not finite and stops the arm with a TIMEOUT error
    if not np.all(np.isfinite(joints)):
        raise ValueError(f'{cmd} asks for joints that are not finite: {joints}')
    return joints


class CronState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(JOINTS + JOINTS + 7 + 1,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:JOINTS]

    @property
    def dq(self) -> np.ndarray:
        return self.array[JOINTS : 2 * JOINTS]

    @property
    def ee_pose(self) -> geom.Transform3D:
        start = 2 * JOINTS
        quat = self.array[start + 3 : start + 7]
        return geom.Transform3D(translation=self.array[start : start + 3], rotation=geom.Rotation.from_quat(quat))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[2 * JOINTS + 7]))

    def encode(self, q, dq, ee_pose, status: RobotStatus):
        start = 2 * JOINTS
        self.array[:JOINTS] = q
        self.array[JOINTS:start] = dq
        self.array[start : start + 3] = ee_pose.translation
        self.array[start + 3 : start + 7] = ee_pose.rotation.as_quat
        self.array[start + 7] = status.value


class _Arm(DriverRun[command.CommandType]):
    """The arm the driver drives. A streamed setpoint is tracked by velocity commands that the speed and
    acceleration limits shape. A move that a caller waits for is planned by the controller.

    The arm holds the joints it has when no streamed target is set.
    """

    def __init__(
        self,
        controller: CronController,
        sync_move: pimm.calls.ControlSystemHandler[command.CommandType, None],
        async_move: pimm.SignalReceiver[command.CommandType],
        out: pimm.SignalEmitter[CronState],
        dynamics_factor: float,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=_HZ)
        self.controller = controller
        self.out = out
        self.state = CronState()
        self.max_speed = dynamics_factor * _MAX_JOINT_SPEED * _WRIST_SCALE
        self.max_accel = dynamics_factor * _MAX_JOINT_ACCEL * _WRIST_SCALE
        self.reading = controller.read()
        self._condition = (self.reading.power, self.reading.in_error)
        self.target: np.ndarray | None = None
        # The tool pose of a streamed Cartesian target, which the damped steps of each tick follow
        self._pose: geom.Transform3D | None = None
        # A move asked for while the controller holds an error: it starts once the error clears
        self._deferred: np.ndarray | None = None
        self.velocity = np.zeros(JOINTS)
        self._last_step = clock.now()

    @property
    def usable(self) -> bool:
        return self.reading.power is PowerState.RUN and not self.reading.in_error

    def settle(self) -> None:
        """Judge a move in flight against the joints the arm reads."""
        if not self.moves.active:
            return
        if self.moves.settle(self.reading.q, self.clock.now()) is MoveStatus.GAVE_UP:
            # A move left in place would resume once whatever blocked it goes away
            self._deferred = None
            self.controller.stop_move()

    def _follow(self, pose: geom.Transform3D, seed: np.ndarray, max_turn: np.ndarray) -> np.ndarray:
        """The joints a few damped least squares steps from ``seed`` toward the tool at ``pose``, each joint at most
        ``max_turn`` from ``seed``, with the elbow on its side, and not nearer the arm itself than the clearance."""
        joints = np.array(seed, dtype=np.float64)
        side = 1.0 if np.sin(seed[_ELBOW]) >= 0 else -1.0
        # An elbow already straighter than the limit does not get straighter, and is not pushed out either
        least = min(_MIN_ELBOW, side * float(np.remainder(seed[_ELBOW] + np.pi, 2 * np.pi) - np.pi))
        for _ in range(_DLS_ITERATIONS):
            now = self.controller.forward(joints)
            error = np.concatenate([pose.translation - now.translation, (pose.rotation * now.rotation.inv).as_rotvec])
            jacobian = self.controller.jacobian(joints)
            smallest = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            damping = _DLS_DAMPING**2 * max(0.0, 1.0 - (smallest / _DLS_SINGULAR) ** 2)
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            joints += step * min(1.0, _DLS_MAX_STEP / max(float(np.max(np.abs(step))), 1e-12))
            elbow = float(np.remainder(joints[_ELBOW] + np.pi, 2 * np.pi) - np.pi)
            if side * elbow < least:
                joints[_ELBOW] += side * least - elbow
        turn = joints - seed
        joints = seed + turn * min(1.0, float(np.min(max_turn / np.maximum(np.abs(turn), 1e-12))))
        if _drifts(pose, self.controller.forward(seed), self.controller.forward(joints)):
            return np.array(seed, dtype=np.float64)
        clearance = self.controller.clearance(joints)
        if clearance < _MIN_CLEARANCE and clearance < self.controller.clearance(seed):
            return np.array(seed, dtype=np.float64)
        return joints

    def _pose_of(self, cmd: command.CommandType, seed: np.ndarray) -> geom.Transform3D | None:
        """The tool pose that a Cartesian ``cmd`` asks for from the joints ``seed``, or None for a joint target."""
        # TODO: PositionControl and Impedance need a controller that takes them; the velocity loop runs its own law
        command.require_native_mode(cmd, 'CRON')
        match cmd:
            case command.CartesianPosition(pose):
                return pose
            case command.CartesianDelta() as delta_cmd:
                return delta_cmd.apply(self.controller.forward(seed))
            case command.JointPosition():
                return None
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def _to_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints that a move ``cmd`` goes to: a Cartesian target through the exact solution of the controller."""
        pose = self._pose_of(cmd, self.reading.q)
        if pose is not None:
            return self.controller.inverse(pose, self.reading.q)
        assert isinstance(cmd, command.JointPosition)
        return _finite(cmd, np.asarray(cmd.positions, dtype=np.float64))

    def track(self, cmd: command.CommandType) -> None:
        """Put the loop on the target ``cmd`` asks for, with nobody waiting on the arrival. A Cartesian target is the
        goal of the damped steps of each tick: the joints move toward it no faster than the arm can follow."""
        seed = self.reading.q if self.target is None else self.target
        pose = self._pose_of(cmd, seed)
        joints = np.array(seed, dtype=np.float64) if pose is not None else self._to_joints(cmd)
        if self.target is None:
            self.controller.to_hold()
        self.target, self._pose = _finite(cmd, joints), pose

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Put the loop on the target ``call`` asks for; ``settle`` answers it once the arm reads back there."""
        with pimm.calls.raise_to(call):
            target = self._to_joints(call.request)
            if self.target is not None:
                self.target, self._pose = None, None
                self.velocity[:] = 0.0
                self.controller.stop()
            if self.reading.in_error:
                # The controller refuses a move while it holds an error
                self.controller.resume()
                self._deferred = target
            else:
                self.controller.move_to(target, float(self.max_speed.min()), float(self.max_accel.min()))
            travel_s = _MOVE_GRACE_S + _MOVE_SLACK * float(
                np.max(np.abs(target - self.reading.q)) / self.max_speed.min()
            )
            self.moves.accept(call, target, _ARRIVED_TOL, self.clock.now(), travel_s)

    def step(self) -> None:
        """Read the arm, and command the velocity that takes it towards the target."""
        self.reading = self.controller.read()
        self._report()
        now = self.clock.now()
        dt = min(now - self._last_step, _MAX_STEP_S)
        self._last_step = now
        if not self.usable:
            # An arm that left RUN holds no target: it must not resume one when it comes back
            self.target, self._pose = None, None
            self.velocity[:] = 0.0
            return
        if self._deferred is not None:
            self.controller.move_to(self._deferred, float(self.max_speed.min()), float(self.max_accel.min()))
            self._deferred = None
        if self.target is None:
            return
        if self._pose is not None:
            self.target = self._follow(self._pose, self.target, self.max_speed * dt)
        error = self.target - self.reading.q
        # Not faster than the arm can stop from in the error left: at the gain alone, braking would need more
        # than max_accel and the arm would overshoot
        limit_speed = np.minimum(np.sqrt(2.0 * self.max_accel * np.abs(error)), self.max_speed)
        wanted = np.clip(_GAIN * error, -limit_speed, limit_speed)
        limit = self.max_accel * dt
        self.velocity += np.clip(wanted - self.velocity, -limit, limit)
        self.controller.command_velocity(self.velocity)

    def _report(self) -> None:
        """Log each change of the power state or the error state of the arm."""
        condition = (self.reading.power, self.reading.in_error)
        if condition == self._condition:
            return
        self._condition = condition
        if self.reading.in_error:
            joints = np.round(np.degrees(self.reading.q), 1).tolist()
            errors = self.controller.errors
            logger.error(f'{self.controller.ip}: {self.reading.power.name}, error {errors}, joints {joints} deg')
        else:
            logger.info(f'{self.controller.ip}: {self.reading.power.name}, no error')

    def publish(self) -> None:
        """Ship the arm as it reads now."""
        if self.reading.in_error or self.reading.power is PowerState.FAILURE or self.moves.errored:
            status = RobotStatus.ERROR
        elif self.moves.active:  # the driver owns the arm until the move settles, and reads no command meanwhile
            status = RobotStatus.BUSY
        else:
            status = RobotStatus.AVAILABLE
        self.state.encode(self.reading.q, self.reading.dq, self.reading.tcp_pose, status)
        self.out.emit(self.state)

    def __enter__(self) -> '_Arm':
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb) -> None:
        """A stop halts the loop, and the arm only travels while the loop commands it; answer what was waiting."""
        self.moves.abandon(exc)


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        relative_dynamics_factor: float = 0.2,
        power_off_on_exit: bool = True,
        tool_length: float = 0.0,
        projects: Sequence[str] = (),
        tool_capsules: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.power_off_on_exit = power_off_on_exit
        self.tool_length = tool_length
        self.projects = tuple(projects)
        self.tool_capsules = tuple(tool_capsules)
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[CronState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def _arm(self, controller: CronController, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Arm:
        """The arm this run drives, built from the driver's configuration."""
        return _Arm(
            controller, self.sync_move, self.commands, self.state, self.relative_dynamics_factor, should_stop, clock
        )

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        controller = CronController(
            self.ip, self.power_off_on_exit, self.tool_length, self.projects, self.tool_capsules
        )
        with controller:
            with self._arm(controller, should_stop, clock) as arm:
                self.robot_meta.emit(NO_MODEL_ROBOT_META)
                while not should_stop.value:
                    arm.settle()
                    asked = arm.moves.next_request()
                    if isinstance(asked, pimm.calls.Call):
                        arm.sync_move(asked)
                    elif asked is not None:
                        with log_failure(asked):
                            arm.track(asked)

                    arm.step()
                    arm.publish()
                    arm.moves.answer()  # the state a settled move is answered with is out

                    yield arm.limiter.wait()
