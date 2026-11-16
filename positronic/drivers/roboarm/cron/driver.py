import collections
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State, command
from positronic.drivers.roboarm.cron.controller import (
    JOINTS,
    POWER_TIMEOUT_S,
    RESUME_EVERY_S,
    STARTUP_S,
    CronController,
    PowerState,
)
from positronic.drivers.utils import DriverRun, MoveRefused, MoveStatus, log_failure
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
# Degrees: where the controller stops a joint in a stream, with a following error (Hand B, 2026-11-11). The solver keeps
# _LIMIT_MARGIN inside them. TODO: the limits of all joints, from the vendor; these are the ones measured
_JOINT_LIMITS = np.radians([
    [-np.inf, np.inf],
    [-190.8, np.inf],
    [-167.0, 167.0],
    [-np.inf, np.inf],
    [-np.inf, np.inf],
    [-np.inf, np.inf],
])
_LIMIT_MARGIN = np.radians(2.0)
# At the edge of the reach the steps move the tool sideways and turn it, and the tool drifts from where the operator
# points (0.5 m and 60 degrees on Hand B). Such a step takes the tool further from the target along an axis or in its
# turn, and brings it nearer by less than _MIN_PROGRESS of how far it moves it; it is not taken. A step that does only
# one of the two is taken: on a curved way a straight step leaves the way a little, and where a part of the target is
# out of reach a step that serves the rest brings the tool nearer by little
_MIN_PROGRESS = 0.5
_DRIFT_M, _DRIFT_RAD = 1e-4, 1e-3
# Metres: a step that takes the arm nearer than this to itself, by the collision model of the controller, is not taken
_MIN_CLEARANCE = 0.01
# Radians: the spacing of the points where the clearance of a planned move is checked
_PATH_STEP = np.radians(1.0)
# How long a move waits for the controller to clear its error: a FAILURE or a STANDBY needs a power off and a power up,
# each of which can take POWER_TIMEOUT_S, and then the joint move after RUN
_RECOVER_S = 2 * POWER_TIMEOUT_S + STARTUP_S + 4.0
# How many times a planned move starts again after the planner stops on an error. The driver does not know the space
# around the arm: an error that comes back on each start can be a contact, and the move must not drive into it again
_MAX_RESTARTS = 1
# The ticks of a stream that an error report looks back over
_LAG_TICKS = _HZ // 2


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

    @dataclass
    class _Move:
        """A planned move, from its call until the call is answered."""

        target: np.ndarray
        call: pimm.calls.Call[command.CommandType, None]
        # The move fails when it waits past this
        deadline: float
        # The joints when the wait started: recovery can move the arm
        start: np.ndarray
        # A FAILURE is a safety stop. The move clears a FAILURE that it was asked in, and is refused on one that comes
        # after
        in_failure: bool
        restarts: int = 0
        # The move waits to start: until the stream stops, or until the controller clears its error
        waiting: bool = True
        # The largest turn of a joint since the wait started
        moved_rad: float = 0.0
        cleared_at: float | None = None

    @dataclass
    class _PowerUp:
        """The RUN that the driver asked for to clear a FAILURE. It outlives the move that asked for it."""

        asked_at: float | None = None
        run_at: float | None = None

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
        # The joints of a streamed joint target, which the steps of each tick go to
        self._joint_goal: np.ndarray | None = None
        self._move: _Arm._Move | None = None
        self._power = self._PowerUp()
        # The joints the arm read, and the travel commanded on each tick of the stream
        self._stream: collections.deque[tuple[np.ndarray, np.ndarray]] = collections.deque(maxlen=_LAG_TICKS)
        self.velocity = np.zeros(JOINTS)
        self._last_step = clock.now()

    @property
    def usable(self) -> bool:
        return self.reading.power is PowerState.RUN and not self.reading.in_error

    def settle(self) -> None:
        """Judge a move in flight against the joints the arm reads. A move that did not start yet cannot arrive."""
        move = self._move
        if move is None or not self.moves.active:
            self._move = None
            return
        if move.waiting:
            move.moved_rad = max(move.moved_rad, float(np.max(np.abs(self.reading.q - move.start))))
        if self.reading.power is not PowerState.FAILURE:
            move.in_failure = False
        elif not move.in_failure:
            if not move.waiting:
                self.controller.stop_move()
            self._refuse(move)
            return
        if move.waiting:
            if self.clock.now() >= move.deadline:
                self._refuse(move)
            return
        if not self.usable:
            # The planner stopped on an error: the move waits for the error to clear and starts again
            self.controller.stop_move()
            # An arm out of RUN with no FAILURE was powered off, as deliberately as a stop
            if move.restarts >= _MAX_RESTARTS or self.reading.power is not PowerState.RUN:
                self._refuse(move)
            else:
                # The wait counts the travel of the recovery from where the planner braked
                self.reading = self.controller.read()
                self._wait(move.target, move.call, move.restarts + 1)
            return
        if self.moves.settle(self.reading.q, self.clock.now()) is MoveStatus.GAVE_UP:
            self.controller.stop_move()

    def _refuse(self, move: _Move) -> None:
        self._move = None
        self.moves.fail(MoveRefused(self.controller.errors, move.moved_rad))

    @staticmethod
    def _pose_error(target: geom.Transform3D, pose: geom.Transform3D) -> float:
        """How far ``pose`` is from ``target``: metres of the tool point plus radians of the turn."""
        turn = float(np.linalg.norm((target.rotation * pose.rotation.inv).as_rotvec))
        return float(np.linalg.norm(np.asarray(target.translation) - np.asarray(pose.translation))) + turn

    @classmethod
    def _drifts(cls, target: geom.Transform3D, before: geom.Transform3D, after: geom.Transform3D) -> bool:
        """Whether the step from ``before`` to ``after`` takes the tool further from ``target`` along an axis or in its
        turn, and brings it nearer by less than ``_MIN_PROGRESS`` of how far it moves it."""
        shift_before = np.abs(np.asarray(target.translation) - np.asarray(before.translation))
        shift_after = np.abs(np.asarray(target.translation) - np.asarray(after.translation))
        turn_before = float(np.linalg.norm((target.rotation * before.rotation.inv).as_rotvec))
        turn_after = float(np.linalg.norm((target.rotation * after.rotation.inv).as_rotvec))
        away = np.any(shift_after - shift_before > _DRIFT_M) or turn_after - turn_before > _DRIFT_RAD
        gain = cls._pose_error(target, before) - cls._pose_error(target, after)
        return bool(away and gain < _MIN_PROGRESS * cls._pose_error(after, before))

    @staticmethod
    def _clamp_to_limits(joints: np.ndarray, start: np.ndarray) -> np.ndarray:
        """``joints`` with each joint ``_LIMIT_MARGIN`` inside its limit. A joint that ``start`` has past a limit
        already does not go further, and is not pushed back either."""
        low = np.minimum(_JOINT_LIMITS[:, 0] + _LIMIT_MARGIN, start)
        high = np.maximum(_JOINT_LIMITS[:, 1] - _LIMIT_MARGIN, start)
        return np.clip(joints, low, high)

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
            joints = self._clamp_to_limits(joints, seed)
        turn = joints - seed
        joints = seed + turn * min(1.0, float(np.min(max_turn / np.maximum(np.abs(turn), 1e-12))))
        if self._drifts(pose, self.controller.forward(seed), self.controller.forward(joints)):
            return np.array(seed, dtype=np.float64)
        return self._keep_clear(seed, joints)

    def _toward(self, goal: np.ndarray, seed: np.ndarray, max_turn: np.ndarray) -> np.ndarray:
        """The joints a step from ``seed`` toward ``goal``, each joint at most ``max_turn`` from ``seed``, and not
        nearer the arm itself than the clearance."""
        return self._keep_clear(seed, seed + np.clip(goal - seed, -max_turn, max_turn))

    def _keep_clear(self, seed: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """``joints``, or ``seed`` if the step from ``seed`` to ``joints`` takes the arm nearer than the clearance to
        itself, by the collision model of the controller, at points ``_PATH_STEP`` apart. The model gives no depth in a
        collision, so a step into one is never taken."""
        start = self.controller.clearance(seed)
        points = max(1, int(np.ceil(np.max(np.abs(joints - seed)) / _PATH_STEP)))
        for share in np.linspace(0.0, 1.0, points + 1)[1:]:
            clearance = self.controller.clearance(seed + share * (joints - seed))
            if clearance <= 0.0 or (clearance < _MIN_CLEARANCE and clearance < start):
                return np.array(seed, dtype=np.float64)
        return joints

    def _check_move(self, target: np.ndarray) -> None:
        """Refuse a planned move from the joints the arm reads to ``target`` that ends past a joint limit or in a
        collision, or that takes the arm nearer to itself than the clearance and than it came along the way. The
        controller plans a straight line in the joints, and does not know the tool spheres of the local model."""
        start = self.reading.q
        if not np.array_equal(self._clamp_to_limits(target, start), target):
            raise ValueError(f'a move to {np.round(target, 3)} goes past the joint limits')
        least = min(_MIN_CLEARANCE, self.controller.clearance(start))
        points = max(1, int(np.ceil(np.max(np.abs(target - start)) / _PATH_STEP)))
        for share in np.linspace(0.0, 1.0, points + 1)[1:]:
            clearance = self.controller.clearance(start + share * (target - start))
            # The model gives no depth in a collision: a move that ends in one is not known to leave it
            if clearance < least or (share == 1.0 and clearance <= 0.0):
                raise ValueError(f'a move to {np.round(target, 3)} takes the arm {clearance:.3f} m from itself')
            least = min(_MIN_CLEARANCE, max(least, clearance))

    def _pose_of(self, cmd: command.CommandType) -> geom.Transform3D | None:
        """The tool pose that a Cartesian ``cmd`` asks for, or None for a joint target. A delta applies to the pose
        that the arm reads."""
        # TODO: PositionControl and Impedance need a controller that takes them; the velocity loop runs its own law
        command.require_native_mode(cmd, 'CRON')
        match cmd:
            case command.CartesianPosition(pose):
                pass
            case command.CartesianDelta() as delta_cmd:
                pose = delta_cmd.apply(self.reading.tcp_pose)
            case command.JointPosition():
                return None
            case other:
                raise NotImplementedError(f'Unsupported command {other}')
        if not (np.all(np.isfinite(pose.translation)) and np.all(np.isfinite(pose.rotation.as_quat))):
            raise ValueError(f'{cmd} asks for a pose that is not finite: {pose}')
        return pose

    @staticmethod
    def _checked(cmd: command.CommandType, joints: np.ndarray) -> np.ndarray:
        # A scalar would broadcast to every joint, and the controller stops the arm with a TIMEOUT error on a velocity
        # that is not finite
        if joints.shape != (JOINTS,) or not np.all(np.isfinite(joints)):
            raise ValueError(f'{cmd} asks for joints that are not {JOINTS} finite values: {joints}')
        return joints

    def _to_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints that a move ``cmd`` goes to: a Cartesian target through the exact solution of the controller."""
        pose = self._pose_of(cmd)
        if pose is not None:
            return self.controller.inverse(pose, self.reading.q)
        assert isinstance(cmd, command.JointPosition)
        return self._checked(cmd, np.asarray(cmd.positions, dtype=np.float64))

    def track(self, cmd: command.CommandType) -> None:
        """Put the loop on the target ``cmd`` asks for, with nobody waiting on the arrival. The target is the goal of
        steps on each tick, damped least squares steps for a Cartesian target: the joints move toward it no faster than
        the arm can follow, and not into the arm itself."""
        if not self.usable or self._power.run_at is not None:
            # An arm that is not in RUN, or that makes the joint move after RUN, takes no target; its status says so
            return
        if self.target is None and not self.velocity.any():
            # Only from a stop: a stream that still brakes goes on, and a HOLD at speed stops the arm. The arm can move
            # while it takes HOLD (the planner of an earlier move brakes), so the target comes from the joints after it
            self.controller.to_hold()
            self.reading = self.controller.read()
            self._stream.clear()
        # A delta starts from the arm: a seed at an earlier target that the arm has not reached would drive on to it
        fresh = self.target is None or isinstance(cmd, command.CartesianDelta)
        seed = self.reading.q if fresh else self.target
        try:
            pose = self._pose_of(cmd)
            goal = None if pose is not None else self._clamp_to_limits(self._to_joints(cmd), self.reading.q)
        except Exception:
            # The stream ends on a target it refuses, and the arm brakes: it must not drive on to the target before
            self.target, self._pose, self._joint_goal = None, None, None
            raise
        self.target, self._pose, self._joint_goal = np.array(seed, dtype=np.float64), pose, goal

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Put the loop on the target ``call`` asks for; ``settle`` answers it once the arm reads back there.

        The move waits while the stream brakes to a stop, and while the controller clears an error.
        """
        # The move ends the stream also when it is refused: the arm brakes and does not drive on to the stream target
        self.target, self._pose, self._joint_goal = None, None, None
        with pimm.calls.raise_to(call):
            target = self._to_joints(call.request)
            # The check samples each degree and the loop sends no velocity meanwhile, so a stream at speed brakes
            # first: the move is checked when it starts
            if not self.velocity.any():
                self._check_move(target)
            self._wait(target, call, 0)

    def _wait(self, target: np.ndarray, call: pimm.calls.Call[command.CommandType, None], restarts: int) -> None:
        """Put the move to ``target`` in wait, until the stream brakes to a stop and the controller clears an error."""
        now = self.clock.now()
        in_failure = self.reading.power is PowerState.FAILURE
        self._move = self._Move(target, call, now + _RECOVER_S, self.reading.q.copy(), in_failure, restarts)
        self.moves.accept(call, target, _ARRIVED_TOL, now, _RECOVER_S)

    def step(self) -> None:
        """Read the arm, and command the velocity that takes it towards the target."""
        self.reading = self.controller.read()
        self._report()
        now = self.clock.now()
        dt = min(now - self._last_step, _MAX_STEP_S)
        self._last_step = now
        if not self.usable:
            # An arm that left RUN holds no target: it must not resume one when it comes back
            self.target, self._pose, self._joint_goal = None, None, None
            self.velocity[:] = 0.0
            self._recover(now)
            return
        if self._power.asked_at is not None:
            self._power = self._PowerUp(run_at=now)
        if self._power.run_at is not None:
            # The arm gets no command until the joint move after RUN is over
            if now - self._power.run_at < STARTUP_S:
                return
            self._power = self._PowerUp()
        if self._move is not None and self._move.waiting and not self.velocity.any():
            self._start(self._move, now)
            return
        if self.target is not None:
            try:
                if self._pose is not None:
                    self.target = self._follow(self._pose, self.target, self.max_speed * dt)
                elif self._joint_goal is not None:
                    self.target = self._toward(self._joint_goal, self.target, self.max_speed * dt)
            except (ValueError, np.linalg.LinAlgError) as e:
                # The model of the controller can refuse a pose on one tick. The stream ends, and the arm brakes and
                # stays under control, as after a target it refuses
                logger.error(f'{self.controller.ip}: the streamed target is dropped: {e}')
                self.target, self._pose, self._joint_goal = None, None, None
        if self.target is None:
            # A stream that ended, by a planned move or a refused target, brakes to a stop
            if self.velocity.any():
                self._command(np.zeros(JOINTS), dt)
                if not self.velocity.any():
                    # One lost command would leave the arm at the last speed of the brake
                    self.controller.stop()
            return
        error = self.target - self.reading.q
        # Not faster than the arm can stop from in the error left: at the gain alone, braking would need more
        # than max_accel and the arm would overshoot
        limit_speed = np.minimum(np.sqrt(2.0 * self.max_accel * np.abs(error)), self.max_speed)
        self._command(np.clip(_GAIN * error, -limit_speed, limit_speed), dt)

    def _command(self, wanted: np.ndarray, dt: float) -> None:
        """Command the velocity that goes toward ``wanted`` no faster than ``max_accel``."""
        # Also on a stop: after a jump of the velocity to zero, the arm stopped on a following error (0.44 rad,
        # 2026-11-11)
        limit = self.max_accel * dt
        self.velocity += np.clip(wanted - self.velocity, -limit, limit)
        self.controller.command_velocity(self.velocity)
        self._stream.append((self.reading.q.copy(), self.velocity * dt))

    def _recover(self, now: float) -> None:
        """While a move waits, clear the error of the controller: a resume, which turns a FAILURE off, and then RUN,
        from OFF only. The error can come after the move was asked for."""
        move = self._move
        if move is None or not move.waiting:
            return
        clear_due = move.cleared_at is None or now - move.cleared_at >= RESUME_EVERY_S
        # A request that did not reach RUN in the time the controller allows is lost
        power_up_due = self._power.asked_at is None or now - self._power.asked_at >= POWER_TIMEOUT_S
        if self.reading.in_error or self.reading.power is PowerState.FAILURE:
            # A power up that ends in an error out of RUN is over: the next one asks for RUN again. One that reached
            # RUN keeps its mark, so the joint move after RUN still gets its wait
            if self.reading.power is not PowerState.RUN:
                self._power.asked_at = None
            if clear_due:
                self.controller.resume()
                move.cleared_at = now
        elif self.reading.power is PowerState.STANDBY:
            # A power up that stopped in STANDBY goes back to OFF first. It keeps its mark: a RUN that comes late still
            # gets the wait for the joint move after RUN
            if power_up_due and clear_due:
                self.controller.power_off()
                move.cleared_at = now
        elif self.reading.power is PowerState.OFF and power_up_due:
            self.controller.power_up()
            self._power = self._PowerUp(asked_at=now)

    def _start(self, move: _Move, now: float) -> None:
        """Start the move that waits, from the joints the arm reads in HOLD: a brake, a recovery or the end of an
        earlier move can move the arm."""
        try:
            self.controller.to_hold()
            self.reading = self.controller.read()
            self._check_move(move.target)
            self.controller.move_to(move.target, float(self.max_speed.min()), float(self.max_accel.min()))
        except TimeoutError as e:
            # No HOLD came, as when an error comes on this tick: the move waits on, and the next tick tries again
            logger.warning(f'{self.controller.ip}: the move waits: {e}')
            return
        except ValueError as e:
            self._move = None
            self.moves.fail(e)
            return
        distance = float(np.max(np.abs(move.target - self.reading.q)))
        travel_s = _MOVE_GRACE_S + _MOVE_SLACK * distance / float(self.max_speed.min())
        self.moves.accept(move.call, move.target, _ARRIVED_TOL, now, travel_s)
        move.waiting = False
        # The lag of a stream is measured from where the stream starts: the planner moves the arm on its own
        self._stream.clear()

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
            if self._stream:
                # The controller does not name the joint of a following error
                commanded = np.sum([travel for _, travel in self._stream], axis=0)
                lag = np.round(np.degrees(commanded - (self.reading.q - self._stream[0][0])), 1).tolist()
                logger.error(f'{self.controller.ip}: in the last {len(self._stream)} ticks the joints lag {lag} deg')
                self._stream.clear()
        else:
            logger.info(f'{self.controller.ip}: {self.reading.power.name}, no error')

    def publish(self) -> None:
        """Ship the arm as it reads now."""
        if not self.usable:
            status = RobotStatus.ERROR
        elif self.moves.active:  # the driver owns the arm until the move settles, and reads no command meanwhile
            status = RobotStatus.BUSY
        elif self.moves.errored:
            status = RobotStatus.ERROR
        else:
            status = RobotStatus.AVAILABLE
        self.state.encode(self.reading.q, self.reading.dq, self.reading.tcp_pose, status)
        self.out.emit(self.state)

    def __enter__(self) -> '_Arm':
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb) -> None:
        """Brake the stream to a stop, and answer what was waiting."""
        try:
            while self.usable and self.velocity.any():
                self._command(np.zeros(JOINTS), 1.0 / _HZ)
                # blocking-sleep-ok: the loop is over, and the controller needs the velocities at its own rate
                time.sleep(1.0 / _HZ)
                self.reading = self.controller.read()
        finally:
            self.moves.abandon(exc)


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        relative_dynamics_factor: float = 0.2,
        power_off_on_exit: bool = True,
        tool_length: float = 0.0,
        projects: Sequence[str] = (),
        tool_spheres: Sequence[tuple[Sequence[float], float]] = (),
    ) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.power_off_on_exit = power_off_on_exit
        self.tool_length = tool_length
        self.projects = tuple(projects)
        self.tool_spheres = tuple(tool_spheres)
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[CronState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        controller = CronController(self.ip, self.power_off_on_exit, self.tool_length, self.projects, self.tool_spheres)
        with controller:
            arm = _Arm(
                controller, self.sync_move, self.commands, self.state, self.relative_dynamics_factor, should_stop, clock
            )
            with arm:
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
