import logging
from collections.abc import Iterator, Sequence
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
_GAIN = 6.0  # 1/s
# A loop that runs late must not step the velocity by more than a few ticks
_MAX_STEP_S = 5.0 / _HZ
# Radians; the arm reports joints but no goal, so arrival is judged from the joints it reads
_ARRIVED_TOL = 0.01
# On top of the travel itself: the arm slows down when it nears the target
_MOVE_GRACE_S = 3.0
# The planner of the controller travels at about 0.75 of the speed it is asked for (2026-11-05: 164 deg in 35 s at
# 0.1 rad/s), so the time a move gets is its travel at the asked speed, times this
_MOVE_SLACK = 1.5
# The elbow, whose bend says how far the arm stretches, and the joint of the wrist that lines up the axes of joints 4
# and 6 at 0 and at a half turn
_ELBOW, _WRIST = 2, 4
# A streamed target that turns a joint further than this from the previous one jumps to another branch of the
# solutions: up from the start pose, the IK turned the wrist by 144 and 180 degrees for 1 cm of the tool
_MAX_TARGET_TURN = 0.5  # rad
# The halvings of the path to a target that the arm cannot hold, to find the furthest point it can
_LIMIT_SEARCH_STEPS = 12


def _crosses(joints: np.ndarray, seed: np.ndarray, joint: int) -> bool:
    """Whether ``joint`` goes from ``seed`` to ``joints`` across 0 or a half turn. A start on one of them crosses
    nothing."""
    return bool(np.sin(joints[joint]) * np.sin(seed[joint]) < 0)


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
        min_elbow: float | None = None,
        min_wrist: float | None = None,
        min_shoulder: float | None = None,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=_HZ)
        self.controller = controller
        # The margins from the three poses where joints turn with no limit for a small move of the tool: the elbow
        # straight, joint 5 at 0 or a half turn (radians each), and the wrist on the axis of joint 1 (metres, see
        # `CronController.shoulder_clearance`). The arm does not take a Cartesian target inside a margin. Near full
        # stretch the controller stopped the arm on a following error, and near the axis of joint 1 on a JOINT_VEL error
        self.min_elbow = min_elbow
        self.min_wrist = min_wrist
        self.min_shoulder = min_shoulder
        self._at_limit = False
        self.out = out
        self.state = CronState()
        self.max_speed = dynamics_factor * _MAX_JOINT_SPEED
        self.max_accel = dynamics_factor * _MAX_JOINT_ACCEL
        self.reading = controller.read()
        self._condition = (self.reading.power, self.reading.in_error)
        self.target: np.ndarray | None = None
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

    def _holdable(self, pose: geom.Transform3D, seed: np.ndarray, max_turn: float | None) -> np.ndarray | None:
        """The joints for ``pose`` next to ``seed``, or None where the arm cannot hold ``pose``: no solution, the elbow
        inside a margin from a pose where joints turn with no limit, or a joint turned further than ``max_turn`` from
        ``seed``."""
        try:
            joints = self.controller.inverse(pose, seed)
        except ValueError:
            return None
        elbow = abs(float(np.remainder(joints[_ELBOW] + np.pi, 2 * np.pi) - np.pi))
        # A step can cross a straight elbow or wrist and land past the margin on the other side
        if self.min_elbow is not None and (elbow < self.min_elbow or _crosses(joints, seed, _ELBOW)):
            return None
        if self.min_wrist is not None and (
            abs(np.sin(joints[_WRIST])) < np.sin(self.min_wrist) or _crosses(joints, seed, _WRIST)
        ):
            return None
        if self.min_shoulder is not None and self.controller.shoulder_clearance(joints) < self.min_shoulder:
            return None
        if max_turn is not None and np.max(np.abs(joints - seed)) > max_turn:
            return None
        return joints

    def _toward(self, pose: geom.Transform3D, seed: np.ndarray) -> np.ndarray:
        """The joints for ``pose``, or for the furthest pose on the way to it from ``seed`` that the arm can hold."""
        joints = self._holdable(pose, seed, _MAX_TARGET_TURN)
        if joints is not None:
            self._at_limit = False
            return joints
        if not self._at_limit:
            logger.info(f'{self.controller.ip}: the arm cannot hold {pose}: the target stops at the limit on the way')
            self._at_limit = True
        start, joints, low, high = self.controller.forward(seed), seed, 0.0, 1.0
        for _ in range(_LIMIT_SEARCH_STEPS):
            middle = (low + high) / 2
            found = self._holdable(start.interpolate(pose, middle), seed, _MAX_TARGET_TURN)
            if found is None:
                high = middle
            else:
                low, joints = middle, found
        return joints

    def _solve(self, pose: geom.Transform3D, seed: np.ndarray, streamed: bool) -> np.ndarray:
        if streamed:
            return self._toward(pose, seed)
        joints = self._holdable(pose, seed, None)
        if joints is None:
            raise ValueError(f'the arm cannot hold {pose}')
        return joints

    def _to_joints(self, cmd: command.CommandType, streamed: bool) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold. A streamed Cartesian target that the arm cannot hold stops at the
        limit on the way to it; a move that a caller waits for fails."""
        # TODO: PositionControl and Impedance need a controller that takes them; the velocity loop runs its own law
        command.require_native_mode(cmd, 'CRON')
        seed = self.target if streamed and self.target is not None else self.reading.q
        match cmd:
            case command.CartesianPosition(pose):
                joints = self._solve(pose, seed, streamed)
            case command.CartesianDelta() as delta_cmd:
                joints = self._solve(delta_cmd.apply(self.controller.forward(seed)), seed, streamed)
            case command.JointPosition(positions):
                joints = np.asarray(positions, dtype=np.float64)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')
        # The controller refuses a velocity that is not finite and stops the arm with a TIMEOUT error
        if not np.all(np.isfinite(joints)):
            raise ValueError(f'{cmd} asks for joints that are not finite: {joints}')
        return joints

    def track(self, cmd: command.CommandType) -> None:
        """Put the loop on the target ``cmd`` asks for, with nobody waiting on the arrival."""
        if self.target is None:
            self.controller.to_hold()
        self.target = self._to_joints(cmd, streamed=True)

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Put the loop on the target ``call`` asks for; ``settle`` answers it once the arm reads back there."""
        with pimm.calls.raise_to(call):
            target = self._to_joints(call.request, streamed=False)
            if self.target is not None:
                self.target = None
                self.velocity[:] = 0.0
                self.controller.stop()
            if self.reading.in_error:
                # The controller refuses a move while it holds an error
                self.controller.resume()
                self._deferred = target
            else:
                self.controller.move_to(target, self.max_speed, self.max_accel)
            travel_s = _MOVE_GRACE_S + _MOVE_SLACK * float(np.max(np.abs(target - self.reading.q)) / self.max_speed)
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
            self.target = None
            self.velocity[:] = 0.0
            return
        if self._deferred is not None:
            self.controller.move_to(self._deferred, self.max_speed, self.max_accel)
            self._deferred = None
        if self.target is None:
            return
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
        min_elbow: float | None = None,
        min_wrist: float | None = None,
        min_shoulder: float | None = None,
    ) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.power_off_on_exit = power_off_on_exit
        self.tool_length = tool_length
        self.projects = tuple(projects)
        self.min_elbow = min_elbow
        self.min_wrist = min_wrist
        self.min_shoulder = min_shoulder
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[CronState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def _arm(self, controller: CronController, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Arm:
        """The arm this run drives, built from the driver's configuration."""
        return _Arm(
            controller,
            self.sync_move,
            self.commands,
            self.state,
            self.relative_dynamics_factor,
            should_stop,
            clock,
            self.min_elbow,
            self.min_wrist,
            self.min_shoulder,
        )

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        with CronController(self.ip, self.power_off_on_exit, self.tool_length, self.projects) as controller:
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
