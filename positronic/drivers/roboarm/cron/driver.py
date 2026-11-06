import logging
from collections.abc import Iterator
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


class CronState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(JOINTS + JOINTS + 7 + 1,), dtype=np.float32)

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

    def _to_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold, solved from the joints it reads now."""
        # TODO: PositionControl and Impedance need a controller that takes them; the velocity loop runs its own law
        command.require_native_mode(cmd, 'CRON')
        match cmd:
            case command.CartesianPosition(pose):
                joints = self.controller.inverse(pose, self.reading.q)
            case command.CartesianDelta() as delta_cmd:
                held = self.reading.q if self.target is None else self.target
                joints = self.controller.inverse(delta_cmd.apply(self.controller.forward(held)), held)
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
        self.target = self._to_joints(cmd)

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Put the loop on the target ``call`` asks for; ``settle`` answers it once the arm reads back there."""
        with pimm.calls.raise_to(call):
            target = self._to_joints(call.request)
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
            logger.error(f'{self.controller.ip}: {self.reading.power.name}, error {self.controller.errors}')
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
        project: str | None = None,
    ) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.power_off_on_exit = power_off_on_exit
        self.tool_length = tool_length
        self.project = project
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
        with CronController(self.ip, self.power_off_on_exit, self.tool_length, self.project) as controller:
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
