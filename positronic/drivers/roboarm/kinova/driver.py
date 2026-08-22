import logging
import os
from collections.abc import Iterator
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State, command
from positronic.drivers.roboarm.kinova.api import KinovaAPI
from positronic.drivers.roboarm.kinova.base import JointCompliantController, KinematicsSolver, wrap_joint_angle
from positronic.drivers.utils import DriverRun, MoveStatus, PendingMove

logger = logging.getLogger(__name__)


def _set_realtime_priority():
    try:
        # Set realtime scheduling priority
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        logger.info('Successfully set realtime scheduling priority')
    except (OSError, PermissionError) as e:
        logger.warning(f'Warning: Could not set realtime scheduling priority: {e}')
        logger.warning("Run `sudo setcap 'cap_sys_nice=eip' $(which python3)` to enable this")
        logger.warning('Control loop will run with normal scheduling')


class KinovaState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(7 + 7 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:7]

    @property
    def dq(self) -> np.ndarray:
        return self.array[7:14]

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(
            translation=self.array[14 : 14 + 3], rotation=geom.Rotation.from_quat(self.array[14 + 3 : 14 + 7])
        )

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[14 + 7]))

    def encode(self, q, dq, ee_pose, status: RobotStatus):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14 : 14 + 3] = ee_pose.translation
        self.array[14 + 3 : 14 + 7] = ee_pose.rotation.as_quat
        self.array[14 + 7] = status.value


class _Arm(DriverRun):
    """The arm the driver drives: the API it commands current through, and the controller that decides it.

    Torque-controlled, so the arm only travels while the loop steps it and cannot be held for a move.
    """

    # Radians; the arm reports joints but no goal, so arrival is judged from the joints it reads
    _ARRIVED_TOL = 0.02
    # On top of the travel itself: the controller ramps in and out of its speed cap, and the arm settles late
    _MOVE_GRACE_S = 3.0
    # Per actuator: the controller decides a torque, the API takes the current that produces it
    _TORQUE_CONSTANT = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])

    def __init__(
        self,
        api: KinovaAPI,
        calls: pimm.calls.ControlSystemHandler[command.CommandType, None],
        prepare: pimm.calls.ControlSystemHandler[Any, None],
        out: pimm.SignalEmitter[KinovaState],
        home_joints: list[float],
        dynamics_factor: float,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(should_stop, clock, hz=1000)
        actuators = api.actuator_count
        assert actuators is not None, 'the arm counts its actuators on connect, and this one is not connected'
        self.api = api
        self.out = out
        self.state = KinovaState()
        self.solver = KinematicsSolver()
        self.controller = JointCompliantController(actuator_count=actuators, relative_dynamics_factor=dynamics_factor)
        self._home_joints = home_joints
        self._calls = calls
        self._prepare = prepare
        self._move = PendingMove[command.CommandType](self._ARRIVED_TOL)
        self._current = np.zeros(actuators, dtype=np.float32)
        # Read here rather than left empty: every setpoint below is solved from where the arm is now
        self.q, self.dq, self.tau = api.apply_current_command(None)
        self.controller.compute_torque(self.q, self.dq, self.tau)

    @property
    def takes_commands(self) -> bool:
        """Whether the arm will take a setpoint: a move owns it until it is answered."""
        return not (self._move.active or self._move.settled)

    def take_prepare(self) -> pimm.calls.Call[Any, None] | None:
        """The next prepare asked for, if the arm is free to take one."""
        return self._move.take(self._prepare)

    def take_sync_move(self) -> pimm.calls.Call[command.CommandType, None] | None:
        """The next move asked for, if the arm is free to take one."""
        return self._move.take(self._calls)

    def settle(self) -> None:
        """Judge a move in flight against the joints the arm reads."""
        if not self._move.active:
            return
        # The actuators report 0..2pi, so a move across the boundary reads a turn from its target
        # until the reading is put on the same branch the controller tracks.
        settled = self._move.settle(wrap_joint_angle(self.q, self._move.target), self.clock.now())
        if settled is MoveStatus.GAVE_UP:
            # Leaving the controller on the target the arm stopped short of would resume the move
            # once whatever blocked it goes away, long after its asker was told it failed.
            self.controller.set_target_qpos(self.q)

    def _target_qpos(self, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold, solved from the joints it reports now."""
        match cmd:
            case command.Reset():
                return np.asarray(self._home_joints, dtype=np.float32)
            case command.CartesianPosition(pose):
                return self.solver.inverse(pose, self.q)
            case command.CartesianDelta() as delta_cmd:
                target = delta_cmd.apply(self.solver.forward(self.controller.q_s))
                return self.solver.inverse(target, self.q)
            case command.JointPosition(positions):
                return np.array(positions, dtype=np.float32)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def track(self, cmd: command.CommandType) -> None:
        """Put the controller on the setpoint ``cmd`` asks for, with nobody waiting on the arrival."""
        self.controller.set_target_qpos(self._target_qpos(cmd))

    def _travel_s(self, target: np.ndarray) -> float:
        """How long the arm may take to reach ``target``, from the speed its controller is capped at."""
        return self._MOVE_GRACE_S + float(np.max(np.abs(target - self.q)) / np.min(self.controller.max_velocity))

    def serve_sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> None:
        """Put the controller where ``call`` asks; ``settle`` answers it once the arm reads back there."""
        with pimm.calls.raise_to(call):
            self._hold_for(call, self._target_qpos(call.request))

    def serve_prepare(self, call: pimm.calls.Call[Any, None]) -> None:
        """Put the controller at home; ``settle`` answers ``call`` once the arm reads back there."""
        with pimm.calls.raise_to(call):
            self._hold_for(call, self._target_qpos(command.Reset()))

    def _hold_for(self, call: pimm.calls.Call[Any, None], target: np.ndarray) -> None:
        self.controller.set_target_qpos(target)
        # The branch the controller tracks: it wraps the target once, when it is set
        wrapped = wrap_joint_angle(target, self.q)
        self._move.accept(call, wrapped, self.clock.now(), self._travel_s(wrapped))

    def step(self) -> None:
        """Drive one cycle: the controller's torque as current, and what the actuators report back."""
        torque = self.controller.compute_torque(self.q, self.dq, self.tau)
        np.divide(torque, self._TORQUE_CONSTANT, out=self._current)
        self.q, self.dq, self.tau = self.api.apply_current_command(self._current)

    def publish(self) -> None:
        """Ship the arm as it reads now."""
        if self._move.active:  # the driver owns the arm until the move settles, and reads no command meanwhile
            status = RobotStatus.BUSY
        elif self._move.errored:  # the controller reports its trajectory, not whether the arm got there
            status = RobotStatus.ERROR
        else:  # a streamed setpoint still in flight is one the arm is tracking, not one it owns
            status = RobotStatus.AVAILABLE
        self.state.encode(self.q, self.dq, self.solver.forward(self.controller.q_s), status)
        self.out.emit(self.state)

    def answer(self) -> None:
        """Hand a move settled this tick its outcome, now that the state saying so is out."""
        self._move.answer()

    def __enter__(self) -> '_Arm':
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb) -> None:
        """The loop stepping the arm is what makes it travel, so a stop halts it; answer what was waiting."""
        self._move.abandon(exc)


class Robot(pimm.ControlSystem):
    def __init__(self, ip: str, relative_dynamics_factor=0.2, home_joints: list[float] | None = None) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.home_joints = home_joints if home_joints is not None else [0.0, -0, 0.5, -1.5, 0.0, -0.5, 1.57079633]
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.prepare = pimm.calls.ControlSystemHandler[Any, None](self)
        self.state = pimm.ControlSystemEmitter[KinovaState](self)

    def _arm(self, api: KinovaAPI, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Arm:
        """The arm this run drives, built from the driver's configuration."""
        return _Arm(
            api,
            self.sync_move,
            self.prepare,
            self.state,
            self.home_joints,
            self.relative_dynamics_factor,
            should_stop,
            clock,
        )

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        _set_realtime_priority()
        with KinovaAPI(self.ip) as api:
            with self._arm(api, should_stop, clock) as arm:
                while not should_stop.value:
                    arm.settle()
                    if arm.takes_commands:
                        if (call := arm.take_prepare()) is not None:
                            arm.serve_prepare(call)
                        elif (call := arm.take_sync_move()) is not None:
                            arm.serve_sync_move(call)
                        elif (cmd := pimm.value_updated(self.commands)) is not None:
                            try:
                                arm.track(cmd)
                            # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
                            except Exception as exc:
                                logger.warning(f'{cmd} not applied: {exc}')

                    arm.step()
                    arm.publish()
                    arm.answer()  # the state a settled move is answered with is out

                    yield arm.limiter.wait()
