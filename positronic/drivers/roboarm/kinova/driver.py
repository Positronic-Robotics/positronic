import logging
import os
from collections.abc import Iterator

import numpy as np
from mujoco import Any

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State, command
from positronic.drivers.roboarm.kinova.api import KinovaAPI
from positronic.drivers.roboarm.kinova.base import JointCompliantController, KinematicsSolver, wrap_joint_angle
from positronic.drivers.utils import MoveStatus, PendingMove


def _set_realtime_priority():
    try:
        # Set realtime scheduling priority
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        logging.info('Successfully set realtime scheduling priority')
    except (OSError, PermissionError) as e:
        logging.warning(f'Warning: Could not set realtime scheduling priority: {e}')
        logging.warning("Run `sudo setcap 'cap_sys_nice=eip' $(which python3)` to enable this")
        logging.warning('Control loop will run with normal scheduling')


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


class Robot(pimm.ControlSystem):
    # Radians; the arm reports joints but no goal, so arrival is judged from the joints it reads
    _ARRIVED_TOL = 0.02
    # On top of the travel itself: the controller ramps in and out of its speed cap, and the arm settles late
    _MOVE_GRACE_S = 3.0

    def __init__(self, ip: str, relative_dynamics_factor=0.2, home_joints: list[float] | None = None) -> None:
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.solver = KinematicsSolver()
        self.home_joints = home_joints if home_joints is not None else [0.0, -0, 0.5, -1.5, 0.0, -0.5, 1.57079633]
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state: pimm.SignalEmitter[KinovaState] = pimm.ControlSystemEmitter(self)

    def _target_qpos(self, joint_controller, q: np.ndarray, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold, solved from the joints ``q`` it reports now."""
        match cmd:
            case command.Reset():
                return np.asarray(self.home_joints, dtype=np.float32)
            case command.CartesianPosition(pose):
                return self.solver.inverse(pose, q)
            case command.CartesianDelta() as delta_cmd:
                target = delta_cmd.apply(self.solver.forward(joint_controller.q_s))
                return self.solver.inverse(target, q)
            case command.JointPosition(positions):
                return np.array(positions, dtype=np.float32)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def _travel_s(self, joint_controller, q: np.ndarray, target: np.ndarray) -> float:
        """How long the arm may take to reach ``target``, from the speed its controller is capped at.

        ``relative_dynamics_factor`` scales that cap, so a conservative factor buys proportionally more time
        rather than failing moves the arm is tracking perfectly well.
        """
        return self._MOVE_GRACE_S + float(np.max(np.abs(target - q)) / np.min(joint_controller.max_velocity))

    def _take_setpoint(self, joint_controller, move: PendingMove, q: np.ndarray, clock: pimm.Clock) -> None:
        """Put the controller on whichever setpoint is on offer: a synchronous move first, then the stream."""
        if (call := next(self.sync_move.incoming(), None)) is not None:
            with pimm.calls.forward_failure(call):
                target = self._target_qpos(joint_controller, q, call.request)
                joint_controller.set_target_qpos(target)
                # The branch the controller tracks: it wraps the target once, when it is set
                wrapped = wrap_joint_angle(target, q)
                move.accept(call, wrapped, clock.now(), self._travel_s(joint_controller, q, wrapped))
        elif (cmd := pimm.value_updated(self.commands)) is not None:
            try:
                joint_controller.set_target_qpos(self._target_qpos(joint_controller, q, cmd))
            # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
            except Exception as exc:
                logging.warning(f'{cmd} not applied: {exc}')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        _set_realtime_priority()
        robot_state = KinovaState()
        rate_limiter = pimm.RateLimiter(clock, hz=1000)

        torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])

        with KinovaAPI(self.ip) as api:
            joint_controller = JointCompliantController(
                actuator_count=api.actuator_count, relative_dynamics_factor=self.relative_dynamics_factor
            )

            q, dq, tau = api.apply_current_command(None)  # Warm up
            joint_controller.compute_torque(q, dq, tau)
            current_command = np.zeros(api.actuator_count, dtype=np.float32)

            # The arm is torque-controlled, so it only travels while this loop runs: it cannot be held for a move
            move = PendingMove(self._ARRIVED_TOL)

            try:
                while not should_stop.value:
                    # The actuators report 0..2pi, so a move across the boundary reads a turn from its target
                    # until the reading is put on the same branch the controller tracks.
                    if move.active and move.settle(wrap_joint_angle(q, move.target), clock.now()) is MoveStatus.GAVE_UP:
                        # Leaving the controller on the target the arm stopped short of would resume the move
                        # once whatever blocked it goes away, long after its asker was told it failed.
                        joint_controller.set_target_qpos(q)
                    if not move.active:
                        self._take_setpoint(joint_controller, move, q, clock)

                    torque_command = joint_controller.compute_torque(q, dq, tau)
                    np.divide(torque_command, torque_constant, out=current_command)
                    q, dq, tau = api.apply_current_command(current_command)
                    ee_pose = self.solver.forward(joint_controller.q_s)

                    if move.active:  # the driver owns the arm until the move answers, and reads no command meanwhile
                        status = RobotStatus.BUSY
                    elif move.errored:  # the controller reports its trajectory, not whether the arm got there
                        status = RobotStatus.ERROR
                    else:  # a streamed setpoint still in flight is one the arm is tracking, not one it owns
                        status = RobotStatus.AVAILABLE
                    robot_state.encode(q, dq, ee_pose, status)
                    self.state.emit(robot_state)

                    yield rate_limiter.wait()
            except Exception as exc:
                move.fail(exc)  # a run that dies mid-move must not leave its asker waiting
                raise
