import logging
import os
from collections.abc import Iterator

import numpy as np
from mujoco import Any

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State, command
from positronic.drivers.roboarm.kinova.api import KinovaAPI
from positronic.drivers.roboarm.kinova.base import JointCompliantController, KinematicsSolver


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

    def _start_reset(self):
        self.array[14 + 7] = RobotStatus.RESETTING.value

    def _finish_reset(self):
        self.array[14 + 7] = RobotStatus.AVAILABLE.value

    def encode(self, q, dq, ee_pose, status: RobotStatus):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14 : 14 + 3] = ee_pose.translation
        self.array[14 + 3 : 14 + 7] = ee_pose.rotation.as_quat
        self.array[14 + 7] = status.value


class Robot(pimm.ControlSystem):
    def __init__(self, ip: str, relative_dynamics_factor=0.2, home_joints: list[float] | None = None) -> None:
        self.ip = ip
        self.relative_dynamics_factor = relative_dynamics_factor
        self.solver = KinematicsSolver()
        self.home_joints = home_joints if home_joints is not None else [0.0, -0, 0.5, -1.5, 0.0, -0.5, 1.57079633]
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        # The synchronous version of the above
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state: pimm.SignalEmitter[KinovaState] = pimm.ControlSystemEmitter(self)

    def _target_qpos(self, joint_controller, robot_state: KinovaState, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold."""
        match cmd:
            case command.Reset():
                return np.asarray(self.home_joints, dtype=np.float32)
            case command.CartesianPosition(pose):
                return self.solver.inverse(pose, robot_state.q)
            case command.CartesianDelta() as delta_cmd:
                target = delta_cmd.apply(self.solver.forward(joint_controller.q_s))
                return self.solver.inverse(target, robot_state.q)
            case command.JointPosition(positions):
                return np.array(positions, dtype=np.float32)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
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
            pending_move = None

            while not should_stop.value:
                if pending_move is not None and joint_controller.finished:
                    pending_move.set_result(None)
                    pending_move = None
                if pending_move is None and (call := next(self.sync_move.incoming(), None)) is not None:
                    try:
                        joint_controller.set_target_qpos(self._target_qpos(joint_controller, robot_state, call.request))
                    # rules-allow: swallowed-error — an arm that cannot be placed is the asker's failure to hear about
                    except Exception as exc:
                        call.set_exception(exc)
                    else:
                        pending_move = call
                elif (cmd := pimm.value_updated(self.commands)) is not None:
                    joint_controller.set_target_qpos(self._target_qpos(joint_controller, robot_state, cmd))

                torque_command = joint_controller.compute_torque(q, dq, tau)
                np.divide(torque_command, torque_constant, out=current_command)
                q, dq, tau = api.apply_current_command(current_command)
                ee_pose = self.solver.forward(joint_controller.q_s)

                status = RobotStatus.MOVING if not joint_controller.finished else RobotStatus.AVAILABLE
                robot_state.encode(q, dq, ee_pose, status)
                self.state.emit(robot_state)

                yield rate_limiter.wait()
