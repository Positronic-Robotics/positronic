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
        print('Successfully set realtime scheduling priority')
    except (OSError, PermissionError) as e:
        print(f'Warning: Could not set realtime scheduling priority: {e}')
        print("Run `sudo setcap 'cap_sys_nice=eip' $(which python3)` to enable this")
        print('Control loop will run with normal scheduling')


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
        self.commands: pimm.SignalReceiver[command.CommandType] = pimm.ControlSystemReceiver(self)
        self.state: pimm.SignalEmitter[KinovaState] = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        _set_realtime_priority()
        commands = pimm.DefaultReceiver(pimm.ValueUpdated(self.commands), (None, False))
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

            while not should_stop.value:
                cmd, updated = commands.value
                if updated:
                    match cmd:
                        case command.Reset():
                            joint_controller.set_target_qpos(self.home_joints)
                        case command.CartesianMove(pose):
                            qpos = self.solver.inverse(pose, robot_state.q)
                            joint_controller.set_target_qpos(qpos)
                        case command.JointMove(positions):
                            qpos = np.array(positions, dtype=np.float32)
                            joint_controller.set_target_qpos(qpos)
                        case _:
                            print(f'Unsuported command: {cmd}')

                torque_command = joint_controller.compute_torque(q, dq, tau)
                np.divide(torque_command, torque_constant, out=current_command)
                q, dq, tau = api.apply_current_command(current_command)
                ee_pose = self.solver.forward(joint_controller.q_s)

                status = RobotStatus.MOVING if not joint_controller.finished else RobotStatus.AVAILABLE
                robot_state.encode(q, dq, ee_pose, status)
                self.state.emit(robot_state)

                yield pimm.Sleep(rate_limiter.wait_time())
