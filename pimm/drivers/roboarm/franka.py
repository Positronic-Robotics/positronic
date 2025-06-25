from dataclasses import dataclass
from enum import Enum

import franky
import numpy as np

import geom
import ironic2 as ir
from ironic.utils import RateLimiter


class CommandType(Enum):
    RESET = 0
    MOVE = 1


class RobotStatus(Enum):
    AVAILABLE = 0
    RESETTING = 1


class CartesianMode(Enum):
    LIBFRANKA = "libfranka"
    POSITRONIC = "positronic"


@dataclass
class Command:
    type: CommandType
    value: np.array


class State:
    _values: np.array

    def __init__(self):
        POSITION_DIM, JOINTS_DIM, STATUS_DIM = 7, 7, 1
        self._values = np.zeros(POSITION_DIM + JOINTS_DIM + STATUS_DIM)

    @property
    def position(self) -> geom.Transform3D:
        return geom.Transform3D(self._values[:3], geom.Rotation.from_quat(self._values[3:7]))

    @property
    def joints(self) -> np.array:
        return self._values[7:14]

    @property
    def status(self) -> RobotStatus:
        return RobotStatus.AVAILABLE if self._values[14] == 0 else RobotStatus.RESETTING

    def _start_reset(self):
        self._values[14] = 1

    def _finish_reset(self):
        self._values[14] = 0


class Franka:
    commands = ir.SignalReader()
    state = ir.SignalEmitter()

    def __init__(self, ip: str, relative_dynamics_factor=0.2, cartesian_mode=CartesianMode.LIBFRANKA) -> None:
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self._cartesian_mode = cartesian_mode

    @staticmethod
    def _init_robot(robot, rel_dynamics_factor: float):
        robot.set_joint_impedance([150, 150, 150, 125, 125, 250, 250])
        robot.set_collision_behavior(
            lower_torque_threshold_acceleration=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            upper_torque_threshold_acceleration=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            lower_torque_threshold_nominal=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_torque_threshold_nominal=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            lower_force_threshold_acceleration=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_force_threshold_acceleration=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            lower_force_threshold_nominal=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_force_threshold_nominal=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )

        robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
        robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])
        robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        robot.relative_dynamics_factor = rel_dynamics_factor

    def run(self, should_stop: ir.SignalReader) -> None:
        robot = franky.Robot(self._ip, realtime_config=franky.RealtimeConfig.Ignore)
        Franka._init_robot(robot, self._relative_dynamics_factor)

        home_joints_config = [0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785]  # TODO: Allow customisation

        commands = ir.ValueUpdated(self.commands)
        robot_state = State()
        last_q = None
        rate_limiter = RateLimiter(hz=500)

        def encode_state_data():
            pos = robot.current_pose.end_effector_pose
            robot_state._values[:3] = pos.translation
            # xyzw to wxyz
            robot_state._values[3] = pos.quaternion[3]
            robot_state._values[4:7] = pos.quaternion[:3]

            return robot_state

        # Reset robot
        reset_motion = franky.JointWaypointMotion([franky.JointWaypoint(home_joints_config)])

        def reset():
            robot_state._start_reset()
            self.state.emit(robot_state)
            robot.join_motion(timeout=0.1)
            robot.move(reset_motion, asynchronous=False)
            robot_state._finish_reset()
            self.state.emit(encode_state_data())

        reset()

        while not ir.signal_value(should_stop):
            command, updated = ir.signal_value(commands, (None, False))
            if updated:
                if command.type == CommandType.RESET:
                    reset()
                    continue

                q_xyzw = np.array([command.value[4], command.value[5], command.value[6], command.value[3]])
                pos = franky.Affine(translation=command.value[:3], quaternion=q_xyzw)

                if self._cartesian_mode == CartesianMode.LIBFRANKA:
                    motion = franky.CartesianMotion(pos, franky.ReferenceType.Absolute)
                else:
                    if last_q is None:
                        last_q = robot.current_joint_state.position
                    last_q = robot.inverse_kinematics(pos, last_q)
                    motion = franky.JointMotion(last_q, return_when_finished=False)

                try:
                    robot.move(motion, asynchronous=True)
                except franky.ControlException as e:
                    robot.recover_from_errors()
                    print(f"Motion failed for {pos}: {e}")

            rate_limiter.wait()
            self.state.emit(encode_state_data())
