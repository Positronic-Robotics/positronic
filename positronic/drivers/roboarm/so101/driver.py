from collections.abc import Iterator

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.kinematics import Kinematics


class SO101State(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(5 + 5 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self) -> tuple[geom.Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:5]

    @property
    def dq(self) -> np.ndarray:
        return self.array[5:10]

    @property
    def ee_pose(self) -> geom.Transform3D:
        translation = self.array[10 : 10 + 3]
        quaternion = geom.Rotation.from_quat(self.array[10 + 3 : 10 + 7])
        return geom.Transform3D(translation, quaternion)

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[17]))

    def encode(self, q, dq, ee_pose):
        self.array[:5] = q
        self.array[5:10] = dq
        self.array[10 : 10 + 3] = ee_pose.translation
        self.array[10 + 3 : 10 + 7] = ee_pose.rotation.as_quat
        self.array[17] = RobotStatus.AVAILABLE.value


class Robot(pimm.ControlSystem):
    def __init__(self, motor_bus: MotorBus, home_joints: list[float] | None = None):
        self.motor_bus = motor_bus
        self.mujoco_model_path = 'positronic/drivers/roboarm/so101/so101.xml'
        self.kinematic = Kinematics('positronic/drivers/roboarm/so101/so101.urdf', 'gripper_frame_joint')
        self.joint_limits = self.kinematic.joint_limits
        self.home_joints = home_joints if home_joints is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self)

        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.state: pimm.SignalEmitter[SO101State] = pimm.ControlSystemEmitter(self)

        print('================================================================')
        print('Warning: Proper dq units is not implemented for SO101!')
        print('================================================================')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.motor_bus.connect()

        rate_limit = pimm.RateLimiter(hz=1000, clock=clock)
        state = SO101State()
        initial_grip = self.motor_bus.position[-1]

        command_receiver = pimm.DefaultReceiver(pimm.ValueUpdated(self.commands), (None, False))
        target_grip = pimm.DefaultReceiver(self.target_grip, initial_grip)

        while not should_stop.value:
            command, is_updated = command_receiver.value
            if is_updated:
                match command:
                    case roboarm_command.Reset():
                        raise NotImplementedError('Reset not implemented')
                    case roboarm_command.CartesianMove():
                        qpos = self._solve_ik(state, command)
                        q_with_gripper = np.concatenate([qpos, [target_grip.value]])
                        self.motor_bus.set_target_position(q_with_gripper)
                    case roboarm_command.JointMove(qpos):
                        q_norm = self.rad_to_norm(qpos)
                        q_with_gripper = np.concatenate([q_norm, [target_grip.value]])
                        self.motor_bus.set_target_position(q_with_gripper)
                    case _:
                        raise ValueError(f'Unknown command: {command}')

            q = self.motor_bus.position
            dq = self.motor_bus.velocity[:-1]
            ee_pose, gripper = self._forward_kinematics(q)
            position_rad = self.norm_to_rad(q)[:-1]
            state.encode(position_rad, dq, ee_pose)

            self.state.emit(state)
            self.grip.emit(gripper)
            yield pimm.Sleep(rate_limit.wait_time())

    def _solve_ik(self, state, command: roboarm_command.CartesianMove) -> np.ndarray:
        q = np.array(state.q).tolist()
        q.append(0.0)  # ignore gripper in ik
        q_rad_new = self.kinematic.inverse(q, command.pose, n_iter=10)
        return self.rad_to_norm(q_rad_new)[:-1]

    def _forward_kinematics(self, motor_position) -> geom.Transform3D:
        q_rad = self.norm_to_rad(motor_position)
        ee_pose = self.kinematic.forward(q_rad)
        gripper = motor_position[-1]
        return ee_pose, gripper

    def norm_to_rad(self, qpos: np.ndarray) -> np.ndarray:
        """Convert normalized position [0, 1] to radians.

        Args:
            qpos: Normalized position.

        Returns:
            Radians.
        """
        return qpos * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]

    def rad_to_norm(self, qpos: np.ndarray) -> np.ndarray:
        """Convert radians to normalized position [0, 1]."""
        return (qpos - self.joint_limits[:, 0]) / (self.joint_limits[:, 1] - self.joint_limits[:, 0])
