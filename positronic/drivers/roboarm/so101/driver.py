import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.kinematics import Kinematics
from positronic.drivers.roboarm.models import DEFAULT_FRAME, add_default_frame


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


_SO101_URDF_PATH = 'positronic/drivers/roboarm/so101/so101.urdf'
_SO101_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
_SO101_EE_LINK = 'gripper_frame_link'
_SO101_EE_JOINT = 'gripper_frame_joint'


class Robot(pimm.ControlSystem):
    def __init__(self, motor_bus: MotorBus, home_joints: list[float] | None = None):
        self.motor_bus = motor_bus
        self.mujoco_model_path = 'positronic/drivers/roboarm/so101/so101.xml'
        self.kinematic = Kinematics(_SO101_URDF_PATH, _SO101_EE_JOINT)
        self.joint_limits = self.kinematic.joint_limits
        self.home_joints = home_joints if home_joints is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        self.commands: pimm.SignalReceiver[roboarm_command.Trajectory[roboarm_command.CommandType]] = (
            pimm.ControlSystemReceiver(self, default=[])
        )
        self.executed_commands: pimm.ControlSystemEmitter[
            list[roboarm_command.Applied[roboarm_command.CommandType]]
        ] = pimm.ControlSystemEmitter(self)
        self.target_grip: pimm.SignalReceiver[roboarm_command.Trajectory[float]] = pimm.ControlSystemReceiver(
            self, default=[]
        )
        self.executed_target_grip: pimm.ControlSystemEmitter[list[roboarm_command.Applied[float]]] = (
            pimm.ControlSystemEmitter(self)
        )
        self._last_grip: float = 0.0

        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.state: pimm.SignalEmitter[SO101State] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

        print('================================================================')
        print('Warning: Proper dq units is not implemented for SO101!')
        print('================================================================')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.motor_bus.connect()
        urdf = ET.fromstring(Path(_SO101_URDF_PATH).read_text())
        add_default_frame(urdf, _SO101_EE_LINK)
        self.robot_meta.emit({
            keys.URDF: ET.tostring(urdf, encoding='unicode'),
            keys.JOINT_NAMES: _SO101_JOINT_NAMES,
            keys.CONTROL_FRAME: DEFAULT_FRAME,
        })

        rate_limit = pimm.RateLimiter(hz=1000, clock=clock)
        state = SO101State()

        player = roboarm_command.TrajectoryPlayer(reduce=roboarm_command.reduce)
        grip_player = roboarm_command.TrajectoryPlayer[float]()
        # The motors take one position for the whole chain, so a grip target reaches them only with the next
        # arm command; until then it is played but unsent.
        pending_grip: roboarm_command.Applied[float] | None = None

        while not should_stop.value:
            cmd_msg = self.commands.read()
            if cmd_msg is not None and cmd_msg.updated:
                player.set(cmd_msg.data)
            grip_msg = self.target_grip.read()
            if grip_msg is not None and grip_msg.updated:
                grip_player.set(grip_msg.data)
                if not grip_msg.data:  # cancelled: a grip still waiting for a write never ran
                    pending_grip = None
            played_grip = grip_player.advance(clock.now_ns())
            if played_grip is not None:
                pending_grip = played_grip
                self._last_grip = played_grip.value
            played = player.advance(clock.now_ns())
            if played is not None:
                self._apply_command(played.value, state)
                self.executed_commands.emit([played])
                if pending_grip is not None:
                    self.executed_target_grip.emit([pending_grip])
                    pending_grip = None

            q = self.motor_bus.position
            dq = self.motor_bus.velocity[:-1]
            ee_pose, gripper = self._forward_kinematics(q)
            position_rad = self.norm_to_rad(q)[:-1]
            state.encode(position_rad, dq, ee_pose)

            self.state.emit(state)
            self.grip.emit(gripper)
            yield rate_limit.wait()

    def _apply_command(self, cmd, state: SO101State) -> None:
        """Drive the chain to ``cmd``. The motors take one position for the whole chain, so every write
        carries the latest grip target alongside the joints."""
        match cmd:
            case roboarm_command.Reset():
                raise NotImplementedError('Reset not implemented')
            case roboarm_command.CartesianPosition(pose):
                qpos = self._solve_ik(state, pose)
                q_with_gripper = np.concatenate([qpos, [self._last_grip]])
                self.motor_bus.set_target_position(q_with_gripper)
            case roboarm_command.CartesianDelta() as delta_cmd:
                ee_pose, _ = self._forward_kinematics(self.motor_bus.position)
                target = delta_cmd.apply(ee_pose)
                qpos = self._solve_ik(state, target)
                q_with_gripper = np.concatenate([qpos, [self._last_grip]])
                self.motor_bus.set_target_position(q_with_gripper)
            case roboarm_command.JointPosition(qpos):
                q_norm = self.rad_to_norm(qpos)
                q_with_gripper = np.concatenate([q_norm, [self._last_grip]])
                self.motor_bus.set_target_position(q_with_gripper)
            case _:
                raise ValueError(f'Unknown command: {cmd}')

    def _solve_ik(self, state, pose: geom.Transform3D) -> np.ndarray:
        q = np.array(state.q).tolist()
        q.append(0.0)  # ignore gripper in ik
        q_rad_new = self.kinematic.inverse(q, pose, n_iter=10)
        return self.rad_to_norm(q_rad_new)[:-1]

    def _forward_kinematics(self, motor_position) -> tuple[geom.Transform3D, float]:
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
