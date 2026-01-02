import abc
from abc import abstractmethod
from functools import partial
from typing import Any

import numpy as np

from positronic import geom
from positronic.dataset import transforms
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Signal
from positronic.dataset.transforms.episode import Derive
from positronic.drivers.roboarm import command

RotRep = geom.Rotation.Representation


def _convert_quat_to_array(q: geom.Rotation, representation: RotRep | str) -> np.ndarray:
    return q.to(representation).reshape(-1)


def _relative_rot_vec(q_current: np.ndarray, q_target: np.ndarray, representation: RotRep | str) -> np.ndarray:
    r_cur = geom.Rotation.from_quat(q_current)
    r_tgt = geom.Rotation.from_quat(q_target)
    rel = r_cur.inv * r_tgt
    return _convert_quat_to_array(rel, representation)


class ActionDecoder(Derive):
    def __init__(self):
        super().__init__(action=self.encode_episode)
        self._metadata = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    @meta.setter
    def meta(self, value: dict[str, Any]):
        self._metadata = value

    @abstractmethod
    def encode_episode(self, episode: Episode) -> Signal[Any]:
        pass

    @abstractmethod
    def decode(self, action: dict[str, Any], inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        """Decode action dictionary into a robot command and target grip value.

        Args:
            action: Dictionary containing action data. For existing implementations,
                   this should contain an 'action' key with the action vector.
            inputs: Dictionary of input signals (e.g., robot state)

        Returns:
            dict: Dictionary containing 'robot_command' and 'target_grip' keys
                   where 'robot_command' is a roboarm.command type and 'target_grip' is a float
        """
        pass


class RotationTranslationGripAction(ActionDecoder, abc.ABC):
    def __init__(self, rotation_representation: RotRep | str = RotRep.QUAT):
        super().__init__()
        self.rot_rep = RotRep(rotation_representation)


class AbsolutePositionAction(RotationTranslationGripAction):
    def __init__(self, tgt_ee_pose_key: str, tgt_grip_key: str, rotation_representation: RotRep | str = RotRep.QUAT):
        super().__init__(rotation_representation)
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.tgt_grip_key = tgt_grip_key

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self.tgt_ee_pose_key]
        pose = transforms.recode_transform(RotRep.QUAT, self.rot_rep, pose)
        return transforms.concat(pose, episode[self.tgt_grip_key], dtype=np.float32)

    def decode(self, action: dict[str, Any], inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        action_vector = action['action']
        target_pose = geom.Transform3D.from_vector(action_vector[:-1], self.rot_rep)
        target_grip = action_vector[-1].item()
        return {
            'robot_command': command.to_wire(command.CartesianPosition(pose=target_pose)),
            'target_grip': target_grip,
        }


class AbsoluteJointsAction(ActionDecoder):
    """Absolute joint position action decoder.

    Action vector: (num_joints + 1,) = [joint_positions..., gripper_position]
    """

    def __init__(
        self, tgt_joints_key: str = 'robot_commands.joints', tgt_grip_key: str = 'target_grip', num_joints: int = 7
    ):
        super().__init__()
        self.tgt_joints_key = tgt_joints_key
        self.tgt_grip_key = tgt_grip_key
        self.num_joints = num_joints

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        joints = episode[self.tgt_joints_key]
        return transforms.concat(joints, episode[self.tgt_grip_key], dtype=np.float32)

    def decode(self, action_vector: np.ndarray, _inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        joint_positions = action_vector[: self.num_joints]
        target_grip = action_vector[-1].item()
        return {
            'robot_command': command.to_wire(command.JointPosition(positions=joint_positions)),
            'target_grip': target_grip,
        }


class RelativeTargetPositionAction(RotationTranslationGripAction):
    def __init__(
        self,
        rotation_representation: RotRep | str = RotRep.QUAT,
        robot_pose_key: str = 'robot_state.ee_pose',
        target_pose_key: str = 'robot_commands.pose',
        target_grip_key: str = 'target_grip',
    ):
        super().__init__(rotation_representation)
        self.robot_pose_key = robot_pose_key
        self.target_pose_key = target_pose_key
        self.target_grip_key = target_grip_key

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        robot_pose = episode[self.robot_pose_key]
        target_pose = episode[self.target_pose_key]

        # Rotation difference: q_cur.inv * q_target, recoded
        robot_quat = transforms.view(robot_pose, slice(3, None))
        target_quat = transforms.view(target_pose, slice(3, None))
        rotations = transforms.pairwise(
            robot_quat, target_quat, partial(_relative_rot_vec, representation=self.rot_rep)
        )

        # Translation difference: target - current
        robot_trans = transforms.view(robot_pose, slice(0, 3))
        target_trans = transforms.view(target_pose, slice(0, 3))
        translations = transforms.pairwise(target_trans, robot_trans, np.subtract)

        # Grip: target_grip
        grips = episode[self.target_grip_key]

        return transforms.concat(rotations, translations, grips, dtype=np.float32)

    def decode(self, action: dict[str, Any], inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        print(action)
        action_vector = action['action']
        rotation = action_vector[: self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size : self.rot_rep.size + 3]

        robot_pose = inputs['robot_state.ee_pose']

        rot_mul = geom.Rotation.from_quat(robot_pose[3:7]) * q_diff
        tr_add = robot_pose[0:3] + tr_diff

        target_pose = geom.Transform3D(translation=tr_add, rotation=rot_mul)
        target_grip = action_vector[self.rot_rep.size + 3].item()
        return {
            'robot_command': command.to_wire(command.CartesianPosition(pose=target_pose)),
            'target_grip': target_grip,
        }


class JointDeltaAction(ActionDecoder):
    """DROID-style joint velocity action decoder.

    Action vector: (num_joints + 1,) = [joint_velocities..., gripper_position]
    """

    # Copied from Droid: https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L10
    RELATIVE_MAX_JOIN_DELTA = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    MAX_JOINT_DELTA = RELATIVE_MAX_JOIN_DELTA.max()
    MAX_JONIT_VEL = RELATIVE_MAX_JOIN_DELTA / MAX_JOINT_DELTA

    def __init__(self, num_joints: int = 7):
        super().__init__()
        self.num_joints = num_joints

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        raise NotImplementedError('JointVelocityAction is not supposed for training yet')

    def decode(self, action: dict[str, Any], _inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        action_vector = action['action']
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        action_vector = action_vector.clip(-1.0, 1.0)
        velocities = action_vector[: self.num_joints]
        grip = 1.0 if action_vector[self.num_joints].item() > 0.5 else 0.0

        max_vel_norm = (np.abs(velocities) / self.MAX_JONIT_VEL).max()
        if max_vel_norm > 1.0:
            velocities = velocities / max_vel_norm

        return {
            'robot_command': command.to_wire(command.JointDelta(velocities=velocities * self.MAX_JOINT_DELTA)),
            'target_grip': grip,
        }


class GrootActionDecoder(AbsolutePositionAction):
    """Decodes GR00T N1.6 action output to robot commands.

    N1.6 returns actions with keys like 'target_robot_position_translation' (no prefix).
    Each value has shape (D,) for a single time step.
    """

    def __init__(self):
        super().__init__('fake', 'fake')

    def decode(self, action: dict[str, Any], _inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        target_pose = geom.Transform3D(
            action['target_robot_position_translation'],
            geom.Rotation.from_quat(action['target_robot_position_quaternion']),
        )
        target_grip = action['target_grip'].item()
        return {
            'robot_command': command.to_wire(command.CartesianPosition(pose=target_pose)),
            'target_grip': target_grip,
        }
