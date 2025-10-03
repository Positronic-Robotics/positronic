import abc
from abc import abstractmethod
from functools import partial
from typing import Any

import numpy as np

from positronic import geom
from positronic.dataset import transforms
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Signal

RotRep = geom.Rotation.Representation


def _convert_quat_to_array(q: geom.Rotation, representation: RotRep | str) -> np.ndarray:
    return q.to(representation).reshape(-1)


def _relative_rot_vec(q_current: np.ndarray, q_target: np.ndarray, representation: RotRep | str) -> np.ndarray:
    r_cur = geom.Rotation.from_quat(q_current)
    r_tgt = geom.Rotation.from_quat(q_target)
    rel = r_cur.inv * r_tgt
    rel = geom.Rotation.from_quat(geom.normalise_quat(rel.as_quat))
    return _convert_quat_to_array(rel, representation)


class ActionDecoder(transforms.EpisodeTransform):
    @property
    def keys(self) -> list[str]:
        return ['action']

    def transform(self, name: str, episode: Episode) -> Signal[Any] | Any:
        if name != 'action':
            raise ValueError(f'Unknown action key: {name}')
        return self.encode_episode(episode)

    @abstractmethod
    def encode_episode(self, episode: Episode) -> Signal[Any]:
        pass

    @abstractmethod
    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        pass


class RotationTranslationGripAction(ActionDecoder, abc.ABC):
    def __init__(self, rotation_representation: RotRep | str = RotRep.QUAT):
        self.rot_rep = RotRep(rotation_representation)
        # self.rotation_shape = self.rot_rep.shape


class AbsolutePositionAction(RotationTranslationGripAction):
    def __init__(self, tgt_ee_pose_key: str, tgt_grip_key: str, rotation_representation: RotRep | str = RotRep.QUAT):
        super().__init__(rotation_representation)
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.tgt_grip_key = tgt_grip_key

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self.tgt_ee_pose_key]
        rotations = transforms.recode_rotation(RotRep.QUAT, self.rot_rep, pose, slice=slice(3, None))
        names = [
            'rotation_0',
            'rotation_1',
            'rotation_2',
            'rotation_3',
            'translation_x',
            'translation_y',
            'translation_z',
            'grip',
        ]
        return transforms.concat(
            rotations,
            transforms.view(pose, slice(0, 3)),
            episode[self.tgt_grip_key],
            dtype=np.float32,
            names=names,
        )

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[: self.rot_rep.size].reshape(self.rot_rep.shape)
        rot = geom.Rotation.create_from(rotation, self.rot_rep)
        trans = action_vector[self.rot_rep.size : self.rot_rep.size + 3]
        return {
            'target_robot_position': geom.Transform3D(trans, rot),
            'target_grip': action_vector[self.rot_rep.size + 3],
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
        translations = transforms.pairwise(robot_trans, target_trans, np.subtract)

        # Grip: target_grip
        grips = episode[self.target_grip_key]

        return transforms.concat(rotations, translations, grips, dtype=np.float32)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[: self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size : self.rot_rep.size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(translation=tr_add, rotation=rot_mul),
            'target_grip': action_vector[self.rot_rep.size + 3],
        }
        return outputs


class RelativeRobotPositionAction(RotationTranslationGripAction):
    def __init__(
        self,
        offset_ns: int,
        rotation_representation: RotRep | str = RotRep.QUAT,
        pose_key: str = 'robot_state.ee_pose',
        grip_key: str = 'grip',
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset_ns` nanoseconds.

        Target_position_i = Pose_i ^ -1 * Pose_{i+offset_ns}
        Target_grip_i = Grip_{i+offset_ns}

        Args:
            offset_ns: (int) Time delta to look ahead, in nanoseconds.
            rotation_representation: (Rotation.Representation | str) The representation of the rotation.
        """
        super().__init__(rotation_representation)

        self.offset_ns = offset_ns
        self.pose_key = pose_key
        self.grip_key = grip_key

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self.pose_key]
        pose_future = transforms.TimeOffsets(pose, self.offset_ns)

        current_quat = transforms.view(pose, slice(3, None))
        future_quat = transforms.view(pose_future, slice(3, None))
        rotations = transforms.pairwise(
            current_quat,
            future_quat,
            partial(_relative_rot_vec, representation=self.rot_rep),
        )

        current_trans = transforms.view(pose, slice(0, 3))
        future_trans = transforms.view(pose_future, slice(0, 3))
        translations = transforms.pairwise(current_trans, future_trans, np.subtract)

        # Grip at future time
        grips_future = transforms.TimeOffsets(episode[self.grip_key], self.offset_ns)

        return transforms.concat(rotations, translations, grips_future, dtype=np.float32)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[: self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size : self.rot_rep.size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['reference_robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['reference_robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(translation=tr_add, rotation=rot_mul),
            'target_grip': action_vector[self.rot_rep.size + 3],
        }
        return outputs
