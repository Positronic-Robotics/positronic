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


class ActionDecoder(transforms.KeyFuncEpisodeTransform):
    def __init__(self):
        super().__init__(action=self.encode_episode)

    @abstractmethod
    def encode_episode(self, episode: Episode) -> Signal[Any]:
        pass

    @abstractmethod
    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
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

        robot_pose = inputs['robot_state.ee_pose']

        rot_mul = geom.Rotation.from_quat(robot_pose[3:7]) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = robot_pose[0:3] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(translation=tr_add, rotation=rot_mul),
            'target_grip': action_vector[self.rot_rep.size + 3],
        }
        return outputs
