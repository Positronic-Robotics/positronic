import abc
from functools import partial
from abc import abstractmethod
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

    def transform(self, name: str, episode: transforms.Episode) -> Signal[Any] | Any:
        if name != 'action':
            raise ValueError(f"Unknown action key: {name}")
        return self.encode_episode(episode)

    @abstractmethod
    def encode_episode(self, episode: Episode) -> Signal[Any]:
        pass

    @abstractmethod
    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_features(self) -> dict[str, dict]:
        pass


class RotationTranslationGripAction(ActionDecoder, abc.ABC):

    def __init__(self, rotation_representation: RotRep | str = RotRep.QUAT):
        self.rot_rep = RotRep(rotation_representation)
        # self.rotation_shape = self.rot_rep.shape

    def get_features(self):
        return {
            'action': {
                'dtype':
                'float32',
                'shape': (self.rot_rep.size + 4, ),
                'names': [
                    *[f'rotation_{i}' for i in range(self.rot_rep.size)], 'translation_x', 'translation_y',
                    'translation_z', 'grip'
                ]
            }
        }


class AbsolutePositionAction(RotationTranslationGripAction):

    def __init__(self, rotation_representation: RotRep | str = RotRep.QUAT):
        super().__init__(rotation_representation)

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        rotations = transforms.recode_rotation(RotRep.QUAT, self.rot_rep, episode['target_robot_position_quaternion'])
        names = [
            'rotation_0', 'rotation_1', 'rotation_2', 'rotation_3', 'translation_x', 'translation_y', 'translation_z',
            'grip'
        ]
        return transforms.concat(rotations,
                                 episode['target_robot_position_translation'],
                                 episode['target_grip'],
                                 dtype=np.float32,
                                 names=names)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[:self.rot_rep.size].reshape(self.rot_rep.shape)
        rot = geom.Rotation.create_from(rotation, self.rot_rep)
        trans = action_vector[self.rot_rep.size:self.rot_rep.size + 3]
        return {
            'target_robot_position': geom.Transform3D(trans, rot),
            'target_grip': action_vector[self.rot_rep.size + 3]
        }


class RelativeTargetPositionAction(RotationTranslationGripAction):

    def __init__(self, rotation_representation: RotRep | str = RotRep.QUAT):
        super().__init__(rotation_representation)

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        # Rotation difference: q_cur.inv * q_target, recoded
        rotations = transforms.pairwise(
            episode['robot_position_quaternion'],
            episode['target_robot_position_quaternion'],
            partial(_relative_rot_vec, representation=self.rot_rep),
        )

        # Translation difference: target - current
        translations = transforms.pairwise(
            episode['robot_position_translation'],
            episode['target_robot_position_translation'],
            np.subtract,
        )

        # Grip: target_grip
        grips = episode['target_grip']

        return transforms.concat(rotations, translations, grips, dtype=np.float32)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[:self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size:self.rot_rep.size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(translation=tr_add, rotation=rot_mul),
            'target_grip': action_vector[self.rot_rep.size + 3]
        }
        return outputs


class RelativeRobotPositionAction(RotationTranslationGripAction):

    def __init__(self, offset_ns: int, rotation_representation: RotRep | str = RotRep.QUAT):
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

    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        # Future quaternions and translations at t + offset_ns
        q_future = transforms.TimeOffsets(episode['robot_position_quaternion'], self.offset_ns)
        t_future = transforms.TimeOffsets(episode['robot_position_translation'], self.offset_ns)

        # Rotation: q_cur.inv * q_future
        rotations = transforms.pairwise(episode['robot_position_quaternion'], q_future,
                                        partial(_relative_rot_vec, representation=self.rot_rep))

        # Translation: t_future - t_current
        translations = transforms.pairwise(episode['robot_position_translation'], t_future, np.subtract)

        # Grip at future time
        grips_future = transforms.TimeOffsets(episode['grip'], self.offset_ns)

        return transforms.concat(rotations, translations, grips_future, dtype=np.float32)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rotation = action_vector[:self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size:self.rot_rep.size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['reference_robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['reference_robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(translation=tr_add, rotation=rot_mul),
            'target_grip': action_vector[self.rot_rep.size + 3]
        }
        return outputs
