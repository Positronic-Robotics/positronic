import abc

import numpy as np

from positronic import geom
from positronic.dataset.core import Signal
from positronic.utils.registration import umi_relative


def _convert_quat_to_tensor(q: geom.Rotation, representation: geom.Rotation.Representation | str) -> np.ndarray:
    array = q.to(representation)

    return np.array(array).flatten()


class ActionDecoder(abc.ABC):
    @abc.abstractmethod
    def encode_episode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> np.ndarray:
        """
        Encode the episode data into a tensor. Used for creating the dataset for training.

        Args:
            signal_dict: (dict[str, Signal]) Data of the entire episode.
            timestamps: (np.ndarray) Timestamps of the signals.

        Returns:
            (np.ndarray) The encoded episode data.
        """
        pass

    @abc.abstractmethod
    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Decode the action vector into a dictionary of tensors. Used for inference.

        Args:
            action_vector: (np.ndarray) Action vector to decode.
            inputs: (dict[str, np.ndarray]) Additional inputs with things like current state.

        Returns:
            (dict[str, np.ndarray]) The decoded action.
        """
        pass

    @abc.abstractmethod
    def get_features(self) -> dict[str, dict]:
        """
        Get the features of the action.
        """
        pass


class RotationTranslationGripAction(ActionDecoder, abc.ABC):
    def __init__(self, rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT):
        rotation_representation = geom.Rotation.Representation(rotation_representation)

        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def get_features(self):
        return {
            'action': {
                'dtype': 'float32',
                'shape': (self.rotation_size + 4,),
                'names': [
                    *[f'rotation_{i}' for i in range(self.rotation_size)],
                    'translation_x',
                    'translation_y',
                    'translation_z',
                    'grip',
                ]
            },
        }


class AbsolutePositionAction(RotationTranslationGripAction):
    def __init__(self, rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT):
        super().__init__(rotation_representation)

    def encode_episode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> np.ndarray:
        quat = np.array([x[0] for x in signal_dict['target_robot_position_quaternion'].time[timestamps]])
        rotations = np.zeros((len(quat), self.rotation_size), dtype=np.float32)

        # TODO: make this vectorized
        for i, q in enumerate(quat):
            q = geom.Rotation(*q)
            rotation = _convert_quat_to_tensor(q, self.rotation_representation)
            rotations[i] = rotation

        translations = np.array([x[0] for x in signal_dict['target_robot_position_translation'].time[timestamps]])
        grips = np.array([x[0] for x in signal_dict['target_grip'].time[timestamps]])
        if grips.ndim == 1:
            grips = grips[:, np.newaxis]

        return np.concatenate([rotations, translations, grips], axis=1)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        rot = geom.Rotation.create_from(rotation, self.rotation_representation)

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=action_vector[self.rotation_size:self.rotation_size + 3],
                rotation=rot
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeTargetPositionAction(RotationTranslationGripAction):
    def __init__(self, rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT):
        super().__init__(rotation_representation)

    def encode_episode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> np.ndarray:
        quat = signal_dict['target_robot_position_quaternion'].time[timestamps]
        mtxs = np.zeros(len(quat), self.rotation_size)

        # TODO: make this vectorized
        for i, q_target in enumerate(quat):
            q_target = geom.Rotation.from_quat(*q_target)
            q_current = geom.Rotation.from_quat(*signal_dict['robot_position_quaternion'].time[timestamps][i])
            q_relative = q_current.inv * q_target
            q_relative = geom.Rotation.from_quat(geom.normalise_quat(q_relative.as_quat))

            mtx = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            mtxs[i] = mtx.flatten()

        translation_diff = (np.asarray(signal_dict['target_robot_position_translation'].time[timestamps])[:, 0] -
                            np.asarray(signal_dict['robot_position_translation'].time[timestamps])[:, 0])

        grips = np.asarray(signal_dict['target_grip'].time[timestamps])[:, 0]
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return np.concatenate([mtxs, translation_diff, grips], axis=1)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Rotation.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                rotation=rot_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeRobotPositionAction(RotationTranslationGripAction):
    def __init__(
            self,
            offset: int,
            rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset` timesteps.

        Target_position_i = Pose_i ^ -1 * Pose_{i+offset}
        Target_grip_i = Grip_{i+offset}

        Args:
            offset: (int) The number of timesteps to look ahead.
            rotation_representation: (Rotation.Representation | str) The representation of the rotation.
        """
        super().__init__(rotation_representation)

        self.offset = offset

    def encode_episode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> np.ndarray:
        quat = np.array([x[0] for x in signal_dict['robot_position_quaternion'].time[timestamps]])
        robot_position_translation = np.array(
            [x[0] for x in signal_dict['robot_position_translation'].time[timestamps]]
        )
        grip = np.array([x[0] for x in signal_dict['grip'].time[timestamps]])

        rotations = np.zeros((len(quat), self.rotation_size), dtype=np.float32)
        translation_diff = -robot_position_translation
        grips = np.zeros(len(quat), dtype=np.float32)

        # TODO: make this vectorized
        for i, q_current in enumerate(quat):
            if i + self.offset >= len(quat):
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = np.zeros(3)
                continue
            q_current = geom.Rotation.from_quat(q_current)
            q_target = geom.Rotation.from_quat(quat[i + self.offset])
            q_relative = q_current.inv * q_target
            q_relative = geom.Rotation.from_quat(geom.normalise_quat(q_relative.as_quat))

            rotation = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] += robot_position_translation[i + self.offset]
            grips[i] = grip[i + self.offset]

        if grips.ndim == 1:
            grips = grips[:, np.newaxis]

        return np.concatenate([rotations, translation_diff, grips], axis=1)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Rotation.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        rot_mul = geom.Rotation.from_quat(inputs['reference_robot_position_quaternion']) * q_diff
        rot_mul = geom.Rotation.from_quat(geom.normalise_quat(rot_mul.as_quat))

        tr_add = inputs['reference_robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                rotation=rot_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class UMIRelativeRobotPositionAction(RotationTranslationGripAction):
    def __init__(
            self,
            offset: int,
            rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT,
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset` timesteps.
        Target_position_i = Pose_i ^ -1 * Pose_{i+offset}
        Target_grip_i = Grip_{i+offset}
        Args:
            offset: (int) The number of timesteps to look ahead.
            rotation_representation: (Rotation.Representation | str) The representation of the rotation.
        """
        super().__init__(rotation_representation)

        self.offset = offset

    def _prepare(self, episode_data):
        left_trajectory = geom.trajectory.AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(episode_data['umi_left_translation'], episode_data['umi_left_quaternion'])
        ])

        right_trajectory = geom.trajectory.AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(episode_data['umi_right_translation'], episode_data['umi_right_quaternion'])
        ])

        return umi_relative(left_trajectory, right_trajectory)

    def encode_episode(self, signal_dict: dict[str, Signal], timestamps: np.ndarray) -> np.ndarray:
        n_samples = len(signal_dict['target_grip'].time[timestamps])
        rotations = np.zeros((n_samples, self.rotation_size), dtype=np.float32)
        translation_diff = np.zeros((n_samples, 3), dtype=np.float32)
        grips = np.zeros(n_samples, dtype=np.float32)
        registration_transform = geom.Transform3D(
            translation=np.asarray(signal_dict['registration_transform_translation'].time[timestamps])[:, 0],
            rotation=geom.Rotation.from_quat(
                np.asarray(signal_dict['registration_transform_quaternion'].time[timestamps])[:, 0]
            )
        )

        relative_trajectory = self._prepare(signal_dict)

        # TODO: make this vectorized
        for i, q_relative in enumerate(relative_trajectory):
            if i + self.offset >= n_samples:
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = np.zeros(3)
                continue
            relative_registered = registration_transform.inv * q_relative * registration_transform
            q_relative_registered = geom.Rotation.from_quat(geom.normalise_quat(relative_registered.rotation.as_quat))

            rotation = _convert_quat_to_tensor(q_relative_registered, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] = relative_registered.translation
            grips[i] = np.asarray(signal_dict['target_grip'].time[timestamps])[i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return np.concatenate([rotations, translation_diff, grips], axis=1)

    def decode(self, action_vector: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Rotation.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        diff_pose = geom.Transform3D(translation=tr_diff, rotation=q_diff)

        reference_pose = geom.Transform3D(
            inputs['reference_robot_position_translation'],
            geom.Rotation.from_quat(inputs['reference_robot_position_quaternion'])
        )

        new_pose = reference_pose * diff_pose

        outputs = {
            'target_robot_position': new_pose,
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs
