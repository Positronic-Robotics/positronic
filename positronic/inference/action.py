import abc

import torch

from positronic import geom
from positronic.utils.registration import umi_relative


def _convert_quat_to_tensor(q: geom.Rotation, representation: geom.Rotation.Representation | str) -> torch.Tensor:
    array = q.to(representation)

    return torch.from_numpy(array).flatten()


class ActionDecoder(abc.ABC):
    @abc.abstractmethod
    def encode_episode(self, episode_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode the episode data into a tensor. Used for creating the dataset for training.

        Args:
            episode_data: (dict[str, torch.Tensor]) Data of the entire episode.

        Returns:
            (torch.Tensor) The encoded episode data.
        """
        pass

    @abc.abstractmethod
    def decode(self, action_vector: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Decode the action vector into a dictionary of tensors. Used for inference.

        Args:
            action_vector: (torch.Tensor) Action vector to decode.
            inputs: (dict[str, torch.Tensor]) Additional inputs with things like current state.

        Returns:
            (dict[str, torch.Tensor]) The decoded action.
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

    def encode_episode(self, episode_data: dict[str, torch.Tensor]) -> torch.Tensor:
        rotations = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q in enumerate(episode_data['target_robot_position_quaternion']):
            q = geom.Rotation(*q)
            rotation = _convert_quat_to_tensor(q, self.rotation_representation)
            rotations[i] = rotation

        translations = episode_data['target_robot_position_translation']
        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translations, grips], dim=1)

    def decode(self, action_vector: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def encode_episode(self, episode_data: dict[str, torch.Tensor]) -> torch.Tensor:
        mtxs = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q_target in enumerate(episode_data['target_robot_position_quaternion']):
            q_target = geom.Rotation.from_quat(*q_target)
            q_current = geom.Rotation.from_quat(*episode_data['robot_position_quaternion'][i])
            q_relative = q_current.inv * q_target
            q_relative = geom.Rotation.from_quat(geom.normalise_quat(q_relative.as_quat))

            mtx = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            mtxs[i] = mtx.flatten()

        translation_diff = (episode_data['target_robot_position_translation'] -
                            episode_data['robot_position_translation'])

        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([mtxs, translation_diff, grips], dim=1)

    def decode(self, action_vector: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def encode_episode(self, episode_data: dict[str, torch.Tensor]) -> torch.Tensor:
        rotations = torch.zeros(len(episode_data['robot_position_quaternion']), self.rotation_size)
        translation_diff = -episode_data['robot_position_translation'].clone()
        grips = torch.zeros_like(episode_data['grip'])

        # TODO: make this vectorized
        for i, q_current in enumerate(episode_data['robot_position_quaternion']):
            if i + self.offset >= len(episode_data['robot_position_quaternion']):
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            q_current = geom.Rotation.from_quat(q_current)
            q_target = geom.Rotation.from_quat(episode_data['robot_position_quaternion'][i + self.offset])
            q_relative = q_current.inv * q_target
            q_relative = geom.Rotation.from_quat(geom.normalise_quat(q_relative.as_quat))

            rotation = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] += episode_data['robot_position_translation'][i + self.offset]
            grips[i] = episode_data['grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def encode_episode(self, episode_data: dict[str, torch.Tensor]) -> torch.Tensor:
        n_samples = len(episode_data['target_grip'])
        rotations = torch.zeros(n_samples, self.rotation_size)
        translation_diff = torch.zeros(n_samples, 3)
        grips = torch.zeros(n_samples)
        registration_transform = geom.Transform3D(
            translation=episode_data['registration_transform_translation'],
            rotation=geom.Rotation.from_quat(episode_data['registration_transform_quaternion'])
        )

        relative_trajectory = self._prepare(episode_data)

        # TODO: make this vectorized
        for i, q_relative in enumerate(relative_trajectory):
            if i + self.offset >= n_samples:
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            relative_registered = registration_transform.inv * q_relative * registration_transform
            q_relative_registered = geom.Rotation.from_quat(geom.normalise_quat(relative_registered.rotation.as_quat))

            rotation = _convert_quat_to_tensor(q_relative_registered, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] = torch.from_numpy(relative_registered.translation)
            grips[i] = episode_data['target_grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
