import torch

import geom


def _convert_quat_to_tensor(q: geom.Quaternion, representation: geom.RotationRepresentation | str) -> torch.Tensor:
    array = q.to(representation)

    return torch.from_numpy(array).flatten()


class AbsolutePositionAction:
    def __init__(self, rotation_representation: geom.RotationRepresentation | str = geom.RotationRepresentation.QUAT):
        rotation_representation = geom.RotationRepresentation(rotation_representation)

        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q in enumerate(episode_data['target_robot_position_quaternion']):
            q = geom.Quaternion(*q)
            rotation = _convert_quat_to_tensor(q, self.rotation_representation)
            rotations[i] = rotation

        translations = episode_data['target_robot_position_translation']
        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translations, grips], dim=1)

    def decode(self, action_vector, inputs):
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q = geom.Quaternion.create_from(rotation, self.rotation_representation)

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=action_vector[self.rotation_size:self.rotation_size + 3],
                quaternion=q
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeTargetPositionAction:
    def __init__(self, rotation_representation: geom.RotationRepresentation | str = geom.RotationRepresentation.QUAT):
        rotation_representation = geom.RotationRepresentation(rotation_representation)

        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def encode_episode(self, episode_data):
        mtxs = torch.zeros(len(episode_data['target_robot_position_quaternion']), self.rotation_size)

        # TODO: make this vectorized
        for i, q_target in enumerate(episode_data['target_robot_position_quaternion']):
            q_target = geom.Quaternion(*q_target)
            q_current = geom.Quaternion(*episode_data['robot_position_quaternion'][i])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            mtx = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            mtxs[i] = mtx.flatten()

        translation_diff = episode_data['target_robot_position_translation'] - episode_data['robot_position_translation']

        grips = episode_data['target_grip']
        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([mtxs, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Quaternion.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        q_mul = geom.quat_mul(inputs['robot_position_quaternion'], q_diff)
        q_mul = geom.normalise_quat(q_mul)

        tr_add = inputs['robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                quaternion=q_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs


class RelativeRobotPositionAction:
    def __init__(
            self,
            offset: int,
            rotation_representation: geom.RotationRepresentation | str = geom.RotationRepresentation.QUAT
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset` timesteps.

        Target_position_i = Pose_i ^ -1 * Pose_i+offset
        Target_grip_i = Grip_i+offset

        Args:
            offset: (int) The number of timesteps to look ahead.
            rotation_representation: (RotationRepresentation | str) The representation of the rotation.
        """
        rotation_representation = geom.RotationRepresentation(rotation_representation)

        self.offset = offset
        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['robot_position_quaternion']), self.rotation_size)
        translation_diff = -episode_data['robot_position_translation'].clone()
        grips = torch.zeros_like(episode_data['grip'])

        # TODO: make this vectorized
        for i, q_current in enumerate(episode_data['robot_position_quaternion']):
            if i + self.offset >= len(episode_data['robot_position_quaternion']):
                rotations[i] = _convert_quat_to_tensor(geom.Quaternion(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            q_current = geom.Quaternion(*q_current)
            q_target = geom.Quaternion(*episode_data['robot_position_quaternion'][i + self.offset])
            q_relative = geom.quat_mul(q_current.inv, q_target)
            q_relative = geom.normalise_quat(q_relative)

            rotation = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] += episode_data['robot_position_translation'][i + self.offset]
            grips[i] = episode_data['grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Quaternion.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        q_mul = geom.quat_mul(inputs['reference_robot_position_quaternion'], q_diff)
        q_mul = geom.normalise_quat(q_mul)

        tr_add = inputs['reference_robot_position_translation'] + tr_diff

        outputs = {
            'target_robot_position': geom.Transform3D(
                translation=tr_add,
                quaternion=q_mul
            ),
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs
