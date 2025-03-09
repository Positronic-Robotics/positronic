import abc

import torch
import geom


def _convert_quat_to_tensor(q: geom.Rotation, representation: geom.Rotation.Representation | str) -> torch.Tensor:
    array = q.to(representation)

    return torch.from_numpy(array).flatten()


class ActionDecoder(abc.ABC):
    @abc.abstractmethod
    def encode_episode(self, episode_data):
        pass

    @abc.abstractmethod
    def decode(self, action_vector, inputs):
        pass


class AbsolutePositionAction(ActionDecoder):
    def __init__(self, rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT):
        rotation_representation = geom.Rotation.Representation(rotation_representation)

        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def encode_episode(self, episode_data):
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

    def decode(self, action_vector, inputs):
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


class RelativeTargetPositionAction(ActionDecoder):
    def __init__(self, rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT):
        rotation_representation = geom.Rotation.Representation(rotation_representation)

        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape

    def encode_episode(self, episode_data):
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

    def decode(self, action_vector, inputs):
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


class RelativeRobotPositionAction(ActionDecoder):
    def __init__(
            self,
            offset: int,
            rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset` timesteps.

        Target_position_i = Pose_i ^ -1 * Pose_i+offset
        Target_grip_i = Grip_i+offset

        Args:
            offset: (int) The number of timesteps to look ahead.
            rotation_representation: (Rotation.Representation | str) The representation of the rotation.
        """
        rotation_representation = geom.Rotation.Representation(rotation_representation)

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
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            q_current = geom.Rotation.from_quat(*q_current)
            q_target = geom.Rotation.from_quat(*episode_data['robot_position_quaternion'][i + self.offset])
            q_relative = q_current.inv * q_target
            q_relative = geom.Rotation.from_quat(geom.normalise_quat(q_relative.as_quat))

            rotation = _convert_quat_to_tensor(q_relative, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] += episode_data['robot_position_translation'][i + self.offset]
            grips[i] = episode_data['grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
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


from adhoc.validate_trajectory import umi_relative
from positronic.tools.registration import AbsoluteTrajectory, RelativeTrajectory


class UMIRelativeRobotPositionAction(ActionDecoder):
    def __init__(
            self,
            offset: int,
            rotation_representation: geom.Rotation.Representation | str = geom.Rotation.Representation.QUAT,
            registration_transform: geom.Transform3D | None = None,
    ):
        """
        Action that represents the relative position between the current robot position and the robot position
        after `offset` timesteps.

        Target_position_i = Pose_i ^ -1 * Pose_i+offset
        Target_grip_i = Grip_i+offset

        Args:
            offset: (int) The number of timesteps to look ahead.
            rotation_representation: (Rotation.Representation | str) The representation of the rotation.
        """
        rotation_representation = geom.Rotation.Representation(rotation_representation)

        self.offset = offset
        self.rotation_representation = rotation_representation
        self.rotation_size = rotation_representation.size
        self.rotation_shape = rotation_representation.shape
        self.registration_transform = registration_transform
        self.base_ee_pose = geom.Transform3D(
            translation=[0.441147, -0.00327101, 0.797503],
            rotation=geom.Rotation.from_quat([0.966373, -0.00267326, 0.257121, -0.00223433])
        )
        self.previous_pose = None

    def _prepare(self, episode_data):
        left_trajectory = AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(episode_data['umi_left_translation'], episode_data['umi_left_quaternion'])
        ])

        right_trajectory = AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(episode_data['umi_right_translation'], episode_data['umi_right_quaternion'])
        ])

        return umi_relative(left_trajectory, right_trajectory)


    def encode_episode(self, episode_data):
        rotations = torch.zeros(len(episode_data['robot_position_quaternion']), self.rotation_size)
        translation_diff = torch.zeros_like(episode_data['robot_position_translation'])
        grips = torch.zeros_like(episode_data['grip'])

        relative_trajectory = self._prepare(episode_data)

        # TODO: make this vectorized
        for i, q_relative in enumerate(relative_trajectory):
            if i + self.offset >= len(episode_data['robot_position_quaternion']):
                rotations[i] = _convert_quat_to_tensor(geom.Rotation(1, 0, 0, 0), self.rotation_representation)
                translation_diff[i] = torch.zeros(3)
                continue
            relative_registered = self.registration_transform.inv * q_relative * self.registration_transform
            q_relative_registered = geom.Rotation.from_quat(geom.normalise_quat(relative_registered.rotation.as_quat))

            rotation = _convert_quat_to_tensor(q_relative_registered, self.rotation_representation)
            rotations[i] = rotation
            translation_diff[i] = torch.from_numpy(relative_registered.translation)
            grips[i] = episode_data['grip'][i + self.offset]

        if grips.ndim == 1:
            grips = grips.unsqueeze(1)

        return torch.cat([rotations, translation_diff, grips], dim=1)

    def decode(self, action_vector, inputs):
        print("===================== STEP ========================")
        print()
        rotation = action_vector[:self.rotation_size].reshape(self.rotation_shape)
        q_diff = geom.Rotation.create_from(rotation, self.rotation_representation)
        tr_diff = action_vector[self.rotation_size:self.rotation_size + 3]

        diff_pose = geom.Transform3D(translation=tr_diff, rotation=q_diff)

        reference_pose = geom.Transform3D(
            inputs['reference_robot_position_translation'],
            geom.Rotation.from_quat(inputs['reference_robot_position_quaternion'])
        )

        # if self.previous_pose is not None:
        #     print("Diff",self.previous_pose, diff_pose)
        #     n_diff_pose = self.previous_pose.inv * diff_pose
        #     self.previous_pose = diff_pose
        #     diff_pose = n_diff_pose
        # else:
        #     self.previous_pose = diff_pose

        # new_pose = reference_pose * (reference_pose.inv * diff_pose * reference_pose).inv

        print("Reference pose", reference_pose)
        print("Diff pose", diff_pose)
        new_pose = reference_pose * diff_pose
        # self.previous_pose = diff_pose
        # rot_mul = geom.Rotation.from_quat(geom.normalise_quat(new_pose.rotation.as_quat))
        # tr_add = new_pose.translation - tr_diff
        # print("Reference pose", reference_pose)
        # print("Diff pose", diff_pose)
        # print("New pose", new_pose)

        # diff_pose = geom.Transform3D()
        outputs = {
            'target_robot_position': new_pose,
            'target_grip': action_vector[self.rotation_size + 3]
        }
        return outputs
