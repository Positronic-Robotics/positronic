from functools import partial

import numpy as np

from positronic import geom
from positronic.dataset import transforms
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Signal
from positronic.dataset.transforms.episode import Derive, Group, Identity
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.ik import ik_joints_from_episode
from positronic.policy.codec import Codec, lerobot_action

RotRep = geom.Rotation.Representation


def _convert_quat_to_array(q: geom.Rotation, representation: RotRep | str) -> np.ndarray:
    return q.to(representation).reshape(-1)


def _relative_rot_vec(q_current: np.ndarray, q_target: np.ndarray, representation: RotRep | str) -> np.ndarray:
    r_cur = geom.Rotation.from_quat(q_current)
    r_tgt = geom.Rotation.from_quat(q_target)
    rel = r_cur.inv * r_tgt
    return _convert_quat_to_array(rel, representation)


class AbsolutePositionAction(Codec):
    def __init__(self, tgt_ee_pose_key: str, tgt_grip_key: str, rotation_rep: RotRep | str = RotRep.QUAT):
        self.rot_rep = RotRep(rotation_rep)
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.tgt_grip_key = tgt_grip_key

        ee_dim = self.rot_rep.size + 3
        self._training_meta = {'lerobot_features': {'action': lerobot_action(ee_dim + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action_vector = data['action']
        target_pose = geom.Transform3D.from_vector(action_vector[:-1], self.rot_rep)
        target_grip = action_vector[-1].item()
        return {'robot_command': command.CartesianPosition(pose=target_pose), 'target_grip': target_grip}

    def _encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self.tgt_ee_pose_key]
        pose = transforms.recode_transform(RotRep.QUAT, self.rot_rep, pose)
        return transforms.concat(pose, episode[self.tgt_grip_key], dtype=np.float32)

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, action=self._encode_episode)


class AbsoluteJointsAction(Codec):
    def __init__(self, tgt_joints_key: str, tgt_grip_key: str, num_joints: int = 7):
        self.tgt_joints_key = tgt_joints_key
        self.tgt_grip_key = tgt_grip_key
        self.num_joints = num_joints

        self._training_meta = {'lerobot_features': {'action': lerobot_action(num_joints + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action_vector = data['action']
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        joint_positions = action_vector[: self.num_joints]
        target_grip = action_vector[-1].item()
        return {'robot_command': command.JointPosition(positions=joint_positions), 'target_grip': target_grip}

    def _encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        return transforms.concat(episode[self.tgt_joints_key], episode[self.tgt_grip_key], dtype=np.float32)

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, action=self._encode_episode)


class IKJointsAction(Codec):
    """Signal-level codec that replaces EE pose targets with joint targets via IK.

    Training: replaces ``tgt_ee_pose_key`` with ``tgt_joints_key`` in the episode.
    Inference: pass-through (robot driver handles IK at runtime).
    Compose with AbsoluteJointsAction for inference decoding.
    """

    def __init__(
        self,
        solver_cls,
        *,
        tgt_ee_pose_key='robot_commands.pose',
        current_q_key='robot_state.q',
        tgt_joints_key='robot_commands.joints',
    ):
        self.solver_cls = solver_cls
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.current_q_key = current_q_key
        self.tgt_joints_key = tgt_joints_key

    def encode(self, data):
        return data

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return data

    def _derive_joints(self, episode: Episode):
        return ik_joints_from_episode(episode, self.solver_cls, self.tgt_ee_pose_key, self.current_q_key)

    @property
    def training_encoder(self):
        return Group(Derive(**{self.tgt_joints_key: self._derive_joints}), Identity(remove=[self.tgt_ee_pose_key]))


class RelativePositionAction(Codec):
    def __init__(
        self,
        rotation_rep: RotRep | str = RotRep.QUAT,
        robot_pose_key: str = 'robot_state.ee_pose',
        target_pose_key: str = 'robot_commands.pose',
        target_grip_key: str = 'target_grip',
    ):
        self.rot_rep = RotRep(rotation_rep)
        self.robot_pose_key = robot_pose_key
        self.target_pose_key = target_pose_key
        self.target_grip_key = target_grip_key

        ee_dim = self.rot_rep.size + 3
        self._training_meta = {'lerobot_features': {'action': lerobot_action(ee_dim + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action_vector = data['action']
        rotation = action_vector[: self.rot_rep.size].reshape(self.rot_rep.shape)
        q_diff = geom.Rotation.create_from(rotation, self.rot_rep)
        tr_diff = action_vector[self.rot_rep.size : self.rot_rep.size + 3]

        robot_pose = context['robot_state.ee_pose']

        rot_mul = geom.Rotation.from_quat(robot_pose[3:7]) * q_diff
        tr_add = robot_pose[0:3] + tr_diff

        target_pose = geom.Transform3D(translation=tr_add, rotation=rot_mul)
        target_grip = action_vector[self.rot_rep.size + 3].item()
        return {'robot_command': command.CartesianPosition(pose=target_pose), 'target_grip': target_grip}

    def _encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        robot_pose = episode[self.robot_pose_key]
        target_pose = episode[self.target_pose_key]

        robot_quat = transforms.view(robot_pose, slice(3, None))
        target_quat = transforms.view(target_pose, slice(3, None))
        rotations = transforms.pairwise(
            robot_quat, target_quat, partial(_relative_rot_vec, representation=self.rot_rep)
        )

        robot_trans = transforms.view(robot_pose, slice(0, 3))
        target_trans = transforms.view(target_pose, slice(0, 3))
        translations = transforms.pairwise(target_trans, robot_trans, np.subtract)

        grips = episode[self.target_grip_key]

        return transforms.concat(rotations, translations, grips, dtype=np.float32)

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, action=self._encode_episode)


class JointDeltaAction(Codec):
    """DROID-style joint velocity action decoder (inference only)."""

    RELATIVE_MAX_JOIN_DELTA = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    MAX_JOINT_DELTA = RELATIVE_MAX_JOIN_DELTA.max()
    MAX_JONIT_VEL = RELATIVE_MAX_JOIN_DELTA / MAX_JOINT_DELTA

    def __init__(self, num_joints: int = 7):
        self.num_joints = num_joints

        self._training_meta = {'lerobot_features': {'action': lerobot_action(num_joints + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action_vector = data['action']
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        action_vector = action_vector.clip(-1.0, 1.0)
        velocities = action_vector[: self.num_joints]
        grip = 1.0 if action_vector[self.num_joints].item() > 0.5 else 0.0

        max_vel_norm = (np.abs(velocities) / self.MAX_JONIT_VEL).max()
        if max_vel_norm > 1.0:
            velocities = velocities / max_vel_norm

        return {'robot_command': command.JointDelta(velocities=velocities * self.MAX_JOINT_DELTA), 'target_grip': grip}
