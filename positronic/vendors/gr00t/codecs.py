"""GR00T codecs: implementation classes and configuronic configs in one file."""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic import geom
from positronic.cfg import codecs
from positronic.dataset import transforms
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Signal
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Group, Identity
from positronic.drivers.roboarm import command
from positronic.policy.codec import Codec, lerobot_action, lerobot_image, lerobot_state

RotRep = geom.Rotation.Representation


class GrootCodec(Codec):
    """GR00T N1.6 codec: observation encoder + action decoder.

    For training (training_encoder): outputs flat dict with separate keys for each state component
    plus the action vector.
    For inference: encode() produces nested GR00T format, _decode_single() converts actions to
    robot commands.
    """

    def __init__(
        self,
        rotation_rep: RotRep | None = None,
        include_joints: bool = False,
        include_ee_pose: bool = True,
        image_size: tuple[int, int] = (224, 224),
        exterior_camera: str = 'image.exterior',
        wrist_camera: str = 'image.wrist',
        tgt_ee_pose_key: str = 'robot_commands.pose',
        tgt_grip_key: str = 'target_grip',
        tgt_joints_key: str | None = None,
        num_joints: int = 7,
    ):
        self._rotation_rep = rotation_rep
        self._action_rot_rep = rotation_rep or RotRep.QUAT
        self._include_joints = include_joints
        self._include_ee_pose = include_ee_pose
        self._image_size = image_size
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera
        self._tgt_ee_pose_key = tgt_ee_pose_key
        self._tgt_grip_key = tgt_grip_key
        self._tgt_joints_key = tgt_joints_key
        self._num_joints = num_joints
        self._joints_action = tgt_joints_key is not None

        self._derive_transforms: dict[str, Any] = {
            'grip': self._derive_grip,
            'wrist_image': partial(self._derive_image, wrist_camera),
            'exterior_image_1': partial(self._derive_image, exterior_camera),
            'action': self._derive_joints_action if self._joints_action else self._derive_ee_action,
            'task': lambda ep: ep['task'] if 'task' in ep else '',
        }
        if include_ee_pose:
            self._derive_transforms['ee_pose'] = self._derive_ee_pose
        if include_joints:
            self._derive_transforms['joint_position'] = self._derive_joints

        state_meta: dict[str, Any] = {'grip': {'start': 0, 'end': 1, 'original_key': 'grip'}}
        lerobot_features: dict[str, Any] = {
            'grip': lerobot_state(1),
            'wrist_image': lerobot_image(*image_size),
            'exterior_image_1': lerobot_image(*image_size),
        }

        if include_ee_pose:
            obs_ee_dim = rotation_rep.size + 3 if rotation_rep else 7
            state_meta['ee_pose'] = {'start': 0, 'end': obs_ee_dim, 'original_key': 'ee_pose'}
            lerobot_features['ee_pose'] = lerobot_state(obs_ee_dim)
        if include_joints:
            state_meta['joint_position'] = {'start': 0, 'end': num_joints, 'original_key': 'joint_position'}
            lerobot_features['joint_position'] = lerobot_state(num_joints)

        if self._joints_action:
            action_dim = num_joints
            action_modality: dict[str, Any] = {
                'joint_position': {'start': 0, 'end': num_joints},
                'grip': {'start': num_joints, 'end': num_joints + 1},
            }
        else:
            action_dim = (rotation_rep.size if rotation_rep else 4) + 3
            action_modality = {
                'ee_pose': {'start': 0, 'end': action_dim},
                'grip': {'start': action_dim, 'end': action_dim + 1},
            }
        lerobot_features['action'] = lerobot_action(action_dim + 1)

        self._training_meta = {
            'gr00t_modality': {
                'state': state_meta,
                'video': {
                    'exterior_image_1': {'original_key': 'exterior_image_1'},
                    'wrist_image': {'original_key': 'wrist_image'},
                },
                'action': action_modality,
            },
            'lerobot_features': lerobot_features,
        }

    def _derive_ee_pose(self, episode: Episode) -> Signal[Any]:
        pose = episode['robot_state.ee_pose']
        if self._rotation_rep is not None:
            pose = tf.recode_transform(RotRep.QUAT, self._rotation_rep, pose)
        return tf.astype(pose, np.float32)

    def _derive_grip(self, episode: Episode) -> Signal[Any]:
        def _reshape_to_1d(values):
            arr = np.asarray(values, dtype=np.float32)
            return arr.reshape(-1, 1)

        return transforms.Elementwise(episode['grip'], _reshape_to_1d)

    def _derive_joints(self, episode: Episode) -> Signal[Any]:
        return tf.astype(episode['robot_state.q'], np.float32)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def _derive_ee_action(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self._tgt_ee_pose_key]
        pose = transforms.recode_transform(RotRep.QUAT, self._action_rot_rep, pose)
        return transforms.concat(pose, episode[self._tgt_grip_key], dtype=np.float32)

    def _derive_joints_action(self, episode: Episode) -> Signal[np.ndarray]:
        joints = tf.astype(episode[self._tgt_joints_key], np.float32)
        return transforms.concat(joints, episode[self._tgt_grip_key], dtype=np.float32)

    def _encode_ee_pose(self, inputs: dict[str, Any]) -> np.ndarray:
        pose = np.asarray(inputs['robot_state.ee_pose'], dtype=np.float32).reshape(-1)
        if self._rotation_rep is not None:
            pose = geom.Transform3D.from_vector(pose, RotRep.QUAT).as_vector(self._rotation_rep).astype(np.float32)
        return pose

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        """Return a zero-filled encoded observation in GR00T's nested format."""
        w, h = self._image_size
        state: dict[str, Any] = {'grip': np.zeros((1, 1, 1), dtype=np.float32)}
        if self._include_ee_pose:
            ee_dim = self._rotation_rep.size + 3 if self._rotation_rep else 7
            state['ee_pose'] = np.zeros((1, 1, ee_dim), dtype=np.float32)
        if self._include_joints:
            state['joint_position'] = np.zeros((1, 1, self._num_joints), dtype=np.float32)
        return {
            'video': {
                'wrist_image': np.zeros((1, 1, h, w, 3), dtype=np.uint8),
                'exterior_image_1': np.zeros((1, 1, h, w, 3), dtype=np.uint8),
            },
            'state': state,
            'language': {'annotation.language.language_instruction': [['warmup']]},
        }

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        grip = np.asarray(inputs['grip'], dtype=np.float32).reshape(-1)
        state_dict: dict[str, Any] = {'grip': grip[np.newaxis, np.newaxis, ...]}

        if self._include_ee_pose:
            ee_pose = self._encode_ee_pose(inputs)
            state_dict['ee_pose'] = ee_pose[np.newaxis, np.newaxis, ...]
        if self._include_joints:
            joints = np.asarray(inputs['robot_state.q'], dtype=np.float32).reshape(-1)
            state_dict['joint_position'] = joints[np.newaxis, np.newaxis, ...]

        return {
            'video': {
                'wrist_image': self._encode_image(self._wrist_camera, inputs)[np.newaxis, np.newaxis, ...],
                'exterior_image_1': self._encode_image(self._exterior_camera, inputs)[np.newaxis, np.newaxis, ...],
            },
            'state': state_dict,
            'language': {'annotation.language.language_instruction': [[inputs.get('task', '')]]},
        }

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        target_grip = data['grip'].item()
        if self._joints_action:
            return {
                'robot_command': command.to_wire(command.JointPosition(positions=data['joint_position'])),
                'target_grip': target_grip,
            }
        target_pose = geom.Transform3D.from_vector(data['ee_pose'], self._action_rot_rep)
        return {
            'robot_command': command.to_wire(command.CartesianPosition(pose=target_pose)),
            'target_grip': target_grip,
        }

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Group(Derive(meta=self._training_meta, **self._derive_transforms), Identity())


@cfn.config(
    rotation_rep=None,
    include_joints=False,
    include_ee_pose=True,
    tgt_ee_pose_key='robot_commands.pose',
    tgt_grip_key='target_grip',
    tgt_joints_key=None,
    num_joints=7,
)
def groot(
    rotation_rep: str | None,
    include_joints: bool,
    include_ee_pose: bool,
    tgt_ee_pose_key: str,
    tgt_grip_key: str,
    tgt_joints_key: str | None,
    num_joints: int,
):
    """GR00T N1.6 codec."""
    rot_rep = RotRep(rotation_rep) if rotation_rep else None
    return GrootCodec(
        rotation_rep=rot_rep,
        include_joints=include_joints,
        include_ee_pose=include_ee_pose,
        tgt_ee_pose_key=tgt_ee_pose_key,
        tgt_grip_key=tgt_grip_key,
        tgt_joints_key=tgt_joints_key,
        num_joints=num_joints,
    )


ee_absolute = codecs.compose.override(obs=groot)
ee_joints = ee_absolute.override(**{'obs.include_joints': True})
ee_rot6d = ee_absolute.override(**{'obs.rotation_rep': 'rot6d'})
ee_rot6d_joints = ee_absolute.override(**{'obs.rotation_rep': 'rot6d', 'obs.include_joints': True})

_traj = {'obs.tgt_ee_pose_key': 'robot_state.ee_pose', 'obs.tgt_grip_key': 'grip', 'binarize_grip_keys': ('grip',)}
ee_absolute_traj = ee_absolute.override(**_traj)
ee_rot6d_traj = ee_rot6d.override(**_traj)
ee_joints_traj = ee_joints.override(**_traj)
ee_rot6d_joints_traj = ee_rot6d_joints.override(**_traj)

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_traj = codecs.compose.override(
    obs=groot.override(include_joints=True, include_ee_pose=False, tgt_joints_key='robot_state.q', tgt_grip_key='grip'),
    binarize_grip_keys=('grip',),
)
