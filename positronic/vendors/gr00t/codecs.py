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
from positronic.dataset.transforms.episode import Derive, Identity
from positronic.policy.codec import Codec, lerobot_image, lerobot_state

RotRep = geom.Rotation.Representation


class GrootObservationCodec(Codec):
    """GR00T N1.6 observation encoder.

    For training (training_encoder): derives flat keys for each state component.
    For inference: encode() produces nested GR00T format (video/state/language).
    """

    def __init__(
        self,
        rotation_rep: RotRep | None = None,
        include_joints: bool = False,
        include_ee_pose: bool = True,
        image_size: tuple[int, int] = (224, 224),
        exterior_camera: str = 'image.exterior',
        wrist_camera: str = 'image.wrist',
        num_joints: int = 7,
    ):
        self._rotation_rep = rotation_rep
        self._include_joints = include_joints
        self._include_ee_pose = include_ee_pose
        self._image_size = image_size
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera
        self._num_joints = num_joints

        self._derive_transforms: dict[str, Any] = {
            'grip': self._derive_grip,
            'wrist_image': partial(self._derive_image, wrist_camera),
            'exterior_image_1': partial(self._derive_image, exterior_camera),
            'task': lambda ep: ep['task'] if 'task' in ep else '',
        }

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
            self._derive_transforms['ee_pose'] = self._derive_ee_pose
        if include_joints:
            state_meta['joint_position'] = {'start': 0, 'end': num_joints, 'original_key': 'joint_position'}
            lerobot_features['joint_position'] = lerobot_state(num_joints)
            self._derive_transforms['joint_position'] = self._derive_joints

        self._training_meta = {
            'gr00t_modality': {
                'state': state_meta,
                'video': {
                    'exterior_image_1': {'original_key': 'exterior_image_1'},
                    'wrist_image': {'original_key': 'wrist_image'},
                },
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

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return {}

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

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)


class _GrootActionModality(Codec):
    """Bridges GR00T modality-keyed actions and flat action vectors.

    Training: adds ``gr00t_modality.action`` metadata.
    Inference decode: converts GR00T's ``{action_key: ..., 'grip': ...}`` output
    into ``{'action': flat_vector}`` so the downstream action decoder can read it.
    """

    def __init__(self, modality: dict[str, Any], action_key: str):
        self._training_meta = {'gr00t_modality': {'action': modality}}
        self._action_key = action_key

    def encode(self, data):
        return data

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action_part = np.asarray(data[self._action_key], dtype=np.float32).reshape(-1)
        grip_part = np.asarray(data['grip'], dtype=np.float32).reshape(-1)
        return {'action': np.concatenate([action_part, grip_part])}

    @property
    def training_encoder(self):
        return Identity(meta=self._training_meta)


@cfn.config(rotation_rep=None, include_joints=False, include_ee_pose=True, num_joints=7)
def groot_obs(rotation_rep: str | None, include_joints: bool, include_ee_pose: bool, num_joints: int):
    """GR00T N1.6 observation encoder."""
    rot_rep = RotRep(rotation_rep) if rotation_rep else None
    return GrootObservationCodec(
        rotation_rep=rot_rep, include_joints=include_joints, include_ee_pose=include_ee_pose, num_joints=num_joints
    )


@cfn.config(action_key='ee_pose', action_dim=7)
def groot_action(base, action_key: str, action_dim: int):
    """Wrap an action codec with GR00T modality metadata and decode adapter.

    Composition is ``base | _GrootActionModality`` so that on decode (right-to-left)
    the modality adapter runs first, converting GR00T's modality-keyed output
    into a flat ``action`` vector that ``base`` can decode.
    """
    return base | _GrootActionModality(
        {action_key: {'start': 0, 'end': action_dim}, 'grip': {'start': action_dim, 'end': action_dim + 1}},
        action_key=action_key,
    )


_ee_action = groot_action.override(base=codecs.absolute_pos_action)
_rot6d_obs = groot_obs.override(rotation_rep='rot6d')
_rot6d_action = _ee_action.override(**{'base.rotation_rep': 'rot6d', 'action_dim': 9})

ee_quat = codecs.compose.override(obs=groot_obs, action=_ee_action)
ee_quat_joints = ee_quat.override(**{'obs.include_joints': True})
ee_rot6d = codecs.compose.override(obs=_rot6d_obs, action=_rot6d_action)
ee_rot6d_joints = ee_rot6d.override(**{'obs.include_joints': True})

_traj_action = _ee_action.override(base=codecs.traj_ee_action)
_rot6d_traj_action = _rot6d_action.override(base=codecs.traj_ee_action.override(rotation_rep='rot6d'))

ee_quat_traj = codecs.compose.override(obs=groot_obs, action=_traj_action, binarize_grip=('grip',))
ee_rot6d_traj = codecs.compose.override(obs=_rot6d_obs, action=_rot6d_traj_action, binarize_grip=('grip',))
ee_quat_joints_traj = ee_quat_traj.override(**{'obs.include_joints': True})
ee_rot6d_joints_traj = ee_rot6d_traj.override(**{'obs.include_joints': True})

joints_traj = codecs.compose.override(
    obs=groot_obs.override(include_joints=True, include_ee_pose=False),
    action=groot_action.override(
        base=codecs.absolute_joints_action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip'),
        action_key='joint_position',
    ),
    binarize_grip=('grip',),
)
