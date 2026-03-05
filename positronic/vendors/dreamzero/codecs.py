"""DreamZero codecs: observation encoder for roboarena inference format."""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic.cfg import codecs
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive
from positronic.policy.codec import Codec, lerobot_image, lerobot_state

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 176


def _reshape_grip(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


class DreamZeroObservationCodec(Codec):
    """Observation encoder for DreamZero's roboarena inference format.

    For training (training_encoder): outputs flat LeRobot keys (state.joint_position,
    state.gripper_position, action.joint_position, action.gripper_position, video columns).
    For inference (encode): outputs roboarena keys (observation/joint_position, etc.)
    with images resized to 320x176 and session_id for stateful frame tracking.
    """

    def __init__(
        self,
        wrist_camera: str = 'image.wrist',
        exterior_camera_1: str = 'image.exterior',
        exterior_camera_2: str = 'image.exterior2',
        image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
        action_joints_key: str | None = None,
        action_grip_key: str | None = None,
    ):
        self._wrist_camera = wrist_camera
        self._exterior_camera_1 = exterior_camera_1
        self._exterior_camera_2 = exterior_camera_2
        self._image_size = image_size

        w, h = image_size
        self._derive_transforms: dict[str, Any] = {
            'state.joint_position': self._derive_joint_position,
            'state.gripper_position': self._derive_gripper_position,
            'video.wrist_image_left': partial(self._derive_image, wrist_camera),
            'video.exterior_image_1_left': partial(self._derive_image, exterior_camera_1),
            'video.exterior_image_2_left': partial(self._derive_image, exterior_camera_2),
            'task': lambda ep: ep.get('task', ''),
        }

        lerobot_features: dict[str, Any] = {
            'state.joint_position': lerobot_state(7),
            'state.gripper_position': lerobot_state(1),
            'video.wrist_image_left': lerobot_image(w, h),
            'video.exterior_image_1_left': lerobot_image(w, h),
            'video.exterior_image_2_left': lerobot_image(w, h),
        }

        gr00t_modality: dict[str, Any] = {
            'state': {
                'joint_position': {'start': 0, 'end': 7, 'original_key': 'state.joint_position'},
                'gripper_position': {'start': 0, 'end': 1, 'original_key': 'state.gripper_position'},
            },
            'video': {
                'wrist_image_left': {'original_key': 'video.wrist_image_left'},
                'exterior_image_1_left': {'original_key': 'video.exterior_image_1_left'},
                'exterior_image_2_left': {'original_key': 'video.exterior_image_2_left'},
            },
            'annotation': {
                'language.language_instruction': {'original_key': 'task_index'},
                'language.language_instruction_2': {'original_key': 'task_index'},
                'language.language_instruction_3': {'original_key': 'task_index'},
            },
        }

        if action_joints_key is not None and action_grip_key is not None:
            self._derive_transforms['action.joint_position'] = partial(self._derive_action_joints, action_joints_key)
            self._derive_transforms['action.gripper_position'] = partial(self._derive_action_grip, action_grip_key)
            lerobot_features['action.joint_position'] = lerobot_state(7)
            lerobot_features['action.gripper_position'] = lerobot_state(1)
            gr00t_modality['action'] = {
                'joint_position': {'start': 0, 'end': 7},
                'gripper_position': {'start': 0, 'end': 1},
            }

        self._training_meta: dict[str, Any] = {'lerobot_features': lerobot_features, 'gr00t_modality': gr00t_modality}

    def _derive_joint_position(self, episode: Episode) -> Signal[Any]:
        return transforms.astype(episode['robot_state.q'], np.float32)

    def _derive_gripper_position(self, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode['grip'], _reshape_grip)

    def _derive_action_joints(self, key: str, episode: Episode) -> Signal[Any]:
        return transforms.astype(episode[key], np.float32)

    def _derive_action_grip(self, key: str, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode[key], _reshape_grip)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        if input_key not in episode:
            ref = episode['robot_state.q']
            return transforms.Elementwise(ref, lambda _: np.zeros((h, w, 3), dtype=np.uint8))
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        w, h = self._image_size
        frame = inputs.get(input_key)
        if frame is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, np.asarray(frame))

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        joint_pos = np.asarray(inputs['robot_state.q'], dtype=np.float32).reshape(-1)
        grip = np.asarray(inputs['grip'], dtype=np.float32).reshape(-1)

        obs: dict[str, Any] = {
            'observation/joint_position': joint_pos,
            'observation/gripper_position': grip,
            'observation/wrist_image_left': self._encode_image(self._wrist_camera, inputs),
            'observation/exterior_image_0_left': self._encode_image(self._exterior_camera_1, inputs),
            'observation/exterior_image_1_left': self._encode_image(self._exterior_camera_2, inputs),
        }

        if 'task' in inputs:
            obs['prompt'] = inputs['task']

        return obs

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        w, h = self._image_size
        return {
            'observation/joint_position': np.zeros(7, dtype=np.float32),
            'observation/gripper_position': np.zeros(1, dtype=np.float32),
            'observation/wrist_image_left': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/exterior_image_0_left': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/exterior_image_1_left': np.zeros((h, w, 3), dtype=np.uint8),
            'prompt': 'warmup',
        }

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)


@cfn.config(
    wrist_camera='image.wrist',
    exterior_camera_1='image.exterior',
    exterior_camera_2='image.exterior2',
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    action_joints_key=None,
    action_grip_key=None,
)
def dreamzero_obs(
    wrist_camera: str,
    exterior_camera_1: str,
    exterior_camera_2: str,
    image_size: tuple[int, int],
    action_joints_key: str | None,
    action_grip_key: str | None,
):
    """DreamZero observation codec (3 cameras + joint state)."""
    return DreamZeroObservationCodec(
        wrist_camera=wrist_camera,
        exterior_camera_1=exterior_camera_1,
        exterior_camera_2=exterior_camera_2,
        image_size=tuple(image_size),
        action_joints_key=action_joints_key,
        action_grip_key=action_grip_key,
    )


# Composed codec: DreamZero observation + absolute joints action + timing
_action = codecs.absolute_joints_action.override(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip')
joints = codecs.compose.override(obs=dreamzero_obs, action=_action, fps=4.0)

_traj_obs = dreamzero_obs.override(action_joints_key='robot_state.q', action_grip_key='grip')
_traj_action = _action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip')
joints_traj = codecs.compose.override(obs=_traj_obs, action=_traj_action, fps=4.0)
