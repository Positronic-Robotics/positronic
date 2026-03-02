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
IMAGE_HEIGHT = 180


def _reshape_grip(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


class DreamZeroObservationCodec(Codec):
    """Observation encoder for DreamZero's roboarena inference format.

    For training (training_encoder): outputs flat LeRobot keys (state.joint_position,
    state.gripper_position, action.joint_position, action.gripper_position, video columns).
    For inference (encode): outputs roboarena keys (observation/joint_position, etc.)
    with images resized to 320x180 and session_id for stateful frame tracking.
    """

    def __init__(
        self,
        wrist_camera: str = 'image.wrist',
        exterior_camera_1: str = 'image.exterior',
        exterior_camera_2: str = 'image.exterior2',
        image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    ):
        self._wrist_camera = wrist_camera
        self._exterior_camera_1 = exterior_camera_1
        self._exterior_camera_2 = exterior_camera_2
        self._image_size = image_size

        w, h = image_size
        self._derive_transforms: dict[str, Any] = {
            'state.joint_position': self._derive_joint_position,
            'state.gripper_position': self._derive_gripper_position,
            'wrist_image': partial(self._derive_image, wrist_camera),
            'exterior_image_1': partial(self._derive_image, exterior_camera_1),
            'exterior_image_2': partial(self._derive_image, exterior_camera_2),
            'task': lambda ep: ep.get('task', ''),
        }

        self._training_meta: dict[str, Any] = {
            'lerobot_features': {
                'state.joint_position': lerobot_state(7),
                'state.gripper_position': lerobot_state(1),
                'wrist_image': lerobot_image(w, h),
                'exterior_image_1': lerobot_image(w, h),
                'exterior_image_2': lerobot_image(w, h),
            }
        }

    def _derive_joint_position(self, episode: Episode) -> Signal[Any]:
        return transforms.astype(episode['robot_state.q'], np.float32)

    def _derive_gripper_position(self, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode['grip'], _reshape_grip)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
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


@cfn.config(wrist_camera='image.wrist', exterior_camera_1='image.exterior', exterior_camera_2='image.exterior2')
def dreamzero_obs(wrist_camera: str, exterior_camera_1: str, exterior_camera_2: str):
    """DreamZero observation codec (3 cameras + joint state)."""
    return DreamZeroObservationCodec(
        wrist_camera=wrist_camera, exterior_camera_1=exterior_camera_1, exterior_camera_2=exterior_camera_2
    )


# Composed codec: DreamZero observation + absolute joints action + timing
_action = codecs.absolute_joints_action.override(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip')
joints = codecs.compose.override(obs=dreamzero_obs, action=_action, fps=4.0)

_traj = {'action.tgt_joints_key': 'robot_state.q', 'action.tgt_grip_key': 'grip'}
joints_traj = joints.override(**_traj)
