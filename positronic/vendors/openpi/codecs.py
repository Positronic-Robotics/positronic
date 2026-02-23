"""OpenPI codecs (observation encoder + action decoder pairs).

OpenPI has different key format expectations for training vs inference:

- **Training (LeRobot format)**: Dot-separated keys like `observation.state`, `observation.images.left`.
  This is what LeRobot datasets use and what OpenPI training code consumes.

- **Inference (OpenPI format)**: Slash-separated keys like `observation/state`, `observation/image`.
  This is what OpenPI's policy classes (e.g., `positronic_policy.py`) expect at inference time.

The `OpenpiObservationCodec` class handles both cases:
- `encode()` (inference): Produces OpenPI-compatible format with slash-separated keys
- `training_encoder` (training): Produces LeRobot-compatible format with dot-separated keys

Note: `droid` codec is inference-only, designed to work with pretrained DROID models.
"""

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


class OpenpiObservationCodec(Codec):
    """Observation encoder that outputs LeRobot keys for training, OpenPI keys for inference."""

    def __init__(
        self,
        state_features: dict[str, int],
        exterior_camera: str = 'image.exterior',
        wrist_camera: str = 'image.wrist',
        image_size: tuple[int, int] = (224, 224),
    ):
        self._state_features = state_features
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera
        self._image_size = image_size

        self._derive_transforms = {
            'observation.state': self._derive_state,
            'observation.images.left': partial(self._derive_image, wrist_camera),
            'observation.images.side': partial(self._derive_image, exterior_camera),
            'task': lambda ep: ep['task'] if 'task' in ep else '',
        }

        state_dim = sum(state_features.values())
        w, h = image_size
        self._training_meta: dict[str, Any] = {
            'lerobot_features': {
                'observation.state': lerobot_state(state_dim, list(state_features.keys())),
                'observation.images.left': lerobot_image(w, h),
                'observation.images.side': lerobot_image(w, h),
            }
        }

    def _derive_state(self, episode: Episode) -> Signal[Any]:
        return transforms.concat(*[episode[key] for key in self._state_features], dtype=np.float32)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return {}

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        state_parts: list[np.ndarray] = []
        for feature_key in self._state_features:
            if feature_key not in inputs:
                raise KeyError(f"Missing state input '{feature_key}', available keys: {list(inputs.keys())}")
            state_parts.append(np.asarray(inputs[feature_key], dtype=np.float32).reshape(-1))

        obs: dict[str, Any] = {
            'observation/state': np.concatenate(state_parts) if state_parts else np.empty((0,), dtype=np.float32),
            'observation/wrist_image': self._encode_image(self._wrist_camera, inputs),
            'observation/image': self._encode_image(self._exterior_camera, inputs),
        }
        if 'task' in inputs:
            obs['prompt'] = inputs['task']
        return obs

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        if input_key not in inputs:
            raise KeyError(f"Missing image input '{input_key}', available keys: {list(inputs.keys())}")
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        """Return a zero-filled encoded observation in OpenPI's slash-separated format."""
        state_dim = sum(self._state_features.values())
        w, h = self._image_size
        return {
            'observation/state': np.zeros(state_dim, dtype=np.float32),
            'observation/wrist_image': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/image': np.zeros((h, w, 3), dtype=np.uint8),
            'prompt': 'warmup',
        }

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)


@cfn.config(
    state_features={'robot_state.ee_pose': 7, 'grip': 1},
    exterior_camera='image.exterior',
    wrist_camera='image.wrist',
    image_size=(224, 224),
)
def observation(state_features: dict[str, int], exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """General OpenPI observation encoder with configurable state features."""
    return OpenpiObservationCodec(
        state_features=state_features, exterior_camera=exterior_camera, wrist_camera=wrist_camera, image_size=image_size
    )


ee_obs = observation
ee_joints_obs = observation.override(state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7})

droid_obs = codecs.general_obs.override(
    state_name='observation/joint_position',
    state_features={'robot_state.q': 7, 'grip': 1},
    image_mappings={
        'observation/wrist_image_left': 'image.wrist',
        'observation/exterior_image_1_left': 'image.exterior',
    },
    image_size=(224, 224),
    task_field='prompt',
)

ee = codecs.compose.override(obs=ee_obs, action=codecs.absolute_pos_action)
ee_joints = ee.override(obs=ee_joints_obs)

ee_traj = ee.override(action=codecs.traj_ee_action, binarize_grip=('grip',))
ee_joints_traj = ee_joints.override(action=codecs.traj_ee_action, binarize_grip=('grip',))

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_obs = observation.override(state_features={'robot_state.q': 7, 'grip': 1})
joints_traj = codecs.compose.override(
    obs=joints_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip'),
    binarize_grip=('grip',),
)

droid = codecs.compose.override(obs=droid_obs, action=codecs.joint_delta_action)
