"""OpenPI codecs (observation encoder + action decoder pairs).

OpenPI has different key format expectations for training vs inference:

- **Training (LeRobot format)**: Dot-separated keys like `observation.state`, `observation.images.left`.
  This is what LeRobot datasets use and what OpenPI training code consumes.

- **Inference (OpenPI format)**: Slash-separated keys like `observation/state`, `observation/image`.
  This is what OpenPI's policy classes (e.g., `positronic_policy.py`) expect at inference time.

The `OpenpiObservationEncoder` class handles both cases:
- `__call__` (training): Produces LeRobot-compatible format with dot-separated keys
- `encode()` (inference): Produces OpenPI-compatible format with slash-separated keys

Note: `droid` codec is inference-only, designed to work with pretrained DROID models.
It doesn't need training support since we don't generate DROID-format training data.
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
from positronic.policy.observation import ObservationEncoder


class OpenpiObservationEncoder(ObservationEncoder):
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

        # Define transforms for training (derive from Episode) using LeRobot keys (dot-separated)
        super().__init__(**{
            'observation.state': self._derive_state,
            'observation.images.left': partial(self._derive_image, wrist_camera),
            'observation.images.side': partial(self._derive_image, exterior_camera),
        })

        state_dim = sum(state_features.values())
        w, h = image_size
        self._metadata: dict[str, Any] = {
            'lerobot_features': {
                'observation.state': {'shape': (state_dim,), 'names': list(state_features.keys()), 'dtype': 'float32'},
                'observation.images.left': {
                    'shape': (h, w, 3),
                    'names': ['height', 'width', 'channel'],
                    'dtype': 'video',
                },
                'observation.images.side': {
                    'shape': (h, w, 3),
                    'names': ['height', 'width', 'channel'],
                    'dtype': 'video',
                },
            }
        }

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    def _derive_state(self, episode: Episode) -> Signal[Any]:
        """Concatenate state features into a single vector."""
        return transforms.concat(*[episode[key] for key in self._state_features], dtype=np.float32)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        """Resize image to target size."""
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode raw robot inputs into OpenPI format for inference.

        Args:
            inputs: Dict with keys matching state_features (e.g., 'robot_state.ee_pose', 'grip'),
                   plus camera keys (e.g., 'image.wrist', 'image.exterior') and optional 'task'.

        Returns:
            Dict with OpenPI slash-separated keys:
            {
                'observation/state': np.ndarray,
                'observation/image': np.ndarray (exterior camera),
                'observation/wrist_image': np.ndarray,
                'prompt': str (optional),
            }
        """
        # Encode state vector (concatenate all state features)
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
        """Encode and resize a single image for inference."""
        if input_key not in inputs:
            raise KeyError(f"Missing image input '{input_key}', available keys: {list(inputs.keys())}")
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_input(self) -> dict[str, Any]:
        dummy: dict[str, Any] = {}
        for key, dim in self._state_features.items():
            dummy[key] = np.zeros(dim, dtype=np.float32)
        w, h = self._image_size
        dummy[self._wrist_camera] = np.zeros((h, w, 3), dtype=np.uint8)
        dummy[self._exterior_camera] = np.zeros((h, w, 3), dtype=np.uint8)
        dummy['task'] = 'warmup'
        return dummy


# ===== Observation Encoder Configs =====


@cfn.config(
    state_features={'robot_state.ee_pose': 7, 'grip': 1},
    exterior_camera='image.exterior',
    wrist_camera='image.wrist',
    image_size=(224, 224),
)
def observation(state_features: dict[str, int], exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """General OpenPI observation encoder with configurable state features."""
    return OpenpiObservationEncoder(
        state_features=state_features, exterior_camera=exterior_camera, wrist_camera=wrist_camera, image_size=image_size
    )


# EE pose + grip (default config)
eepose_observation = observation

# EE pose + grip + joint positions
eepose_q_observation = observation.override(state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7})

# DROID: inference-only, uses base general config with DROID-specific keys
droid_observation = codecs.general.override(
    state_name='observation/joint_position',
    state_features={'robot_state.q': 7, 'grip': 1},
    image_mappings={
        'observation/wrist_image_left': 'image.wrist',
        'observation/exterior_image_1_left': 'image.exterior',
    },
    image_size=(224, 224),
    task_field='prompt',
)


# ===== Combined Codec Configs (observation + action pairs) =====

# EE pose + grip -> absolute position (primary codec for training and inference)
eepose = codecs.codec.override(observation=eepose_observation, action=codecs.absolute_position(rotation_rep=None))

# EE pose + grip + joint positions -> absolute position
eepose_q = codecs.codec.override(observation=eepose_q_observation, action=codecs.absolute_position(rotation_rep=None))

# Trajectory variants: use actual robot trajectory as action target instead of commanded targets
eepose_traj = codecs.codec.override(
    observation=eepose_observation,
    action=codecs.absolute_position(rotation_rep=None, tgt_ee_pose_key='robot_state.ee_pose', tgt_grip_key='grip'),
)

eepose_q_traj = codecs.codec.override(
    observation=eepose_q_observation,
    action=codecs.absolute_position(rotation_rep=None, tgt_ee_pose_key='robot_state.ee_pose', tgt_grip_key='grip'),
)

# DROID codec for inference with pretrained DROID models (inference-only)
droid = codecs.codec.override(observation=droid_observation, action=codecs.joint_delta(num_joints=7))
