from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive


class ObservationEncoder(Derive):
    def __init__(self, state: dict[str, list[str]], images: dict[str, tuple[str, tuple[int, int]]]):
        """
        Build an observation encoder.

        Args:
            state: mapping from output state key to an ordered list of episode keys to concatenate.
            images: mapping from output image name to tuple (input_key, (width, height)).
        """
        transforms = {k: partial(self.encode_state, k) for k in state.keys()}
        transforms.update({k: partial(self.encode_image, k) for k in images.keys()})
        super().__init__(**transforms)
        self._state = state
        self._image_configs = images
        self._metadata = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    @meta.setter
    def meta(self, value: dict[str, Any]):
        self._metadata = value

    def encode_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        return transforms.concat(*[episode[k] for k in state_features], dtype=np.float32)

    def encode_image(self, out_name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[out_name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single inference observation from raw images and input dict."""

        obs: dict[str, Any] = {}

        if 'task' in inputs:
            obs['task'] = inputs['task']

        # Encode images
        for out_name, (input_key, (width, height)) in self._image_configs.items():
            if input_key not in inputs:
                raise KeyError(f"Missing image input '{input_key}' for '{out_name}', available keys: {inputs.keys()}")
            frame = inputs[input_key]
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Image '{input_key}' must be HWC with 3 channels, got {frame.shape}")
            resized = image.resize_with_pad_per_frame(width, height, PilImage.Resampling.BILINEAR, frame)
            obs[out_name] = resized

        # Encode state vector
        for out_name, feature_names in self._state.items():
            parts: list[np.ndarray] = []
            for feature in feature_names:
                if feature not in inputs:
                    raise KeyError(f"Missing state input '{feature}' for '{out_name}', available keys: {inputs.keys()}")
                v = inputs[feature]
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                parts.append(arr)
            if parts:
                state_vec = np.concatenate(parts, axis=0)
            else:
                state_vec = np.empty((0,), dtype=np.float32)
            obs[out_name] = state_vec
        return obs


# GR00T N1.6 Observation Encoders
# These encode observations into the nested format expected by N1.6 PolicyServer:
# {
#     'video': {key: np.ndarray[uint8, (B, T, H, W, C)]},
#     'state': {key: np.ndarray[float32, (B, T, D)]},
#     'language': {key: list[list[str]]},  # (B, T)
# }


class GrootInferenceObservationEncoder(ObservationEncoder):
    """Encodes observations for GR00T N1.6 inference (EE pose control)."""

    def __init__(self):
        state = {'observation.state': ['robot_state.ee_pose', 'grip']}
        images = {'wrist_image': ('image.wrist', (224, 224)), 'exterior_image_1': ('image.exterior', (224, 224))}
        super().__init__(state, images)

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs = super().encode(inputs)

        # Extract state components
        state = obs.pop('observation.state').astype(np.float32)
        translation = state[:3]
        quaternion = state[3:7]
        grip = state[7:8]

        # Build N1.6 nested structure with proper shapes and dtypes
        # Video: (H, W, C) -> (B=1, T=1, H, W, C), dtype=uint8
        # State: (D,) -> (B=1, T=1, D), dtype=float32
        # Language: str -> [[str]] (B=1, T=1)
        return {
            'video': {
                'wrist_image': obs['wrist_image'][np.newaxis, np.newaxis, ...],
                'exterior_image_1': obs['exterior_image_1'][np.newaxis, np.newaxis, ...],
            },
            'state': {
                'robot_position_translation': translation[np.newaxis, np.newaxis, ...],
                'robot_position_quaternion': quaternion[np.newaxis, np.newaxis, ...],
                'grip': grip[np.newaxis, np.newaxis, ...],
            },
            'language': {'annotation.language.language_instruction': [[inputs.get('task', '')]]},
        }


class GrootEE_QObservationEncoder(ObservationEncoder):
    """Encodes observations for GR00T N1.6 inference (EE pose + joint position)."""

    def __init__(self):
        state = {'observation.state': ['robot_state.ee_pose', 'grip', 'robot_state.q']}
        images = {'wrist_image': ('image.wrist', (224, 224)), 'exterior_image_1': ('image.exterior', (224, 224))}
        super().__init__(state, images)

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs = super().encode(inputs)

        # Extract state components
        state = obs.pop('observation.state').astype(np.float32)
        translation = state[:3]
        quaternion = state[3:7]
        grip = state[7:8]
        joint_position = state[8:]

        # Build N1.6 nested structure
        return {
            'video': {
                'wrist_image': obs['wrist_image'][np.newaxis, np.newaxis, ...],
                'exterior_image_1': obs['exterior_image_1'][np.newaxis, np.newaxis, ...],
            },
            'state': {
                'robot_position_translation': translation[np.newaxis, np.newaxis, ...],
                'robot_position_quaternion': quaternion[np.newaxis, np.newaxis, ...],
                'grip': grip[np.newaxis, np.newaxis, ...],
                'joint_position': joint_position[np.newaxis, np.newaxis, ...],
            },
            'language': {'annotation.language.language_instruction': [[inputs.get('task', '')]]},
        }
