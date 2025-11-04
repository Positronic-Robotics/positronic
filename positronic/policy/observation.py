from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image


class ObservationEncoder(transforms.KeyFuncEpisodeTransform):
    def __init__(self, state: dict[str, list[str]], images: dict[str, tuple[str, tuple[int, int]]]):
        """
        Build an observation encoder.

        Args:
            state: mapping from output state key to an ordered list of episode keys to concatenate.
            images: mapping from output image name to tuple (input_key, (width, height)).
        """
        transform_fns = {k: partial(self.encode_state, k) for k in state.keys()}
        transform_fns.update({k: partial(self.encode_image, k) for k in images.keys()})
        super().__init__(add=transform_fns, pass_through=False)
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
