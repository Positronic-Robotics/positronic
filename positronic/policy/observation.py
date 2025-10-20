from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image


class ObservationEncoder(transforms.KeyFuncEpisodeTransform):
    def __init__(self, state_features: list[str], images: dict[str, tuple[str, tuple[int, int]]]):
        """
        Build an observation encoder.

        Args:
            state_features: list of keys to concatenate for the state vector.
            images: mapping from output image name (suffix) to tuple (input_key, (width, height)).
                    The output key will be 'observation.images.{name}'.
        """
        image_fns = {f'observation.images.{k}': partial(self.encode_image, k) for k in images.keys()}
        super().__init__(**{'observation.state': self.encode_state}, **image_fns)
        self._state_features = state_features
        self._image_configs = images

    def encode_state(self, episode: Episode) -> Signal[Any]:
        return transforms.concat(
            *[episode[k] for k in self._state_features], dtype=np.float32, names=self._state_features
        )

    def encode_image(self, name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single inference observation from raw images and input dict.

        Returns numpy arrays:
          - images: (H, W, C), uint8
          - state: (D,), float32
        """

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
            obs[f'observation.images.{out_name}'] = resized

        # Encode state vector
        parts: list[np.ndarray] = []
        for k in self._state_features:
            v = inputs[k]
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            parts.append(arr)
        if parts:
            state_vec = np.concatenate(parts, axis=0)
        else:
            state_vec = np.empty((0,), dtype=np.float32)
        obs['observation.state'] = state_vec
        return obs
