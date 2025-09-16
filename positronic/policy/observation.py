from typing import Any, Sequence

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms


class ObservationEncoder(transforms.EpisodeTransform):

    def __init__(self, state_features: list[str], **image_configs):
        """
        Build an observation encoder.

        Args:
            state_features: list of keys to concatenate for the state vector.
            image_configs: mapping from output image name (suffix) to tuple (input_key, (width, height)).
                           The output key will be 'observation.images.{name}'.
        """
        self._state_features = state_features
        self._image_configs = image_configs

    @property
    def keys(self) -> Sequence[str]:
        return ['observation.state'] + [f'observation.images.{k}' for k in self._image_configs.keys()]

    def transform(self, name: str, episode: transforms.Episode) -> Signal[Any] | Any:
        if name == 'observation.state':
            return transforms.concat(*[episode[k] for k in self._state_features],
                                     dtype=np.float64,
                                     names=self._state_features)
        elif name.startswith('observation.images.'):
            key = name[len('observation.images.'):]
            input_key, (widht, height) = self._image_configs[key]
            return transforms.Image.resize_with_pad(widht, height, episode[input_key])
        else:
            raise ValueError(f"Unknown observation key: {name}")

    def get_features(self):
        features = {}
        for key, (_, (width, height)) in self._image_configs.items():
            features['observation.images.' + key] = {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
        features['observation.state'] = {
            "dtype": "float64",
            "shape": (8, ),  # TODO: Invent the way to compute it dynamically
            "names": list(self._state_features),
        }
        return features

    def encode(self, images: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single inference observation from raw images and input dict.

        Returns numpy arrays:
          - images: (1, C, H, W), float32 in [0,1]
          - state: (1, D), float32
        """

        obs: dict[str, Any] = {}

        # Encode images
        for out_name, (input_key, (width, height)) in self._image_configs.items():
            if input_key not in images:
                raise KeyError(f"Missing image input '{input_key}' for '{out_name}'")
            frame = images[input_key]
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Image '{input_key}' must be HWC with 3 channels, got {frame.shape}")
            resized = transforms.Image.resize_with_pad_per_frame(width, height, PilImage.Resampling.BILINEAR, frame)
            chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
            obs[f'observation.images.{out_name}'] = chw[np.newaxis, ...]

        # Encode state vector
        parts: list[np.ndarray] = []
        for k in self._state_features:
            v = inputs[k]
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            parts.append(arr)
        if parts:
            state_vec = np.concatenate(parts, axis=0)
        else:
            state_vec = np.empty((0, ), dtype=np.float32)
        obs['observation.state'] = state_vec[np.newaxis, ...]
        return obs
