from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive


class ObservationEncoder(Derive, ABC):
    """Abstract base for observation encoders.

    Encoders must implement:
    - __call__(episode): for training data generation (inherited from Derive)
    - encode(inputs): for inference
    - meta: metadata dict (e.g., lerobot_features), default empty
    """

    @property
    def meta(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode raw robot inputs into observation dict for inference."""
        ...


class SimpleObservationEncoder(ObservationEncoder):
    """Configurable observation encoder that uses the same keys for training and inference.

    Args:
        state: mapping from output state key to an ordered list of episode keys to concatenate.
        images: mapping from output image name to tuple (input_key, (width, height)).
        task_field: name of the field to store task string ('task', 'prompt', etc.), or None to disable.
    """

    def __init__(
        self,
        state: dict[str, list[str]],
        images: dict[str, tuple[str, tuple[int, int]]],
        task_field: str | None = 'task',
    ):
        derive_transforms = {k: partial(self._encode_state, k) for k in state.keys()}
        derive_transforms.update({k: partial(self._encode_image, k) for k in images.keys()})
        super().__init__(**derive_transforms)
        self._state = state
        self._image_configs = images
        self._task_field = task_field
        self._metadata: dict[str, Any] = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    @meta.setter
    def meta(self, value: dict[str, Any]):
        self._metadata = value

    def _encode_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        return transforms.concat(*[episode[k] for k in state_features], dtype=np.float32)

    def _encode_image(self, out_name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[out_name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode raw inputs for inference (uses same keys as training)."""
        obs: dict[str, Any] = {}

        if self._task_field and 'task' in inputs:
            obs[self._task_field] = inputs['task']

        for out_name, (input_key, (width, height)) in self._image_configs.items():
            if input_key not in inputs:
                raise KeyError(f"Missing image input '{input_key}' for '{out_name}', available keys: {inputs.keys()}")
            frame = inputs[input_key]
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Image '{input_key}' must be HWC with 3 channels, got {frame.shape}")
            obs[out_name] = image.resize_with_pad_per_frame(width, height, PilImage.Resampling.BILINEAR, frame)

        for out_name, feature_names in self._state.items():
            parts = [np.asarray(inputs[f], dtype=np.float32).reshape(-1) for f in feature_names if f in inputs]
            obs[out_name] = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

        return obs
