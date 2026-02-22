from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Group, Identity
from positronic.policy.codec import Codec, lerobot_image, lerobot_state


class ObservationCodec(Codec):
    """Configurable observation encoder that uses the same keys for training and inference.

    Args:
        state: mapping from output state key to an ordered dict of {episode_key: dim} to concatenate.
        images: mapping from output image name to tuple (input_key, (width, height)).
        task_field: name of the field to store task string ('task', 'prompt', etc.), or None to disable.
    """

    def __init__(
        self,
        state: dict[str, dict[str, int]],
        images: dict[str, tuple[str, tuple[int, int]]],
        task_field: str | None = 'task',
    ):
        self._state = state
        self._image_configs = images
        self._task_field = task_field

        self._derive_transforms = {k: partial(self._derive_state, k) for k in state.keys()}
        self._derive_transforms.update({k: partial(self._derive_image, k) for k in images.keys()})
        if task_field:
            self._derive_transforms['task'] = lambda ep: ep['task'] if 'task' in ep else ''

        lerobot_features: dict[str, Any] = {}
        for name, features in state.items():
            if isinstance(features, dict):
                lerobot_features[name] = lerobot_state(sum(features.values()), list(features.keys()))
        for name, (_, (w, h)) in images.items():
            lerobot_features[name] = lerobot_image(w, h)
        self._training_meta = {'lerobot_features': lerobot_features}

    def _derive_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        return transforms.concat(*[episode[k] for k in state_features], dtype=np.float32)

    def _derive_image(self, out_name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[out_name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
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
            parts = []
            for f in feature_names:
                if f not in inputs:
                    raise KeyError(f"Missing state input '{f}' for '{out_name}', available keys: {list(inputs.keys())}")
                parts.append(np.asarray(inputs[f], dtype=np.float32).reshape(-1))
            obs[out_name] = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

        return obs

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        """Return a zero-filled encoded observation matching the shapes ``encode()`` produces."""
        obs: dict[str, Any] = {}
        for out_name, features in self._state.items():
            obs[out_name] = np.zeros(sum(features.values()), dtype=np.float32)
        for out_name, (_input_key, (width, height)) in self._image_configs.items():
            obs[out_name] = np.zeros((height, width, 3), dtype=np.uint8)
        if self._task_field:
            obs[self._task_field] = 'warmup'
        return obs

    @property
    def meta(self):
        sizes = {input_key: (w, h) for _out, (input_key, (w, h)) in self._image_configs.items()}
        unique = set(sizes.values())
        return {'image_sizes': unique.pop() if len(unique) == 1 else sizes}

    @property
    def training_encoder(self):
        return Group(Derive(meta=self._training_meta, **self._derive_transforms), Identity())
