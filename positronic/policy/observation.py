from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic import keys
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive
from positronic.policy.codec import Codec, lerobot_image, lerobot_state

# The encoded observation's language prompt, under the name LeRobot training and its policies both use. It
# shares a value with ``keys.TASK`` by vocabulary, not by contract: that one names the prompt on the way in.
TASK_FIELD = 'task'


class ObservationCodec(Codec):
    """Configurable observation encoder that uses the same keys for training and inference.

    Args:
        state: mapping from output state key to an ordered dict of {episode_key: dim} to concatenate.
        images: mapping from output image name to tuple (input_key, (width, height)).
        task_field: output key carrying the language prompt at inference; training always emits ``TASK_FIELD``.
        lowercase_task: lowercase the task text at inference, for checkpoints trained on lowercased language
            (the pretrained DROID models; MolmoSpaces' Pi baseline applies the same normalization).
    """

    WIRE_NAME = 'observation_codec'

    def __init__(
        self,
        state: dict[str, dict[str, int]],
        images: dict[str, tuple[str, tuple[int, int]]],
        task_field: str = TASK_FIELD,
        lowercase_task: bool = False,
    ):
        self._state = state
        self._image_configs = images
        self._task_field = task_field
        self._lowercase_task = lowercase_task

        self._derive_transforms: dict[str, Callable[[Episode], Any]] = {
            k: partial(self._derive_state, k) for k in state.keys()
        }
        self._derive_transforms.update({k: partial(self._derive_image, k) for k in images.keys()})
        # Lowercase the training task the same way ``encode`` lowercases the served prompt, so a codec with
        # ``lowercase_task`` trains and infers on one text distribution (the ``Codec`` same-keys contract).
        self._derive_transforms[TASK_FIELD] = self._derive_task

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

    def _normalize_task(self, task: str) -> str:
        return task.lower() if self._lowercase_task else task

    def _derive_task(self, episode: Episode) -> Any:
        return self._normalize_task(episode[keys.TASK] if keys.TASK in episode else '')

    def _decode_single(self, data: dict) -> dict:
        return {}

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        if keys.TASK in inputs:
            obs[self._task_field] = self._normalize_task(inputs[keys.TASK])

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

    @property
    def meta(self):
        sizes = {input_key: (w, h) for _out, (input_key, (w, h)) in self._image_configs.items()}
        unique = set(sizes.values())
        return {'image_sizes': unique.pop() if len(unique) == 1 else sizes}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)

    def to_spec(self):
        # Normalized to lists so the spec is identical before and after a wire round-trip.
        images = {name: [key, list(size)] for name, (key, size) in self._image_configs.items()}
        return {
            'name': self.WIRE_NAME,
            'args': {
                'state': self._state,
                'images': images,
                'task_field': self._task_field,
                'lowercase_task': self._lowercase_task,
            },
        }
