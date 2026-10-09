from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import keys
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode, select_timeline
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get, Identity

from .base import Codec
from .metadata import LEROBOT_FEATURES, lerobot_image, lerobot_vector

# The encoded prompt name shared by LeRobot training and inference.
TASK_FIELD = 'task'


class PackObservationFields(Codec):
    """Select fields into nested dictionaries and add leading singleton dimensions.

    Layout leaves name literal input keys. Arrays retain their dtype; other values gain list layers.
    Training columns and decoded actions pass through unchanged.
    """

    WIRE_NAME = 'pack_observation_fields'

    def __init__(self, layout: dict[str, Any], *, leading_dims: int = 0):
        if type(leading_dims) is not int or leading_dims < 0:
            raise ValueError('leading_dims must be a non-negative integer')
        self._layout = self._copy_layout(layout)
        self._leading_dims = leading_dims

    @staticmethod
    def _copy_layout(layout: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for name, source in layout.items():
            if not isinstance(name, str) or not isinstance(source, (str, dict)):
                raise ValueError('Layout entries must have string keys and contain input key names or dictionaries')
            result[name] = PackObservationFields._copy_layout(source) if isinstance(source, dict) else source
        return result

    def _pack(self, layout: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for name, source in layout.items():
            if isinstance(source, dict):
                result[name] = self._pack(source, data)
            else:
                value = data[source]
                if isinstance(value, np.ndarray):
                    value = value.reshape((1,) * self._leading_dims + value.shape)
                else:
                    for _ in range(self._leading_dims):
                        value = [value]
                result[name] = value
        return result

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._pack(self._layout, data)

    def decode(self, data: Any) -> Any:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'layout': self._copy_layout(self._layout), 'leading_dims': self._leading_dims},
        }


class RenameObservationFields(Codec):
    """Rename top-level inference fields with a source-to-destination mapping.

    Absent fields stay absent; unmapped fields keep their names. Names containing dots or slashes
    are literal keys. Training columns and decoded actions pass through unchanged.
    """

    WIRE_NAME = 'rename_observation_fields'

    def __init__(self, mapping: dict[str, str]):
        self._mapping = dict(mapping)

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        renamed: dict[str, Any] = {}
        for name, value in data.items():
            destination = self._mapping.get(name, name)
            if destination in renamed:
                raise ValueError(f'Observation fields collide at {destination!r}')
            renamed[destination] = value
        return renamed

    def decode(self, data: Any) -> Any:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'mapping': dict(self._mapping)}}


class ObservationCodec(Codec):
    """Encode state vectors and images for training and inference.

    Args:
        state: mapping from output state key to an ordered dict of {episode_key: dim} to concatenate.
        images: mapping from output image name to tuple (input_key, (width, height)).
        task_field: output key carrying the language prompt at inference.
    """

    WIRE_NAME = 'observation_codec'

    def __init__(
        self,
        state: dict[str, dict[str, int]],
        images: dict[str, tuple[str, tuple[int, int]]],
        task_field: str = TASK_FIELD,
    ):
        self._state = state
        self._image_configs = images
        self._task_field = task_field

        self._derive_transforms: dict[str, Any] = {k: partial(self._derive_state, k) for k in state.keys()}
        self._derive_transforms.update({k: partial(self._derive_image, k) for k in images.keys()})
        self._derive_transforms[TASK_FIELD] = Get(keys.TASK, '')

        lerobot_features: dict[str, Any] = {}
        for name, features in state.items():
            if isinstance(features, dict):
                lerobot_features[name] = lerobot_vector(sum(features.values()), list(features.keys()))
        for name, (_, (w, h)) in images.items():
            lerobot_features[name] = lerobot_image(w, h)
        self._training_meta = {LEROBOT_FEATURES: lerobot_features}

    def _derive_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        signals = [episode[k] for k in state_features]
        timeline = select_timeline(name for signal in signals for name in signal.timelines)
        return transforms.concat(*signals, dtype=np.float32, timelines=(timeline,))

    def _derive_image(self, out_name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[out_name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        if keys.TASK in inputs:
            obs[self._task_field] = inputs[keys.TASK]

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
        return {self.IMAGE_SIZES: unique.pop() if len(unique) == 1 else sizes}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)

    def to_spec(self):
        # Normalized to lists so the spec is identical before and after a wire round-trip.
        images = {name: [key, list(size)] for name, (key, size) in self._image_configs.items()}
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'state': self._state, 'images': images, 'task_field': self._task_field},
        }
