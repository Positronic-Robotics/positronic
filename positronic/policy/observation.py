from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import geom, keys
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode, select_timeline
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get, Group, Identity
from positronic.policy.codec import LEROBOT_FEATURES, Codec, lerobot_image, lerobot_vector

# The encoded observation's language prompt, under the name LeRobot training and its policies both use. It
# shares a value with ``keys.TASK`` by vocabulary, not by contract: that one names the prompt on the way in.
TASK_FIELD = 'task'


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


class ConvertPose(Codec):
    """Convert selected observation poses and recorded pose signals to float32 vectors.

    The offset multiplies each rotation on the right; translation and frame metadata stay unchanged.
    Wire offsets are wxyz quaternions. Selected fields are required. Decoded actions pass through.
    """

    WIRE_NAME = 'convert_pose'

    def __init__(
        self,
        output_rotation: str,
        *,
        keys: Sequence[str] = (keys.EE_POSE,),
        input_rotation: str = geom.Rotation.Representation.QUAT.value,
        rotation_offset: geom.Rotation | Sequence[float] = (1, 0, 0, 0),
    ):
        self._keys = tuple(keys)
        self._input_rotation = geom.Rotation.Representation(input_rotation)
        self._output_rotation = geom.Rotation.Representation(output_rotation)
        if not isinstance(rotation_offset, geom.Rotation):
            rotation_offset = geom.Rotation.from_quat(np.asarray(rotation_offset))
        self._rotation_offset = rotation_offset

    def _convert(self, value: Any) -> np.ndarray:
        vector = np.asarray(value)
        expected = 3 + self._input_rotation.size
        if vector.shape != (expected,):
            raise ValueError(f'Expected a pose vector with {expected} values, got shape {vector.shape}')
        pose = geom.Transform3D.from_vector(vector, self._input_rotation)
        converted = geom.Transform3D(pose.translation, pose.rotation * self._rotation_offset)
        return converted.as_vector(self._output_rotation).astype(np.float32)

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, **{key: self._convert(data[key]) for key in self._keys}}

    def decode(self, data: Any) -> Any:
        return data

    def _derive_pose(self, key: str, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode[key], transforms.lazy_sequence(self._convert))

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Group(Derive(meta=None, **{key: partial(self._derive_pose, key) for key in self._keys}), Identity())

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'output_rotation': self._output_rotation.value,
                'keys': list(self._keys),
                'input_rotation': self._input_rotation.value,
                'rotation_offset': self._rotation_offset.as_quat.tolist(),
            },
        }


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
