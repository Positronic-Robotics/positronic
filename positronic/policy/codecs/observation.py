from functools import partial
from typing import Any

import numpy as np
from eval_vocabulary.timeline import recording_timeline
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import keys
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get, Identity

from .base import Codec
from .metadata import LEROBOT_FEATURES, lerobot_vector

# The encoded prompt name shared by LeRobot training and inference.
TASK_FIELD = 'task'


def bimanual_joints_state(data: dict[str, Any]) -> np.ndarray:
    """The two-arm state: per arm, the joints and then the gripper width (1 open, 0 closed), left arm first."""
    state = []
    for arm in keys.BIMANUAL_ARMS:
        state.append(np.asarray(data[f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.JOINTS_SUFFIX}']).reshape(-1))
        state.append(1.0 - np.asarray(data[keys.arm_channel(keys.GRIP, arm)], dtype=np.float32).reshape(-1))
    return np.concatenate(state).astype(np.float32)


class SelectObservationFields(Codec):
    """Select and rename literal input fields; contribute no decoded commands."""

    WIRE_NAME = 'select_observation_fields'

    def __init__(self, mapping: dict[str, str]):
        self._mapping = dict(mapping)

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        return {name: data[source] for name, source in self._mapping.items()}

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'mapping': dict(self._mapping)}}


class PackFields(Codec):
    """Arrange model inputs and outputs using independent destination layouts.

    Layouts are nested dictionaries with source paths at their leaves. Each path contains literal
    string keys or integer indices; ``[]`` selects the whole source value. Only selected fields
    appear in the result, including empty groups. An absent layout passes that direction through;
    an empty layout produces an empty dictionary.

    Encoding adds ``unsqueeze_dims`` leading size-one array dimensions or list layers. Decoding
    removes ``squeeze_dims`` leading size-one array dimensions or list/tuple layers. Arrays retain
    their dtype. Training passes through. Splitting a time axis into records is a separate operation.

    Example::

        PackFields(
            encode_layout={'state': {'joints': ['robot_state.q']}},
            decode_layout={'joints': [0, 'predicted_joints']},
            unsqueeze_dims=2,
            squeeze_dims=1,
        )

    This encodes joints ``(7,)`` as ``state.joints`` with shape ``(1, 1, 7)`` and decodes
    ``({'predicted_joints': array_of_shape_1_T_7}, info)`` into ``{'joints': array_of_shape_T_7}``.
    """

    WIRE_NAME = 'pack_fields'

    def __init__(
        self,
        *,
        encode_layout: dict[str, Any] | None = None,
        decode_layout: dict[str, Any] | None = None,
        unsqueeze_dims: int = 0,
        squeeze_dims: int = 0,
    ):
        for name, value in (('unsqueeze_dims', unsqueeze_dims), ('squeeze_dims', squeeze_dims)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        self._encode_layout = None if encode_layout is None else self._copy_layout(encode_layout)
        self._decode_layout = None if decode_layout is None else self._copy_layout(decode_layout)
        self._unsqueeze_dims = unsqueeze_dims
        self._squeeze_dims = squeeze_dims

    @staticmethod
    def _copy_layout(layout: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(layout, dict):
            raise ValueError('Layouts must be dictionaries')
        result = {}
        for name, source in layout.items():
            if not isinstance(name, str):
                raise ValueError('Layout entries must have string keys')
            if isinstance(source, dict):
                result[name] = PackFields._copy_layout(source)
                continue
            if not isinstance(source, (list, tuple)) or any(type(part) not in (str, int) for part in source):
                raise ValueError('Layout leaves must be source paths of string keys or integer indices')
            result[name] = list(source)
        return result

    @staticmethod
    def _reshape(value: Any, dimensions: int, field: str) -> Any:
        """Add leading singleton dimensions for positive counts; remove them for negative counts."""
        if isinstance(value, np.ndarray):
            if dimensions >= 0:
                return value.reshape((1,) * dimensions + value.shape)
            count = -dimensions
            if value.shape[:count] != (1,) * count:
                raise ValueError(f'Field {field!r} requires {count} leading size-one dimensions')
            return value.reshape(value.shape[count:])

        if dimensions >= 0:
            for _ in range(dimensions):
                value = [value]
            return value

        count = -dimensions
        for _ in range(count):
            if not isinstance(value, (list, tuple)) or len(value) != 1:
                raise ValueError(f'Field {field!r} requires {count} leading size-one dimensions')
            value = value[0]
        return value

    @staticmethod
    def _pack(layout: dict[str, Any], data: Any, dimensions: int) -> dict[str, Any]:
        result = {}
        for name, source in layout.items():
            if isinstance(source, dict):
                result[name] = PackFields._pack(source, data, dimensions)
            else:
                value = data
                for part in source:
                    value = value[part]
                result[name] = PackFields._reshape(value, dimensions, name)
        return result

    def encode(self, data: Any) -> Any:
        if self._encode_layout is None:
            return data
        return self._pack(self._encode_layout, data, self._unsqueeze_dims)

    def decode(self, data: Any) -> Any:
        if self._decode_layout is None:
            return data
        return self._pack(self._decode_layout, data, -self._squeeze_dims)

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'encode_layout': None if self._encode_layout is None else self._copy_layout(self._encode_layout),
                'decode_layout': None if self._decode_layout is None else self._copy_layout(self._decode_layout),
                'unsqueeze_dims': self._unsqueeze_dims,
                'squeeze_dims': self._squeeze_dims,
            },
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


class StateObservationCodec(Codec):
    """Concatenate observation fields into flat float32 state vectors.

    Each output vector selects literal input keys in dictionary insertion order. Inference flattens
    each selected value and concatenates them, without normalization or adding batch dimensions.
    Missing state inputs raise ``KeyError``; an empty selection produces an empty float32 vector.
    Unselected inputs are omitted, except for ``keys.TASK``, which is copied to ``task_field`` when
    present. If a state output uses that same name, its vector takes precedence over the prompt.

    Args:
        state: Mapping of output names to ``{input_key: dimension}`` mappings. Dimensions describe
            LeRobot training features; they do not reshape or validate inference values.
        task_field: Output name for the inference prompt, defaulting to ``'task'``.

    Example::

        codec = StateObservationCodec({'state': {'joints': 2, 'grip': 1}}, task_field='prompt')
        codec.encode({'joints': [0.1, 0.2], 'grip': 0.8, keys.TASK: 'Pick up the cup'})
        # {'state': array([0.1, 0.2, 0.8], dtype=float32), 'prompt': 'Pick up the cup'}

    ``training_encoder`` derives concatenated float32 signals and LeRobot feature metadata.
    Its prompt column is always ``'task'``, independently of ``task_field``, and defaults to an
    empty string when the episode has no task. Decoding contributes no commands.
    """

    WIRE_NAME = 'state_observation'

    def __init__(self, state: dict[str, dict[str, int]], task_field: str = TASK_FIELD):
        self._state = state
        self._task_field = task_field

        self._derive_transforms: dict[str, Any] = {k: partial(self._derive_state, k) for k in state.keys()}
        self._derive_transforms[TASK_FIELD] = Get(keys.TASK, '')

        lerobot_features: dict[str, Any] = {}
        for name, features in state.items():
            if isinstance(features, dict):
                lerobot_features[name] = lerobot_vector(sum(features.values()), list(features.keys()))
        self._training_meta = {LEROBOT_FEATURES: lerobot_features}

    def _derive_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        signals = [episode[k] for k in state_features]
        timeline = recording_timeline({name for signal in signals for name in signal.timelines})
        return transforms.concat(*signals, dtype=np.float32, timelines=(timeline,))

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        if keys.TASK in data:
            obs[self._task_field] = data[keys.TASK]

        for out_name, feature_names in self._state.items():
            parts = []
            for f in feature_names:
                if f not in data:
                    raise KeyError(f"Missing state input '{f}' for '{out_name}', available keys: {list(data.keys())}")
                parts.append(np.asarray(data[f], dtype=np.float32).reshape(-1))
            obs[out_name] = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

        return obs

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'state': self._state, 'task_field': self._task_field},
        }
