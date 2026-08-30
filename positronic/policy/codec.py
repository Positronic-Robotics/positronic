"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Two composition operators:

- ``|`` (sequential): left's output feeds into right. Use for codecs that modify data
  before others see it (e.g. BinarizeGrip before observation/action encoders).
- ``&`` (parallel): both see the same input, outputs merged. Use for independent codecs
  (e.g. observation encoder & action decoder).
"""

import collections.abc as cabc
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Any, final

import numpy as np
from PIL import Image as PilImage

from positronic import geom
from positronic import keys as obs_keys
from positronic.dataset.transforms import Elementwise, lazy_sequence
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, FromValue, Group, Identity
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.ik import assert_default_frame, change_frame, ee_frame
from positronic.drivers.roboarm.models import DEFAULT_FRAME
from positronic.policy.base import PAR, SEQ, ChunkLayer, Layer, _ComposedLayer
from positronic.utils import merge_dicts

_QUAT = geom.Rotation.Representation.QUAT


def lerobot_state(dim: int, names: list[str] | None = None) -> dict[str, Any]:
    """LeRobot feature descriptor for a state vector."""
    f: dict[str, Any] = {'shape': (dim,), 'dtype': 'float32'}
    if names:
        f['names'] = names
    return f


def lerobot_image(width: int, height: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an RGB image."""
    return {'shape': (height, width, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'}


def lerobot_action(dim: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an action vector."""
    return {'shape': (dim,), 'names': ['actions'], 'dtype': 'float32'}


class Codec(ChunkLayer):
    """Base class for observation/action codecs.

    Subclasses override ``encode`` (observation encoding) and/or ``_decode_single``
    (action decoding). The ``training_encoder`` property
    returns an ``EpisodeTransform`` used by the training pipeline to derive dataset columns.

    Reserved ``meta`` key:

    ``image_sizes``
        The image dimensions this codec encodes to. Either a ``(width, height)`` tuple (same
        size for all images) or a dict mapping raw input keys to ``(width, height)`` tuples.
    """

    def encode(self, data: dict) -> dict:
        return {}

    def decode(self, data):
        if isinstance(data, list):
            return [self.decode(d) for d in data]
        return self._decode_single(data)

    def _decode_single(self, data: dict) -> dict:
        return {}

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive()

    @property
    def meta(self) -> dict:
        return {}

    @contextmanager
    def episode_fn(self, infer):
        yield partial(self._coded, infer)

    def _coded(self, infer, obs):
        """The chunk ``infer`` answers for the encoded observation, decoded."""
        return self.decode(infer(self.encode(obs)))

    @final
    def __or__(self, other):
        if isinstance(other, Codec):
            return _ComposedCodec(self, other)
        if isinstance(other, Layer):
            # Mixed Codec | non-Codec layer → generic pipeline, not a Codec.
            return _ComposedLayer((self, *other._layers()))
        return NotImplemented

    @final
    def __and__(self, other):
        if isinstance(other, Codec):
            return _ParallelCodec(self, other)
        return NotImplemented


def _meta_conflicts(left: dict, right: dict, prefix: str = '') -> list[str]:
    """``key: left != right`` for every leaf the two metas declare differently. Nested dicts merge per key,
    so only leaves can conflict."""
    found = []
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if isinstance(a, dict) and isinstance(b, dict):
            found += _meta_conflicts(a, b, f'{prefix}{key}.')
        elif isinstance(a, dict) or isinstance(b, dict) or not np.array_equal(a, b):
            found.append(f'{prefix}{key}: {a!r} != {b!r}')
    return found


def _merged_meta(left: dict, right: dict) -> dict:
    """Two codecs' metadata as one dict.

    A leaf both declare differently has no merged answer, so it raises rather than keeping one: the survivor
    would describe a pipeline neither codec implements. Declare the composition as a single value instead.
    """
    conflicts = _meta_conflicts(left, right)
    if conflicts:
        raise ValueError(f'composed codecs disagree on metadata — {"; ".join(sorted(conflicts))}')
    result: dict[str, Any] = {}
    merge_dicts(result, left)
    merge_dicts(result, right)
    return result


class _ComposedCodec(Codec):
    """Two codecs composed via ``|``. Encodes left-to-right, decodes right-to-left."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return self._right.encode(self._left.encode(data))

    def decode(self, data):
        return self._left.decode(self._right.decode(data))

    @property
    def training_encoder(self):
        return self._left.training_encoder | self._right.training_encoder

    @property
    def meta(self):
        left, right = self._left.meta, self._right.meta
        # ``ee_frame`` states where poses sit, not how a codec is configured, so declaring it means having moved
        # them. Agreeing on a value buys nothing here: the second move starts where the first left off. Under
        # ``&`` both halves see the same input, which is why the check belongs to sequential composition.
        if obs_keys.EE_FRAME in left and obs_keys.EE_FRAME in right:
            raise ValueError(
                f'sequential codecs both declare {obs_keys.EE_FRAME}: poses come out at the product of both '
                f'moves, which neither {left[obs_keys.EE_FRAME]} nor {right[obs_keys.EE_FRAME]} names'
            )
        return _merged_meta(left, right)

    def to_spec(self):
        return {SEQ: [self._left.to_spec(), self._right.to_spec()]}


class _ParallelCodec(Codec):
    """Two codecs composed via ``&``. Both see the same input, outputs merged."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return {**self._left.encode(data), **self._right.encode(data)}

    def decode(self, data):
        left_out = self._left.decode(data)
        right_out = self._right.decode(data)
        if isinstance(data, list):
            return [{**lf, **rt} for lf, rt in zip(left_out, right_out, strict=True)]
        return {**left_out, **right_out}

    @property
    def training_encoder(self):
        return self._left.training_encoder & self._right.training_encoder

    @property
    def meta(self):
        return _merged_meta(self._left.meta, self._right.meta)

    def to_spec(self):
        return {PAR: [self._left.to_spec(), self._right.to_spec()]}


def is_action(entry: dict) -> bool:
    """True for a real command entry, False for a keyless validity sentinel.

    Time codecs close a chunk with a timestamp-only sentinel marking where its validity ends
    (see ``ActionTimestamp``). Consumers that classify or plot per-command fields skip the
    sentinel through this predicate rather than hard-coding its shape.
    """
    return bool(entry.keys() - {obs_keys.ACTION_TIMESTAMP})


class ActionTimestamp(Codec):
    """Stamps each decoded action with a relative ``timestamp`` (seconds from trajectory start).

    Assigns ``timestamp = i * (1/fps)`` starting at 0. The scheduler anchors them to the
    ``time_ns`` of the call that emits the chunk.

    A K-action chunk covers K periods, so the list is closed with a sentinel entry —
    a dict carrying only ``timestamp = K * (1/fps)`` and no command keys — stating when
    the chunk's validity ends. The scheduler reads that end from the last entry, so the
    final action gets a full period before re-inference. The sentinel carries no command
    key, so the key-filtered demux emits it on no channel: drivers and recordings never
    see it.

    At training time, surfaces ``action_fps`` as transform metadata.
    """

    WIRE_NAME = 'action_timestamp'

    def __init__(self, *, fps: float):
        self._fps = fps
        self._dt = 1.0 / fps

    def encode(self, data):
        return data

    def decode(self, data):
        # Build fresh entries rather than stamping in place: a session may hand back a cached template
        # list, and appending the sentinel to it would regrow the chunk on every re-inference.
        if isinstance(data, list):
            stamped = [{**d, obs_keys.ACTION_TIMESTAMP: i * self._dt} for i, d in enumerate(data)]
            if stamped:
                stamped.append({obs_keys.ACTION_TIMESTAMP: len(stamped) * self._dt})
            return stamped
        return {**data, obs_keys.ACTION_TIMESTAMP: 0}

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_fps': self._fps}

    def to_spec(self):
        return {'name': self.WIRE_NAME, 'args': {'fps': self._fps}}


class ActionHorizon(Codec):
    """Truncates action chunks to a time horizon.

    Keeps only actions whose (relative) ``timestamp`` is within ``horizon_sec``
    of trajectory start. Single actions pass through.

    When truncation drops entries, the kept list is closed with a sentinel entry —
    a dict carrying only ``timestamp = horizon_sec`` and no command keys — marking the
    boundary the chunk was cut at as where its validity ends, so the last surviving
    action still gets a full period before re-inference. When nothing is dropped the
    inner timestamp codec's own end-of-chunk sentinel already closes the list, so none
    is added.

    At training time, surfaces ``action_horizon_sec`` as transform metadata.
    """

    WIRE_NAME = 'action_horizon'

    def __init__(self, horizon_sec: float):
        self._horizon_sec = horizon_sec

    def encode(self, data):
        return data

    def decode(self, data):
        if isinstance(data, list):
            # Treat untimestamped actions as t=0 so they always pass the horizon
            # (servers may apply horizon truncation before stamping).
            kept = [d for d in data if d.get(obs_keys.ACTION_TIMESTAMP, 0.0) < self._horizon_sec]
            if len(kept) < len(data):
                kept.append({obs_keys.ACTION_TIMESTAMP: self._horizon_sec})
            return kept
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_horizon_sec': self._horizon_sec}

    def to_spec(self):
        return {'name': self.WIRE_NAME, 'args': {'horizon_sec': self._horizon_sec}}


def ActionTiming(*, fps: float, horizon_sec: float | None = None) -> Codec:
    """Convenience factory composing ``ActionTimestamp`` and ``ActionHorizon``.

    Equivalent to ``ActionTimestamp(fps=fps) | ActionHorizon(horizon_sec)`` when
    horizon_sec is set, or just ``ActionTimestamp(fps=fps)`` otherwise.
    """
    codec = ActionTimestamp(fps=fps)
    if horizon_sec is not None:
        codec = ActionHorizon(horizon_sec) | codec
    return codec


class BinarizeGripTraining(Codec):
    """Binarize grip signals in training data.

    Overrides the specified episode signals with thresholded values (> threshold → 1.0,
    else 0.0) so the model learns to predict binary grip. Compose to the left of
    obs/action codecs::

        timing | BinarizeGripTraining(('grip', 'target_grip')) | BinarizeGripInference() | obs & action
    """

    WIRE_NAME = 'binarize_grip_training'

    def __init__(self, keys: tuple[str, ...], threshold: float = 0.5):
        self._keys = keys
        self._threshold = threshold

    def encode(self, data):
        return data

    def _decode_single(self, data: dict) -> dict:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        threshold = self._threshold

        def _binarize_signal(key):
            def _derive(episode):
                return Elementwise(
                    episode[key], lambda v: (np.asarray(v, dtype=np.float32) > threshold).astype(np.float32)
                )

            return _derive

        transforms = {k: _binarize_signal(k) for k in self._keys}
        return Group(Derive(**transforms), Identity())

    def to_spec(self):
        return {'name': self.WIRE_NAME, 'args': {'keys': list(self._keys), 'threshold': self._threshold}}


class BinarizeGripInference(Codec):
    """Threshold grip in decoded actions at inference time.

    Compose to the left of action codecs so it runs after action decoding::

        timing | BinarizeGripInference() | obs & action
    """

    WIRE_NAME = 'binarize_grip_inference'

    def __init__(self, threshold: float = 0.5, key: str = obs_keys.TARGET_GRIP):
        self._threshold = threshold
        self._key = key

    def encode(self, data):
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        if self._key in data:
            data[self._key] = 1.0 if data[self._key] > self._threshold else 0.0
        return data

    def to_spec(self):
        return {'name': self.WIRE_NAME, 'args': {'threshold': self._threshold, 'key': self._key}}


class FlipGrip(Codec):
    """Serve a checkpoint that speaks the inverted grip convention (1 = open) on the canonical
    1 = closed wire: flips ``grip`` entering the model and ``target_grip`` leaving it.

    HACK: exists only to keep checkpoints trained on inverted-grip sim data alive.
    TODO: Drop it (and its ``flip_grip`` compose hook) when no served checkpoint needs it.

    Compose to the left of obs/action codecs::

        timing | FlipGrip() | obs & action
    """

    WIRE_NAME = 'flip_grip'

    def encode(self, data):
        # ``data`` belongs to the caller, so the flip goes on a copy.
        if obs_keys.GRIP in data:
            data = {**data, obs_keys.GRIP: 1.0 - data[obs_keys.GRIP]}
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        if obs_keys.TARGET_GRIP in data:
            data[obs_keys.TARGET_GRIP] = 1.0 - data[obs_keys.TARGET_GRIP]
        return data

    def to_spec(self):
        return {'name': self.WIRE_NAME}


def _scaled(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, width / w, height / h)
    tw, th = int(w * scale), int(h * scale)
    if (tw, th) == (w, h):
        return image
    return np.array(PilImage.fromarray(image).resize((tw, th), resample=PilImage.Resampling.BILINEAR))


class RestrictImageSize(Codec):
    """Bounds image dimensions on the rig so full-resolution frames never reach the wire.

    Images larger than ``width`` x ``height`` are scaled down to fit it, keeping aspect ratio; anything
    already within it passes through untouched. This decides bandwidth, never geometry — the model's own
    codec still resizes exactly, and serving without this codec differs only in bytes on the wire.

    Declared left of the ``remote`` marker, so the rig applies it before sending::

        ChunkPlayer() | RestrictImageSize() | remote | codec | source
    """

    WIRE_NAME = 'restrict_image_size'

    def __init__(self, width: int = 640, height: int = 640):
        self._width = width
        self._height = height

    def encode(self, data):
        return {key: self._restrict(key, value) for key, value in data.items()}

    def _restrict(self, key: str, value: Any) -> Any:
        # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
        if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
            # A TemporalStack emits a (T, H, W, 3) stack, so bound each frame rather than the stack's first axis.
            if value.ndim == 4:
                return np.stack([_scaled(frame, self._width, self._height) for frame in value])
            return _scaled(value, self._width, self._height)
        if isinstance(value, cabc.Mapping):
            return {k: self._restrict(k, v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._restrict(key, v) for v in value)
        return value

    def decode(self, data):
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        raise NotImplementedError(
            'RestrictImageSize bounds what goes on the wire and is not part of the model contract: '
            'training derives its columns from full-resolution episodes.'
        )

    def to_spec(self):
        return {'name': self.WIRE_NAME, 'args': {'width': self._width, 'height': self._height}}


class ChangeEEFrame(Codec):
    """Convert poses between the embodiment's ``default`` frame and the frame the policy speaks.

    ``transform`` expresses the policy's frame relative to ``default`` — the frame every embodiment declares in
    its model and reports poses in, so one policy-side constant serves every rig that honours the contract.
    Keys in ``keys`` go ``pose * transform`` on encode and ``pose * transform⁻¹`` on decode; absent keys and
    joint-space commands are left alone. A ``CartesianDelta`` has no anchor pose to convert against, so it
    travels with the frame it is expressed in and the driver applies it there.

    Which side of the ``remote`` marker it sits on decides who converts, the rig or the server. Compose it left
    of the observation/action codecs.
    """

    WIRE_NAME = 'change_ee_frame'

    @staticmethod
    def _move(value: Any, transform: geom.Transform3D) -> Any:
        """A pose vector or an arm command, re-expressed through ``transform``."""
        match value:
            case command.CartesianPosition(pose, mode):
                return command.CartesianPosition(pose=pose * transform, mode=mode)
            case command.CartesianDelta(delta, frame, mode):
                return command.CartesianDelta(delta=delta, frame=transform.inv * frame, mode=mode)
            case command.JointPosition() | command.JointDelta():
                return value
            case _:
                return change_frame(value, transform)

    class _ChangeEpisodeFrames(EpisodeTransform):
        """Move the episode's pose signals into the policy's frame and record where they now sit, so a later
        transform solving against the model (``IKJointsAction``) reaches the same frame.
        """

        def __init__(self, codec: 'ChangeEEFrame'):
            self._codec = codec

        def _derive_pose(self, key: str, episode):
            codec = self._codec
            return Elementwise(episode[key], lazy_sequence(partial(codec._move, transform=codec._transform)))

        def __call__(self, episode):
            codec = self._codec
            assert_default_frame(episode)
            existing = ee_frame(episode)
            if not np.allclose(existing.as_matrix, np.eye(4)):
                raise ValueError(
                    f'episode poses already sit at {existing.as_vector(_QUAT).tolist()} relative to '
                    f'{DEFAULT_FRAME!r}; ``transform`` names the policy frame from there, so re-expressing an '
                    'episode a codec already moved would train on a frame the checkpoint does not declare'
                )
            derived: dict[str, Any] = {obs_keys.EE_FRAME: FromValue(codec._transform.as_vector(_QUAT))}
            derived.update({key: partial(self._derive_pose, key) for key in codec._keys if key in episode})
            return Group(Derive(**derived), Identity())(episode)

        @property
        def meta(self):
            return self._codec.meta

    def __init__(
        self,
        transform: geom.Transform3D | cabc.Sequence[float],
        keys: tuple[str, ...] = (obs_keys.EE_POSE, obs_keys.TARGET_EE_POSE, obs_keys.ROBOT_COMMAND),
    ):
        if not isinstance(transform, geom.Transform3D):  # the ``[tx,ty,tz,qw,qx,qy,qz]`` vector a wire spec carries
            transform = geom.Transform3D.from_vector(np.asarray(transform, dtype=np.float64), _QUAT)
        self._transform = transform
        self._keys = tuple(keys)

    def _apply(self, data: dict, transform: geom.Transform3D) -> dict:
        moved = {key: self._move(data[key], transform) for key in self._keys if key in data}
        return {**data, **moved} if moved else data

    def encode(self, data):
        return self._apply(data, self._transform)

    def _decode_single(self, data: dict) -> dict:
        return self._apply(data, self._transform.inv)

    @property
    def training_encoder(self) -> EpisodeTransform:
        return ChangeEEFrame._ChangeEpisodeFrames(self)

    @property
    def meta(self):
        return {obs_keys.EE_FRAME: self._transform.as_vector(_QUAT).tolist()}

    def to_spec(self):
        # Lists, not tuples, so the spec is identical before and after a wire round-trip.
        return {
            'name': self.WIRE_NAME,
            'args': {'transform': self._transform.as_vector(_QUAT).tolist(), 'keys': list(self._keys)},
        }


class SetControlMode(Codec):
    """Sets the control mode a chunk executes under on every robot command it carries (inference only).

    Composes left of an action decoder (``SetControlMode(mode) | action``). Every command of every arm
    carries the mode, not only the first of the unsuffixed channel.
    """

    def __init__(self, mode: command.ControlModeType):
        self._mode = mode

    def encode(self, data):
        return data

    def _decode_single(self, data: dict) -> dict:
        # The command family also holds the pose/joint vectors a recording unfolds into, so what carries a
        # mode is decided by type rather than by name.
        stamped = {
            key: replace(cmd, mode=self._mode)
            for key, cmd in data.items()
            if obs_keys.is_robot_command(key) and isinstance(cmd, command.CommandType)
        }
        return {**data, **stamped} if stamped else data
