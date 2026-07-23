"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Two composition operators:

- ``|`` (sequential): left's output feeds into right. Use for codecs that modify data
  before others see it (e.g. BinarizeGrip before observation/action encoders).
- ``&`` (parallel): both see the same input, outputs merged. Use for independent codecs
  (e.g. observation encoder & action decoder).
"""

from typing import Any, final

import numpy as np

from positronic.dataset.transforms import Elementwise
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Group, Identity
from positronic.policy.base import PAR, SEQ, DelegatingSession, PolicyWrapper, Session, _Pipeline
from positronic.utils import merge_dicts


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


class Codec(PolicyWrapper):
    """Base class for observation/action codecs.

    Subclasses override ``encode`` (observation encoding) and/or ``_decode_single``
    (action decoding). The ``training_encoder`` property
    returns an ``EpisodeTransform`` used by the training pipeline to derive dataset columns.

    Reserved ``meta`` key (part of the remote inference protocol):

    ``image_sizes``
        Expected image dimensions for raw observation inputs. Used by ``RemotePolicy``
        to downscale images before sending them to the server, reducing bandwidth.
        Either a ``(width, height)`` tuple (same size for all images) or a dict mapping
        raw input keys to ``(width, height)`` tuples (per-image sizes).
    """

    def encode(self, data: dict) -> dict:
        return {}

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            return [self.decode(d, context=context) for d in data]
        return self._decode_single(data, context)

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return {}

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive()

    @property
    def meta(self) -> dict:
        return {}

    def dummy_encoded(self, data: dict | None = None) -> dict:
        """Return a dummy version of what ``encode()`` would produce.

        Each codec contributes its part of the encoded output. The default
        pass-through returns the input unchanged — codecs that don't transform
        observations (action decoders, timing) inherit this behavior.
        Composed codecs pipeline left-to-right, mirroring ``encode()``.
        """
        return data or {}

    def wrap_session(self, inner: Session, context, now):
        return _CodecSession(inner, self)

    @final
    def __or__(self, other):
        if isinstance(other, Codec):
            return _ComposedCodec(self, other)
        if isinstance(other, PolicyWrapper):
            # Mixed Codec | non-Codec-wrapper → generic pipeline, not a Codec.
            return _Pipeline((self, *other._pipeline_components()))
        return NotImplemented

    @final
    def __and__(self, other):
        if isinstance(other, Codec):
            return _ParallelCodec(self, other)
        return NotImplemented


class _CodecSession(DelegatingSession):
    """Session wrapped with a codec: encodes observations, decodes actions."""

    def __init__(self, inner: Session, codec: 'Codec'):
        super().__init__(inner)
        self._codec = codec

    def __call__(self, obs):
        encoded = self._codec.encode(obs)
        action = self._inner(encoded)
        if action is None:
            return None
        return self._codec.decode(action, context=obs)

    @property
    def meta(self):
        return self._inner.meta | self._codec.meta


class _ComposedCodec(Codec):
    """Two codecs composed via ``|``. Encodes left-to-right, decodes right-to-left."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return self._right.encode(self._left.encode(data))

    def decode(self, data, *, context=None):
        return self._left.decode(self._right.decode(data, context=context), context=context)

    @property
    def training_encoder(self):
        return self._left.training_encoder | self._right.training_encoder

    @property
    def meta(self):
        result: dict[str, Any] = {}
        merge_dicts(result, self._left.meta)
        merge_dicts(result, self._right.meta)
        return result

    def to_spec(self):
        return {SEQ: [self._left.to_spec(), self._right.to_spec()]}

    def dummy_encoded(self, data=None):
        return self._right.dummy_encoded(self._left.dummy_encoded(data))


class _ParallelCodec(Codec):
    """Two codecs composed via ``&``. Both see the same input, outputs merged."""

    def __init__(self, left: Codec, right: Codec):
        self._left = left
        self._right = right

    def encode(self, data):
        return {**self._left.encode(data), **self._right.encode(data)}

    def decode(self, data, *, context=None):
        left_out = self._left.decode(data, context=context)
        right_out = self._right.decode(data, context=context)
        if isinstance(data, list):
            return [{**lf, **rt} for lf, rt in zip(left_out, right_out, strict=True)]
        return {**left_out, **right_out}

    @property
    def training_encoder(self):
        return self._left.training_encoder & self._right.training_encoder

    @property
    def meta(self):
        result: dict[str, Any] = {}
        merge_dicts(result, self._left.meta)
        merge_dicts(result, self._right.meta)
        return result

    def to_spec(self):
        return {PAR: [self._left.to_spec(), self._right.to_spec()]}

    def dummy_encoded(self, data=None):
        return {**self._left.dummy_encoded(data), **self._right.dummy_encoded(data)}


def is_action(entry: dict) -> bool:
    """True for a real command entry, False for a keyless validity sentinel.

    Time codecs close a chunk with a timestamp-only sentinel marking where its validity ends
    (see ``ActionTimestamp``). Consumers that classify or plot per-command fields skip the
    sentinel through this predicate rather than hard-coding its shape.
    """
    return bool(entry.keys() - {'timestamp'})


class ActionTimestamp(Codec):
    """Stamps each decoded action with a relative ``timestamp`` (seconds from trajectory start).

    Assigns ``timestamp = i * (1/fps)`` starting at 0. The harness converts these
    relative timestamps to absolute wall time at emission, anchoring execution to
    inference-finish rather than inference-start.

    A K-action chunk covers K periods, so the list is closed with a sentinel entry —
    a dict carrying only ``timestamp = K * (1/fps)`` and no command keys — stating when
    the chunk's validity ends. The scheduler reads that end from the last entry, so the
    final action gets a full period before re-inference. The sentinel carries no command
    key, so the key-filtered demux emits it on no channel: drivers and recordings never
    see it.

    At training time, surfaces ``action_fps`` as transform metadata.
    """

    def __init__(self, *, fps: float):
        self._fps = fps
        self._dt = 1.0 / fps

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        # Build fresh entries rather than stamping in place: a session may hand back a cached template
        # list, and appending the sentinel to it would regrow the chunk on every re-inference.
        if isinstance(data, list):
            stamped = [{**d, 'timestamp': i * self._dt} for i, d in enumerate(data)]
            if stamped:
                stamped.append({'timestamp': len(stamped) * self._dt})
            return stamped
        return {**data, 'timestamp': 0}

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_fps': self._fps}

    def to_spec(self):
        return {'name': 'action_timestamp', 'args': {'fps': self._fps}}


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

    def __init__(self, horizon_sec: float):
        self._horizon_sec = horizon_sec

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            # Treat untimestamped actions as t=0 so they always pass the horizon
            # (servers may apply horizon truncation before stamping).
            kept = [d for d in data if d.get('timestamp', 0.0) < self._horizon_sec]
            if len(kept) < len(data):
                kept.append({'timestamp': self._horizon_sec})
            return kept
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_horizon_sec': self._horizon_sec}

    def to_spec(self):
        return {'name': 'action_horizon', 'args': {'horizon_sec': self._horizon_sec}}


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

    def __init__(self, keys: tuple[str, ...], threshold: float = 0.5):
        self._keys = keys
        self._threshold = threshold

    def encode(self, data):
        return data

    def _decode_single(self, data: dict, context: dict | None) -> dict:
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
        return {'name': 'binarize_grip_training', 'args': {'keys': list(self._keys), 'threshold': self._threshold}}


class BinarizeGripInference(Codec):
    """Threshold grip in decoded actions at inference time.

    Compose to the left of action codecs so it runs after action decoding::

        timing | BinarizeGripInference() | obs & action
    """

    def __init__(self, threshold: float = 0.5, key: str = 'target_grip'):
        self._threshold = threshold
        self._key = key

    def encode(self, data):
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        if self._key in data:
            data[self._key] = 1.0 if data[self._key] > self._threshold else 0.0
        return data

    def to_spec(self):
        return {'name': 'binarize_grip_inference', 'args': {'threshold': self._threshold, 'key': self._key}}


class FlipGrip(Codec):
    """Serve a checkpoint that speaks the inverted grip convention (1 = open) on the canonical
    1 = closed wire: flips ``grip`` entering the model and ``target_grip`` leaving it.

    HACK: exists only to keep checkpoints trained on inverted-grip sim data alive.
    TODO: Drop it (and its ``flip_grip`` compose hook) when no served checkpoint needs it.

    Compose to the left of obs/action codecs::

        timing | FlipGrip() | obs & action
    """

    def encode(self, data):
        # Copy: the original dict is also the decode ``context`` and the raw recording tap's input.
        if 'grip' in data:
            data = {**data, 'grip': 1.0 - data['grip']}
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        if 'target_grip' in data:
            data['target_grip'] = 1.0 - data['target_grip']
        return data

    def to_spec(self):
        return {'name': 'flip_grip'}
