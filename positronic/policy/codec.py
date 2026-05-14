"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Two composition operators:

- ``|`` (sequential): left's output feeds into right. Use for codecs that modify data
  before others see it (e.g. BinarizeGrip before observation/action encoders).
- ``&`` (parallel): both see the same input, outputs merged. Use for independent codecs
  (e.g. observation encoder & action decoder).
"""

import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Any, final

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.transforms import Elementwise
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Group, Identity
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy.base import DelegatingSession, PolicyWrapper, Session, _Pipeline
from positronic.utils import merge_dicts
from positronic.utils.rerun_compat import log_numeric_series, set_timeline_sequence, set_timeline_time


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

    def wrap_session(self, inner: Session, context):
        return _CodecSession(inner, self)

    @final
    def __or__(self, other):
        if isinstance(other, Codec):
            return _ComposedCodec(self, other)
        if isinstance(other, PolicyWrapper):
            # Mixed Codec | non-Codec-wrapper → generic pipeline (no longer a Codec).
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

    def dummy_encoded(self, data=None):
        return {**self._left.dummy_encoded(data), **self._right.dummy_encoded(data)}


class ActionTimestamp(Codec):
    """Stamps each decoded action with a relative ``timestamp`` (seconds from trajectory start).

    Assigns ``timestamp = i * (1/fps)`` starting at 0. The harness converts these
    relative timestamps to absolute wall time at emission, anchoring execution to
    inference-finish rather than inference-start.

    At training time, surfaces ``action_fps`` as transform metadata.
    """

    def __init__(self, *, fps: float):
        self._fps = fps
        self._dt = 1.0 / fps

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            for i, d in enumerate(data):
                d['timestamp'] = i * self._dt
            return data
        data['timestamp'] = 0
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_fps': self._fps}


class ActionHorizon(Codec):
    """Truncates action chunks to a time horizon.

    Keeps only actions whose (relative) ``timestamp`` is within ``horizon_sec``
    of trajectory start. Single actions pass through.
    At training time, surfaces ``action_horizon_sec`` as transform metadata.
    """

    def __init__(self, horizon_sec: float):
        self._horizon_sec = horizon_sec

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            return [d for d in data if d['timestamp'] < self._horizon_sec]
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        return {'action_horizon_sec': self._horizon_sec}


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


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """Remove leading size-1 dims from a potential image array."""
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _as_image(value: Any) -> np.ndarray | None:
    """Return squeezed RGB array if *value* looks like an image, else None."""
    if not isinstance(value, np.ndarray):
        return None
    squeezed = _squeeze_batch(value)
    if squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        return squeezed
    return None


def _as_numeric(value: Any) -> Any | None:
    """Return a loggable numeric form of *value*, or None if not numeric."""
    if isinstance(value, np.ndarray | int | float | np.integer | np.floating):
        return value
    if isinstance(value, list | tuple):
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float64)
    return None


def _build_blueprint(image_paths: list[str], numeric_paths: list[str]) -> rrb.Blueprint | None:
    if not image_paths and not numeric_paths:
        return None
    image_views = [rrb.Spatial2DView(name=p.rsplit('/', 1)[-1], origin=p) for p in image_paths]
    numeric_views = [rrb.TimeSeriesView(name=p.rsplit('/', 1)[-1], origin=p) for p in numeric_paths]
    grid_items: list[Any] = []
    if image_views:
        grid_items.append(rrb.Grid(*image_views))
    if numeric_views:
        grid_items.append(rrb.Grid(*numeric_views))
    return rrb.Blueprint(rrb.Grid(*grid_items))


class _RecordingSession(DelegatingSession):
    """Per-session Session wrapper that logs obs + actions to a single ``.rrd`` file.

    Created by ``RecordingWrapper.wrap_session()`` — one per episode, with independent state.
    """

    def __init__(self, inner: Session, rec: Any):
        super().__init__(inner)
        self._rec = rec
        self._step = 0
        self._image_paths: list[str] = []
        self._numeric_paths: list[str] = []

    def _set_timelines(self, time_ns: int, inference_time_ns: int | None):
        set_timeline_time('wall_time', time_ns)
        if inference_time_ns is not None:
            set_timeline_time('inference_time', inference_time_ns)
        set_timeline_sequence('step', self._step)

    def _log(self, prefix: str, data: dict):
        """Recursively log *data* under *prefix*, accumulating entity paths."""
        for key, value in data.items():
            if key.endswith('_time_ns') or isinstance(value, str):
                continue
            path = f'{prefix}/{key}'
            if isinstance(value, dict):
                self._log(path, value)
            elif (img := _as_image(value)) is not None:
                img_path = f'{prefix}/image/{key}'
                rr.log(img_path, rr.Image(img).compress())
                self._image_paths.append(img_path)
            elif (num := _as_numeric(value)) is not None:
                log_numeric_series(path, num)
                self._numeric_paths.append(path)

    def _send_blueprint(self):
        # Deduplicate: _log appends on every step, but entity paths are stable after step 0
        bp = _build_blueprint(list(dict.fromkeys(self._image_paths)), list(dict.fromkeys(self._numeric_paths)))
        if bp is not None:
            rr.send_blueprint(bp)

    def __call__(self, obs):
        time_ns = obs.get('wall_time_ns', time.time_ns())
        inference_time_ns = obs.get('inference_time_ns')

        with self._rec:
            self._set_timelines(time_ns, inference_time_ns)
            self._log('input', obs)

        actions = self._inner(obs)

        if actions is not None:
            with self._rec:
                for action in actions:
                    loggable = {
                        k: roboarm_command.to_wire(v) if isinstance(v, roboarm_command.CommandType) else v
                        for k, v in action.items()
                    }
                    rel_sec = action.get('timestamp', 0.0)
                    dt_ns = int(rel_sec * 1e9)
                    inf_t = inference_time_ns + dt_ns if inference_time_ns is not None else None
                    self._set_timelines(time_ns + dt_ns, inf_t)
                    self._log('action', loggable)
                if self._step == 0:
                    self._send_blueprint()

        self._step += 1
        return actions


class RecordingWrapper(PolicyWrapper):
    """PolicyWrapper that logs each session's obs + actions to per-episode ``.rrd`` files.

    Each ``new_session()`` creates a fresh ``_RecordingSession`` with independent state,
    so concurrent sessions don't interfere. Place in a pipeline wherever you want the
    capture point — outermost (raw obs + decoded actions) or inside a codec chain.
    """

    def __init__(self, recording_dir: str | Path):
        self._dir = Path(recording_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._counter = itertools.count(1)

    def wrap_session(self, inner: Session, context) -> Session:
        episode_num = next(self._counter)
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        rec = rr.RecordingStream(application_id='positronic_inference')
        rec.save(str(self._dir / f'{ts}_{episode_num:04d}.rrd'))
        return _RecordingSession(inner, rec)
