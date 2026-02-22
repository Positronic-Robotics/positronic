"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Codecs compose via ``|``: ``observation_codec | action_codec | timing`` produces a single
codec that encodes left-to-right and decodes right-to-left.
"""

import itertools
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.transforms.episode import EpisodeTransform, Identity
from positronic.policy.base import Policy
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


class Codec(ABC):
    """Base class for observation/action codecs.

    Subclasses implement ``encode`` (observation encoding or pass-through for action codecs)
    and optionally ``_decode_single`` (action decoding). The ``training_encoder`` property
    returns an ``EpisodeTransform`` used by the training pipeline to derive dataset columns.

    Reserved ``meta`` key (part of the remote inference protocol):

    ``image_sizes``
        Expected image dimensions for raw observation inputs. Used by ``RemotePolicy``
        to downscale images before sending them to the server, reducing bandwidth.
        Either a ``(width, height)`` tuple (same size for all images) or a dict mapping
        raw input keys to ``(width, height)`` tuples (per-image sizes).
    """

    @abstractmethod
    def encode(self, data: dict) -> dict: ...

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            return [self.decode(d, context=context) for d in data]
        return self._decode_single(data, context)

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

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

    def wrap(self, policy: Policy) -> Policy:
        return _WrappedPolicy(policy, self)

    @final
    def __or__(self, other: 'Codec') -> 'Codec':
        return _ComposedCodec(self, other)


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


class ActionTiming(Codec):
    """Attaches timings to decoded actions and truncates action sequences to a specified horizon.

    # TODO: Split into two codecs: ActionTimestamp (stamps actions using fps) and
    # ActionHorizon (truncates by timestamp < horizon_sec). This lets horizon be
    # composed independently — e.g. wrapping a RemotePolicy with just ActionHorizon
    # instead of duplicating truncation logic in RemotePolicy.select_action.

    At inference time, truncates action chunks to ``horizon`` seconds and stamps each action
    with a ``timestamp`` field. At training time, surfaces ``action_fps`` (and optionally
    ``action_horizon_sec``) as transform metadata so the training pipeline can read it.
    """

    def __init__(self, *, fps: float, horizon_sec: float | None = None):
        self._fps = fps
        self._horizon_sec = horizon_sec

    def encode(self, data):
        return data

    def decode(self, data, *, context=None):
        if isinstance(data, list):
            dt = 1.0 / self._fps
            for i, d in enumerate(data):
                d['timestamp'] = i * dt
            if self._horizon_sec is not None:
                data = [d for d in data if d['timestamp'] < self._horizon_sec]
            return data
        data['timestamp'] = 0.0
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    @property
    def meta(self):
        result = {'action_fps': self._fps}
        if self._horizon_sec is not None:
            result['action_horizon_sec'] = self._horizon_sec
        return result


class _WrappedPolicy(Policy):
    """Policy wrapped with a codec: encodes observations, decodes actions."""

    def __init__(self, policy: Policy, codec: Codec):
        self._policy = policy
        self._codec = codec

    def select_action(self, obs):
        encoded = self._codec.encode(obs)
        action = self._policy.select_action(encoded)
        return self._codec.decode(action, context=obs)

    def reset(self):
        self._policy.reset()

    @property
    def meta(self):
        return self._policy.meta | self._codec.meta

    def close(self):
        self._policy.close()


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


class _RecordingSession(Codec):
    """Per-session codec that logs the encode/decode cycle to a single ``.rrd`` file.

    Created by ``RecordingCodec._new_session()`` — one per episode, with independent state.
    """

    def __init__(self, inner: Codec, rec: Any, *, action_fps: float):
        self._inner = inner
        self._rec = rec
        self._action_fps = action_fps
        self._step = 0
        # Stashed between encode() and decode() — _WrappedPolicy calls them sequentially
        self._time_ns: int = 0
        self._inference_time_ns: int | None = None
        # Accumulated across encode+decode; used once at step 0 to build the blueprint
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
            if (key.startswith('__') and key.endswith('__')) or isinstance(value, str):
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

    def encode(self, data: dict) -> dict:
        # __wall_time_ns__ / __inference_time_ns__ injected by Inference.run(); ignored by inner codecs
        self._time_ns = data.get('__wall_time_ns__', time.time_ns())
        self._inference_time_ns = data.get('__inference_time_ns__')
        encoded = self._inner.encode(data)
        with self._rec:
            self._set_timelines(self._time_ns, self._inference_time_ns)
            self._log('input', data)
            self._log('encoded', encoded)
        return encoded

    def decode(self, data, *, context=None):
        decoded = self._inner.decode(data, context=context)
        actions = data if isinstance(data, list) else [data]
        decoded_list = decoded if isinstance(decoded, list) else [decoded]
        dt_ns = int(1e9 / self._action_fps)
        with self._rec:
            # Log raw and decoded actions at future timestamps (one per chunk step)
            for i, (action, dec) in enumerate(zip(actions, decoded_list, strict=False)):
                inf_t = self._inference_time_ns + i * dt_ns if self._inference_time_ns is not None else None
                self._set_timelines(self._time_ns + i * dt_ns, inf_t)
                self._log('model', action)
                self._log('decoded', dec)
            if self._step == 0:
                self._send_blueprint()
        self._step += 1
        return decoded

    @property
    def training_encoder(self):
        return self._inner.training_encoder

    @property
    def meta(self):
        return self._inner.meta

    def dummy_encoded(self, data=None):
        return self._inner.dummy_encoded(data)


class RecordingCodec(Codec):
    """Transparent ``Codec`` wrapper that logs the encode/decode cycle to per-episode ``.rrd`` files.

    Each ``reset()`` on the wrapped policy creates a ``_RecordingSession`` with independent
    state, so concurrent sessions don't interfere with each other.
    """

    def __init__(self, inner: Codec, recording_dir: str | Path):
        self._inner = inner
        self._dir = Path(recording_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._action_fps: float = inner.meta.get('action_fps', 15.0)
        self._counter = itertools.count(1)

    def _new_session(self) -> _RecordingSession:
        episode_num = next(self._counter)
        rec = rr.new_recording(application_id='positronic_inference')
        rec.save(str(self._dir / f'episode_{episode_num:04d}.rrd'))
        return _RecordingSession(self._inner, rec, action_fps=self._action_fps)

    def encode(self, data: dict) -> dict:
        return self._inner.encode(data)

    def decode(self, data, *, context=None):
        return self._inner.decode(data, context=context)

    @property
    def training_encoder(self):
        return self._inner.training_encoder

    @property
    def meta(self):
        return self._inner.meta

    def wrap(self, policy: Policy) -> Policy:
        return _RecordingPolicy(policy, self)

    def dummy_encoded(self, data=None):
        return self._inner.dummy_encoded(data)


class _RecordingPolicy(Policy):
    """Policy wrapper that creates a fresh ``_RecordingSession`` on each ``reset()``."""

    def __init__(self, policy: Policy, codec: RecordingCodec):
        self._policy = policy
        self._codec = codec
        self._active: Policy | None = None

    def select_action(self, obs):
        if self._active is None:
            raise RuntimeError('reset() must be called before select_action()')
        return self._active.select_action(obs)

    def reset(self):
        session = self._codec._new_session()
        self._active = _WrappedPolicy(self._policy, session)
        self._active.reset()

    @property
    def meta(self):
        if self._active is not None:
            return self._active.meta
        return self._codec.meta

    def close(self):
        if self._active is not None:
            self._active.close()
