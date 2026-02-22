"""Composable codec for encoding observations and decoding actions.

A Codec pairs an observation encoder (for training and inference) with an action decoder.
Codecs compose via ``|``: ``observation_codec | action_codec | timing`` produces a single
codec that encodes left-to-right and decodes right-to-left.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Group
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
        return Group(self._left.training_encoder, self._right.training_encoder)

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
        return Derive(meta=self.meta)

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


class RecordingCodec(Codec):
    """Transparent ``Codec`` wrapper that logs the encode/decode cycle to per-episode ``.rrd`` files.

    Call ``new_episode()`` before each inference session to start a fresh recording.
    """

    def __init__(self, inner: Codec, recording_dir: str | Path):
        self._inner = inner
        self._dir = Path(recording_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._action_fps: float = inner.meta.get('action_fps', 15.0)
        self._rec: Any = None
        self._counter = 0
        self._step = 0
        self._time_ns: int = 0
        self._inference_time_ns: int | None = None
        self._image_paths: list[str] = []
        self._numeric_paths: list[str] = []

    def new_episode(self):
        """Start recording a new episode."""
        self._counter += 1
        self._rec = rr.new_recording(application_id='positronic_inference')
        self._rec.save(str(self._dir / f'episode_{self._counter:04d}.rrd'))
        self._step = 0
        self._image_paths.clear()
        self._numeric_paths.clear()

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
            if isinstance(value, dict):
                self._log(f'{prefix}/{key}', value)
            elif isinstance(value, np.ndarray):
                squeezed = _squeeze_batch(value)
                if squeezed.ndim == 3 and squeezed.shape[-1] == 3:
                    path = f'{prefix}/image/{key}'
                    rr.log(path, rr.Image(squeezed).compress())
                    self._image_paths.append(path)
                else:
                    log_numeric_series(f'{prefix}/{key}', value)
                    self._numeric_paths.append(f'{prefix}/{key}')
            elif isinstance(value, int | float | np.integer | np.floating):
                log_numeric_series(f'{prefix}/{key}', value)
                self._numeric_paths.append(f'{prefix}/{key}')

    def _send_blueprint(self):
        bp = _build_blueprint(list(dict.fromkeys(self._image_paths)), list(dict.fromkeys(self._numeric_paths)))
        if bp is not None:
            rr.send_blueprint(bp)

    def encode(self, data: dict) -> dict:
        self._time_ns = data.get('__time_ns__', time.time_ns())
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
        first_decoded = decoded[0] if isinstance(decoded, list) else decoded
        dt_ns = int(1e9 / self._action_fps)
        with self._rec:
            for i, action in enumerate(actions):
                inf_t = self._inference_time_ns + i * dt_ns if self._inference_time_ns is not None else None
                self._set_timelines(self._time_ns + i * dt_ns, inf_t)
                self._log('model', action)
            self._set_timelines(self._time_ns, self._inference_time_ns)
            self._log('decoded', first_decoded)
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

    def wrap(self, policy: Policy) -> Policy:
        self.new_episode()
        return super().wrap(policy)

    def dummy_encoded(self, data=None):
        return self._inner.dummy_encoded(data)
