"""Dataset utilities for Positronic dataset visualization (images-only)."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Signal
from positronic.dataset.video import VideoSignal
from positronic.utils.rerun_compat import flatten_numeric, log_numeric_series, log_series_styles, set_timeline_time


@dataclass
class _SignalInfo:
    name: str
    kind: Literal['video', 'vector', 'scalar', 'tensor']
    shape: tuple[int, ...] | None
    dtype: str | None


def _infer_signal_info(ep: Episode, name: str) -> _SignalInfo:
    v = ep[name]
    if isinstance(v, VideoSignal):
        # Decode first frame to get shape
        frame, _ts = v[0]
        h, w = int(frame.shape[0]), int(frame.shape[1])
        return _SignalInfo(name=name, kind='video', shape=(h, w, 3), dtype='uint8')
    elif isinstance(v, Signal):
        # Peek first value to infer shape/dtype
        if len(v) == 0:
            return _SignalInfo(name=name, kind='scalar', shape=(), dtype=None)
        val, _ts = v[0]
        match val:
            case np.ndarray() if val.ndim == 0:
                return _SignalInfo(name=name, kind='scalar', shape=(), dtype=str(val.dtype))
            case np.ndarray() if val.ndim == 1:
                return _SignalInfo(name=name, kind='vector', shape=tuple(map(int, val.shape)), dtype=str(val.dtype))
            case np.ndarray():
                return _SignalInfo(name=name, kind='tensor', shape=tuple(map(int, val.shape)), dtype=str(val.dtype))
            case _:  # Python scalar
                return _SignalInfo(name=name, kind='scalar', shape=(), dtype=type(val).__name__)
    else:
        # Static item; ignore here
        return _SignalInfo(name=name, kind='static', shape=None, dtype=None)


# TODO: Make this part of dataset.Signal, dataset.Episode and dataset.Dataset APIs
def _infer_features(ep: Episode) -> dict[str, dict[str, Any]]:
    """Infer feature metadata (shapes/dtypes) from an episode."""
    features: dict[str, dict[str, Any]] = {}
    for name in ep.signals.keys():
        info = _infer_signal_info(ep, name)
        if info.kind == 'video' and info.shape is not None:
            features[name] = {'dtype': 'image', 'shape': list(info.shape)}
        elif info.kind in ('vector', 'tensor', 'scalar'):
            shape = [] if info.shape is None else list(info.shape)
            features[name] = {'dtype': 'float32' if info.dtype is None else info.dtype, 'shape': shape}
    return features


def get_dataset_info(ds: LocalDataset) -> dict[str, Any]:
    """Return basic info + feature descriptions for the dataset."""
    num_eps = len(ds)
    features: dict[str, dict[str, Any]] = {}

    if num_eps > 0:
        ep0 = ds[0]
        features = _infer_features(ep0)

    return {
        'root': str(ds.root),
        'num_episodes': num_eps,
        'features': features,
    }


def get_episodes_list(ds: LocalDataset) -> list[dict[str, Any]]:
    return [
        {
            'index': idx,
            'duration': ep.duration_ns / 1e9,
            'task': ep.static.get('task', None),
        }
        for idx, ep in enumerate(ds)
    ]


def _collect_signal_groups(ep: Episode) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (video_names, signal_names, signal_dims) for an episode.

    signal_dims gives the number of plotted series per non-video signal (1 for scalar).
    """
    video_names: list[str] = []
    signal_names: list[str] = []
    signal_dims: dict[str, int] = {}
    for name, sig in ep.signals.items():
        if isinstance(sig, VideoSignal):
            try:
                frame, _ = sig[0]
                h, w = frame.shape[:2]
                video_names.append(name)
            except Exception:
                continue
        else:
            signal_names.append(name)
            # infer channel count for legend visibility
            try:
                if len(sig) == 0:
                    signal_dims[name] = 1
                else:
                    v0, _ = sig[0]
                    arr = flatten_numeric(v0)
                    signal_dims[name] = int(arr.size) if arr is not None else 1
            except Exception:
                signal_dims[name] = 1
    return video_names, signal_names, signal_dims


def _build_blueprint(video_names: list[str], signal_names: list[str], signal_dims: dict[str, int]) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in video_names]

    per_signal_views = []
    for sig in signal_names:
        # Legends visible only if a signal has more than one plotted series
        show_legend = signal_dims.get(sig, 1) > 1
        per_signal_views.append(
            rrb.TimeSeriesView(
                name=sig,
                origin=f'/signals/{sig}',
                plot_legend=rrb.PlotLegend(visible=show_legend),
            )
        )

    grid_items = []
    if per_signal_views:
        grid_items.append(rrb.Grid(*per_signal_views))
    if image_views:
        grid_items.append(rrb.Grid(*image_views))

    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Collapsed),
        rrb.Grid(*grid_items, column_shares=[1, 2]),
    )


def _setup_series_names(ep: Episode, signal_names: list[str]) -> None:
    """Log static series metadata with short names ('0','1',...) per signal."""
    for key in signal_names:
        try:
            sig = ep.signals[key]
            if len(sig) == 0:
                dims = 1
            else:
                val, _ = sig[0]
                arr = flatten_numeric(val)
                dims = int(arr.size) if arr is not None else 1
        except Exception:
            dims = 1
        names = [str(i) for i in range(max(1, dims))]
        log_series_styles(f'/signals/{key}', names, static=True)


def _episode_log_entries(ep: Episode, video_names: list[str], signal_names: list[str]):
    for key in video_names:
        for frame, ts_ns in ep.signals[key]:
            yield ('video', key, frame, ts_ns)

    for key in signal_names:
        for value, ts_ns in ep.signals[key]:
            yield ('numeric', key, value, ts_ns)


class _BinaryStreamDrainer:
    def __init__(self, stream: rr.recording_stream.BinaryStream, min_bytes: int):
        self._stream = stream
        self._min_bytes = max(1, min_bytes)
        self._buffer = bytearray()

    def drain(self, force: bool = False) -> Iterator[bytes]:
        # Always flush to get the latest data
        if force:
            self._stream.flush()
        chunk = self._stream.read(flush=force)
        if chunk:
            self._buffer.extend(chunk)
        # Yield in min_bytes-sized chunks
        while len(self._buffer) >= self._min_bytes:
            to_yield = self._buffer[: self._min_bytes]
            yield bytes(to_yield)
            self._buffer = self._buffer[self._min_bytes :]
        # On force, yield any remaining bytes
        if force and self._buffer:
            yield bytes(self._buffer)
            self._buffer.clear()


@rr.recording_stream.recording_stream_generator_ctx
def stream_episode_rrd(ds: LocalDataset, episode_id: int) -> Iterator[bytes]:
    """Yield an episode RRD as chunks while it is being generated."""

    ep = ds[episode_id]
    logging.info(f'Streaming RRD for episode {episode_id}')

    recording_id = f'positronic_ds_{Path(ds.root).name}_episode_{episode_id}'
    rec = rr.new_recording(application_id=recording_id)
    drainer = _BinaryStreamDrainer(rec.binary_stream(), min_bytes=2**20)

    with rec:
        video_names, signal_names, signal_dims = _collect_signal_groups(ep)
        rr.send_blueprint(_build_blueprint(video_names, signal_names, signal_dims))
        yield from drainer.drain()

        _setup_series_names(ep, signal_names)
        yield from drainer.drain()

        for kind, key, payload, ts_ns in _episode_log_entries(ep, video_names, signal_names):
            set_timeline_time('time', np.datetime64(ts_ns, 'ns'))
            if kind == 'numeric':
                log_numeric_series(f'/signals/{key}', payload)
            else:
                rr.log(key, rr.Image(payload).compress())
            yield from drainer.drain()

    yield from drainer.drain(force=True)
