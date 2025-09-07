"""Dataset utilities for Positronic dataset visualization (images-only)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Signal
from positronic.dataset.video import VideoSignal


@dataclass
class _SignalInfo:
    name: str
    kind: str  # 'video' | 'vector' | 'scalar' | 'tensor'
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
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                return _SignalInfo(name=name, kind='scalar', shape=(), dtype=str(val.dtype))
            elif val.ndim == 1:
                return _SignalInfo(name=name, kind='vector', shape=tuple(map(int, val.shape)), dtype=str(val.dtype))
            else:
                return _SignalInfo(name=name, kind='tensor', shape=tuple(map(int, val.shape)), dtype=str(val.dtype))
        else:
            # Python scalar
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
    return [{
        'index': idx,
        'duration': ep.duraion_ns // 1e9,
        'task': ep.static.get('task', None),
    } for idx, ep in enumerate(ds)]


def _collect_signal_groups(ep: Episode) -> Tuple[list[str], list[str], dict[str, int]]:
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
                    if isinstance(v0, np.ndarray):
                        signal_dims[name] = int(v0.size) if v0.ndim >= 1 else 1
                    else:
                        signal_dims[name] = 1
            except Exception:
                signal_dims[name] = 1
    return video_names, signal_names, signal_dims


def _build_blueprint(task: Optional[str], video_names: list[str], signal_names: list[str],
                     signal_dims: dict[str, int]) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k.replace('_', ' ').title(), origin=f'/{k}') for k in video_names]

    per_signal_views = []
    for sig in signal_names:
        # Legends visible only if a signal has more than one plotted series
        show_legend = (signal_dims.get(sig, 1) > 1)
        per_signal_views.append(
            rrb.TimeSeriesView(
                name=sig.replace('_', ' ').title(),
                origin=f'/signals/{sig}',
                plot_legend=rrb.PlotLegend(visible=show_legend),
            ))

    grid_items = []
    if per_signal_views:
        grid_items.append(rrb.Grid(*per_signal_views))
    if image_views:
        grid_items.append(rrb.Grid(*image_views))

    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Expanded),
        rrb.Grid(*grid_items, column_shares=[1, 2]),
    )


def _log_single_signal(key: str, val: Any) -> None:
    """Log a single non-video signal using rr.Scalars with a vector.

    Scalars are wrapped into a length-1 float32 array; tensors are flattened.
    """
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            val = np.asarray([float(val)])
        elif val.ndim >= 2:
            val = val.reshape(-1)
    else:
        try:
            val = np.asarray([float(val)])
        except Exception:
            return
    rr.log(f'/signals/{key}', rr.Scalars(val))


def _setup_series_names(ep: Episode, signal_names: list[str]) -> None:
    """Log static SeriesLines with short names ('0','1',...) per signal.

    This controls how multi-channel Scalars are labeled in the plot legend and tooltips.
    """
    for key in signal_names:
        try:
            sig = ep.signals[key]
            if len(sig) == 0:
                dims = 1
            else:
                val, _ = sig[0]
                if isinstance(val, np.ndarray):
                    dims = int(val.size) if val.ndim >= 1 else 1
                else:
                    dims = 1
        except Exception:
            dims = 1
        names = [str(i) for i in range(max(1, dims))]
        try:
            rr.log(f'/signals/{key}', rr.SeriesLines(names=names), static=True)
        except Exception:
            pass


def _log_episode(ep: Episode, video_names: list[str], signal_names: list[str]) -> None:
    for key in signal_names:
        for v, t in ep.signals[key]:
            rr.time.set_time('time', timestamp=np.datetime64(t, 'ns'))
            _log_single_signal(key, v)

    for key in video_names:
        for frame, ts_ns in ep.signals[key]:
            rr.time.set_time('time', timestamp=np.datetime64(ts_ns, 'ns'))
            rr.log(key, rr.Image(frame).compress())


def generate_episode_rrd(ds: LocalDataset, episode_id: int, cache_path: str) -> str:
    """Generate an RRD for an episode and return its path (cached)."""
    if os.path.exists(cache_path):
        logging.info(f'Using cached RRD file for episode {episode_id}')
        return cache_path

    ep = ds[episode_id]

    logging.info(f'Generating new RRD file for episode {episode_id}')
    recording_id = f'positronic_ds_{Path(ds.root).name}_episode_{episode_id}'
    rec = rr.new_recording(application_id=recording_id)

    with rec:
        task = ep.static.get('task', None)
        if task:
            rr.log('/task', rr.TextDocument(task), static=True)

        video_names, signal_names, signal_dims = _collect_signal_groups(ep)
        rr.send_blueprint(_build_blueprint(task, video_names, signal_names, signal_dims))

        # Provide short per-channel names for legend/tooltips
        _setup_series_names(ep, signal_names)

        _log_episode(ep, video_names, signal_names)
        rr.save(cache_path)

    return cache_path
