"""Dataset utilities for Positronic dataset visualization."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

import av
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Signal
from positronic.dataset.video import VideoSignal


class GenerationMode(Enum):
    IMAGES = 'images'
    VIDEO = 'video'


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


def _estimate_fps_from_video_signal(ep: Episode, name: str) -> Optional[int]:
    """Best-effort FPS estimation from timestamps of a video signal."""
    try:
        ts = np.asarray(ep[name]._ts_at(slice(None)))  # type: ignore[attr-defined]
        if ts.size >= 2:
            diffs = np.diff(ts[:min(ts.size, 100)])
            med = float(np.median(diffs))
            if med > 0:
                return round(1e9 / med)
    except Exception:
        pass
    return None


def _infer_features_and_fps(ep: Episode) -> Tuple[dict[str, dict[str, Any]], Optional[int]]:
    """Infer feature metadata from an episode and return (features, fps)."""
    features: dict[str, dict[str, Any]] = {}
    fps: Optional[int] = None

    for name in ep.signals.keys():
        info = _infer_signal_info(ep, name)
        if info.kind == 'video' and info.shape is not None:
            features[name] = {'dtype': 'image', 'shape': list(info.shape)}
            if fps is None:
                fps = _estimate_fps_from_video_signal(ep, name)
        elif info.kind in ('vector', 'tensor', 'scalar'):
            shape = [] if info.shape is None else list(info.shape)
            features[name] = {'dtype': 'float32' if info.dtype is None else info.dtype, 'shape': shape}

    return features, fps


def _estimate_total_length(ds: LocalDataset) -> int:
    """Approximate total samples as sum of max dynamic lengths per episode."""
    total_len = 0
    for i in range(len(ds)):
        ep = ds[i]
        max_len = 0
        for sig in ep.signals.values():
            try:
                max_len = max(max_len, len(sig))
            except Exception:
                continue
        total_len += max_len
    return total_len


def get_dataset_info(ds: LocalDataset) -> dict[str, Any]:
    """Return basic info + feature descriptions for the dataset."""
    num_eps = len(ds)
    features: dict[str, dict[str, Any]] = {}
    first_video_fps: Optional[int] = None

    if num_eps > 0:
        ep0 = ds[0]
        features, first_video_fps = _infer_features_and_fps(ep0)

    return {
        'root': str(ds.root),
        'num_episodes': num_eps,
        'num_samples': _estimate_total_length(ds),
        'fps': first_video_fps or 0,
        'features': features,
    }


def get_task_description(ep: Episode) -> str | None:
    """Extract a task description if provided as a static key 'task'."""
    try:
        v = ep['task']
        if isinstance(v, str):
            return v
    except KeyError:
        pass
    return None


def get_episodes_list(ds: LocalDataset) -> list[dict[str, Any]]:
    episodes = []
    for idx in range(len(ds)):
        ep = ds[idx]
        # Estimate length
        max_len = 0
        for sig in ep.signals.values():
            try:
                max_len = max(max_len, len(sig))
            except Exception:
                continue
        episodes.append({
            'index': idx,
            'length': max_len,
            'task': get_task_description(ep),
        })
    return episodes


def _init_stream(image_key: str, fps: int, width: int, height: int):
    container = av.open('/dev/null', 'w', format='h264')
    stream = container.add_stream('libx264', rate=max(1, int(fps) or 30))
    stream.width = width
    stream.height = height
    stream.max_b_frames = 0
    rr.log(image_key, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
    return stream


def _log_image(image: np.ndarray, key: str, streams: dict[str, av.VideoStream],
               generation_mode: GenerationMode) -> None:
    if generation_mode == GenerationMode.IMAGES:
        rr.log(key, rr.Image(image).compress())
    else:
        video_frame = av.VideoFrame.from_ndarray(image, format='rgb24')
        for packet in streams[key].encode(video_frame):
            if packet.pts is None:
                continue
            rr.log(key, rr.VideoStream.from_fields(sample=bytes(packet)))


def _iter_reference_timeline(ep: Episode) -> Iterator[tuple[int, int]]:
    """Yield (frame_idx, ts_ns) along a reference signal timeline.

    Preference order: first video signal; otherwise longest dynamic signal.
    """
    ref_name = None
    # Prefer first video signal
    for name, sig in ep.signals.items():
        if isinstance(sig, VideoSignal):
            ref_name = name
            break
    # Fallback to longest signal
    if ref_name is None:
        best_len = -1
        for name, sig in ep.signals.items():
            try:
                sig_len = len(sig)
            except Exception:
                continue
            if sig_len > best_len:
                best_len = sig_len
                ref_name = name
    if ref_name is None:
        return

    ref_sig = ep.signals[ref_name]
    for i in range(len(ref_sig)):
        _v, ts = ref_sig[i]
        yield i, int(ts)


def _collect_signal_groups(ep: Episode) -> Tuple[list[str], list[str], dict[str, tuple[int, int, int]]]:
    """Return (video_names, vector_names, shapes) for an episode."""
    video_names: list[str] = []
    vector_names: list[str] = []
    shapes: dict[str, tuple[int, int, int]] = {}
    for name, sig in ep.signals.items():
        if isinstance(sig, VideoSignal):
            try:
                frame, _ = sig[0]
                h, w = frame.shape[:2]
                shapes[name] = (h, w, 3)
                video_names.append(name)
            except Exception:
                continue
        else:
            vector_names.append(name)
    return video_names, vector_names, shapes


def _build_blueprint(task: Optional[str], video_names: list[str], vector_names: list[str]) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k.replace('_', ' ').title(), origin=f'/{k}')
                   for k in video_names] or [rrb.TextDocumentView(name='No Images', contents=['No image data found'])]
    plot_legend = rrb.PlotLegend(corner=rrb.Corner2D.RightBottom)
    vertical_views: list[rrb.View] = []
    if task:
        vertical_views.append(rrb.TextDocumentView(name='Task', origin='/task'))
    if vector_names:
        vertical_views.append(rrb.TimeSeriesView(name='Signals', origin='/signals', plot_legend=plot_legend))
    else:
        vertical_views.append(rrb.TextDocumentView(name='No Signals', contents=['No vector data found']))
    row_shares = [0.2, 1] if task else [1]
    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Expanded),
        rrb.Grid(
            rrb.Vertical(*vertical_views, row_shares=row_shares),
            rrb.Grid(*image_views),
            column_shares=[1, 2],
        ),
    )


def _init_streams(shapes: dict[str, tuple[int, int, int]], fps: int) -> dict[str, av.VideoStream]:
    streams: dict[str, av.VideoStream] = {}
    for key, (h, w, _c) in shapes.items():
        streams[key] = _init_stream(key, fps=fps, width=w, height=h)
    return streams


def _set_time(frame_idx: int, ts_ns: int) -> None:
    rr.set_time_seconds('time', ts_ns / 1e9)
    rr.set_time_sequence('frame_index', frame_idx)


def _log_vectors_for_snapshot(vector_names: list[str], snap: dict[str, Any]) -> None:
    for key in vector_names:
        try:
            val = snap[key]
        except Exception:
            continue
        _log_single_vector(key, val)


def _log_single_vector(key: str, val: Any) -> None:
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            rr.log(f'signals/{key}', rr.Scalar(float(val)))
        elif val.ndim == 1:
            for i, v in enumerate(val.tolist()):
                rr.log(f'signals/{key}/{i}', rr.Scalar(float(v)))
        else:
            flat = val.reshape(-1)
            for i, v in enumerate(flat.tolist()):
                rr.log(f'signals/{key}/{i}', rr.Scalar(float(v)))
    else:
        try:
            rr.log(f'signals/{key}', rr.Scalar(float(val)))
        except Exception:
            pass


def _log_videos_for_snapshot(video_names: list[str], snap: dict[str, Any], streams: dict[str, av.VideoStream],
                             generation_mode: GenerationMode) -> None:
    for key in video_names:
        try:
            img = snap[key]
            if isinstance(img, np.ndarray) and img.ndim == 3:
                _log_image(img, key, streams, generation_mode)
        except Exception:
            continue


def _flush_streams(streams: dict[str, av.VideoStream]) -> None:
    for key, stream in streams.items():
        for packet in stream.encode():
            if packet.pts is None:
                continue
            rr.set_time('time', duration=float(packet.pts * packet.time_base))
            rr.log(key, rr.VideoStream.from_fields(sample=bytes(packet)))


def _log_episode(
    ep: Episode,
    video_names: list[str],
    vector_names: list[str],
    shapes: dict[str, tuple[int, int, int]],
    generation_mode: GenerationMode,
    fps: int,
) -> None:
    streams: dict[str, av.VideoStream] = _init_streams(shapes, fps) if generation_mode == GenerationMode.VIDEO else {}

    for frame_idx, ts_ns in _iter_reference_timeline(ep) or []:
        _set_time(frame_idx, ts_ns)
        snap = ep.time[ts_ns]
        _log_videos_for_snapshot(video_names, snap, streams, generation_mode)
        _log_vectors_for_snapshot(vector_names, snap)

    if generation_mode == GenerationMode.VIDEO:
        _flush_streams(streams)


def generate_episode_rrd(
    ds: LocalDataset,
    episode_id: int,
    cache_path: str,
    generation_mode: GenerationMode,
) -> str:
    """Generate an RRD for an episode and return its path (cached)."""
    if os.path.exists(cache_path):
        logging.info(f'Using cached RRD file for episode {episode_id}')
        return cache_path

    ep = ds[episode_id]

    logging.info(f'Generating new RRD file for episode {episode_id}')
    recording_id = f'positronic_ds_{Path(ds.root).name}_episode_{episode_id}'
    rec = rr.new_recording(application_id=recording_id)

    with rec:
        task = get_task_description(ep)
        if task:
            rr.log('/task', rr.TextDocument(task), static=True)

        video_names, vector_names, shapes = _collect_signal_groups(ep)

        # Resolve fps from dataset info (fallback 30)
        fps = 30
        info = get_dataset_info(ds)
        if isinstance(info.get('fps'), (int, float)) and info['fps']:
            fps = int(info['fps'])

        rr.send_blueprint(_build_blueprint(task, video_names, vector_names))

        _log_episode(ep, video_names, vector_names, shapes, generation_mode, fps)

        rr.save(cache_path)

    return cache_path
