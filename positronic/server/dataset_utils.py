"""Dataset utilities for Positronic dataset visualization."""

import heapq
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Kind
from positronic.dataset.transforms import TransformedDataset
from positronic.utils.rerun_compat import flatten_numeric, log_numeric_series, log_series_styles, set_timeline_time

_POSE_SUFFIXES = ('.pose', '.ee_pose')
_POSE_COLORS = {
    'command': [255, 100, 50],  # orange — commanded trajectory
    'state': [50, 200, 255],  # cyan — actual/state trajectory
    'default': [180, 180, 180],  # gray fallback
}


def _is_pose_signal(name: str, dim: int) -> bool:
    """Return True if the signal looks like a 7D ee pose (tx, ty, tz, qx, qy, qz, qw)."""
    return dim == 7 and any(name.endswith(s) for s in _POSE_SUFFIXES)


def _pose_color(name: str) -> list[int]:
    prefix = name.split('.')[0] if '.' in name else name
    if prefix.endswith('commands'):
        return _POSE_COLORS['command']
    if prefix.endswith('state'):
        return _POSE_COLORS['state']
    return _POSE_COLORS['default']


@dataclass
class EpisodeSignals:
    videos: list[str]
    numerics: list[str]
    dims: dict[str, int]
    poses: list[str]


def _infer_dims(sig) -> int:
    if len(sig) == 0:
        return 1
    val, _ = sig[0]
    arr = flatten_numeric(val)
    return int(arr.size) if arr is not None else 1


_TRAIL_FADE_NS = 5_000_000_000  # 5-second window of full visibility


def _log_trajectory_trail(
    entity_path: str, positions: list[list[float]], timestamps_ns: list[int], base_rgb: list[int]
) -> None:
    """Log trajectory as per-segment line strips with time-based fade.

    Segments within the last 5 s are bright (alpha scales up to 255).
    Older segments drop to near-transparent (alpha ~15).
    """
    if len(positions) < 2:
        return
    now = timestamps_ns[-1]
    segments = []
    colors = []
    for a, b, ts in zip(positions, positions[1:], timestamps_ns[1:], strict=False):
        segments.append([a, b])
        age = now - ts
        alpha = 15 if age >= _TRAIL_FADE_NS else int(15 + 240 * (1.0 - age / _TRAIL_FADE_NS))
        colors.append([*base_rgb, alpha])
    rr.log(entity_path, rr.LineStrips3D(segments, colors=colors))


def _format_value(value: Any, formatter: str | None, default: Any) -> Any:
    """Formats a single value based on its type and provided formatters/defaults."""
    if isinstance(value, datetime):
        formatted_date = value.strftime(formatter) if formatter else value.isoformat()
        return [value.timestamp(), formatted_date]
    elif value is not None and formatter:
        return [value, formatter % value]
    elif value is not None:
        return value
    else:
        return default


def get_episodes_list(
    ds: Iterator[dict[str, Any]], keys: list[str], formatters: dict[str, str | None], defaults: dict[str, Any]
) -> list[list[Any]]:
    result = []
    for idx, ep in enumerate(ds):
        try:
            episode_index = ep.pop('__episode_index__', idx)
            mapping = {'__index__': idx + 1, **ep}
            episode_data = [_format_value(mapping.get(key), formatters.get(key), defaults.get(key)) for key in keys]
            row = [episode_index, episode_data]

            # Include group metadata if available for using it in URL
            if ep.get('__meta__') and 'group' in ep['__meta__']:
                row.append(ep['__meta__']['group'])

            result.append(row)
        except Exception as e:
            raise Exception(f'Error getting episode {idx}: {ep.get("__meta__", {})}') from e
    return result


def _collect_signal_groups(ep: Episode) -> EpisodeSignals:
    signals = EpisodeSignals(videos=[], numerics=[], dims={}, poses=[])
    for name, sig in ep.signals.items():
        if sig.kind == Kind.IMAGE:
            try:
                sig[0]
                signals.videos.append(name)
            except Exception:
                pass
            continue

        signals.numerics.append(name)
        try:
            signals.dims[name] = _infer_dims(sig)
        except Exception:
            signals.dims[name] = 1
        if _is_pose_signal(name, signals.dims[name]):
            signals.poses.append(name)
    return signals


def _build_blueprint(signals: EpisodeSignals) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in signals.videos]
    per_signal_views = [
        rrb.TimeSeriesView(
            name=sig, origin=f'/signals/{sig}', plot_legend=rrb.PlotLegend(visible=signals.dims.get(sig, 1) > 1)
        )
        for sig in signals.numerics
    ]

    grid_items = []
    column_shares = []
    if per_signal_views:
        grid_items.append(rrb.Grid(*per_signal_views))
        column_shares.append(1)
    if signals.poses:
        grid_items.append(rrb.Spatial3DView(name='3D Trajectory', origin='/3d'))
        column_shares.append(1)
    if image_views:
        grid_items.append(rrb.Grid(*image_views))
        column_shares.append(2)

    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Collapsed),
        rrb.Grid(*grid_items, column_shares=column_shares),
    )


def _setup_series_names(signals: EpisodeSignals) -> None:
    for key in signals.numerics:
        names = [str(i) for i in range(max(1, signals.dims.get(key, 1)))]
        log_series_styles(f'/signals/{key}', names, static=True)


def _episode_log_entries(ep: Episode, signals: EpisodeSignals):
    heap: list[tuple[int, int, str, str, Any, Iterator[tuple[Any, int]]]] = []

    def _push(sig_index: int, kind: str, key: str, iterator: Iterator[tuple[Any, int]]):
        try:
            payload, ts_ns = next(iterator)
        except StopIteration:
            return
        heapq.heappush(heap, (ts_ns, sig_index, kind, key, payload, iterator))

    iterators: list[tuple[int, str, str, Iterator[tuple[Any, int]]]] = []
    for idx, key in enumerate(signals.videos):
        iterators.append((idx, 'video', key, iter(ep.signals[key])))
    base_idx = len(iterators)
    for offset, key in enumerate(signals.numerics):
        iterators.append((base_idx + offset, 'numeric', key, iter(ep.signals[key])))

    for sig_index, kind, key, iterator in iterators:
        _push(sig_index, kind, key, iterator)

    while heap:
        ts_ns, sig_index, kind, key, payload, iterator = heapq.heappop(heap)
        yield (kind, key, payload, ts_ns)
        _push(sig_index, kind, key, iterator)


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
def stream_episode_rrd(ds: Dataset, episode_id: int, max_resolution: int) -> Iterator[bytes]:
    """Yield an episode RRD as chunks while it is being generated."""

    ep = ds[episode_id]
    logging.info(f'Streaming RRD for episode {episode_id}')

    dataset_root = get_dataset_root(ds)
    dataset_name = Path(dataset_root).name if dataset_root else 'unknown'
    recording_id = f'positronic_ds_{dataset_name}_episode_{episode_id}'
    rec = rr.new_recording(application_id=recording_id)
    drainer = _BinaryStreamDrainer(rec.binary_stream(), min_bytes=2**20)

    with rec:
        signals = _collect_signal_groups(ep)
        rr.send_blueprint(_build_blueprint(signals))
        yield from drainer.drain()

        _setup_series_names(signals)
        yield from drainer.drain()

        pose_set = set(signals.poses)
        pose_positions: dict[str, list[list[float]]] = {name: [] for name in signals.poses}
        pose_timestamps: dict[str, list[int]] = {name: [] for name in signals.poses}
        pose_colors = {name: _pose_color(name) for name in signals.poses}

        for kind, key, payload, ts_ns in _episode_log_entries(ep, signals):
            set_timeline_time('time', ts_ns)
            if kind == 'numeric':
                log_numeric_series(f'/signals/{key}', payload)
                if key in pose_set:
                    arr = flatten_numeric(payload)
                    if arr is not None and arr.size == 7:
                        pos = arr[:3].tolist()
                        rr.log(f'/3d/{key}', rr.Points3D([pos], colors=[pose_colors[key]], radii=[0.01]))
                        pose_positions[key].append(pos)
                        pose_timestamps[key].append(ts_ns)
                        _log_trajectory_trail(
                            f'/3d/{key}/trail', pose_positions[key], pose_timestamps[key], pose_colors[key]
                        )
            else:
                rr.log(key, rr.Image(resize_if_needed(payload, max_resolution)).compress())
            yield from drainer.drain()

    yield from drainer.drain(force=True)


def get_dataset_root(dataset: Dataset) -> str | None:
    """Try to extract root path from Dataset type."""

    if 'name' in dataset.meta:
        return dataset.meta['name']

    if isinstance(dataset, LocalDataset):
        return str(dataset.root)

    # If it's a TransformedDataset, unwrap to get the underlying LocalDataset
    if isinstance(dataset, TransformedDataset):
        return get_dataset_root(dataset._dataset)

    return None


def resize_if_needed(image, max_resolution: int):
    height, width = image.shape[:2]
    scale = min(1, max_resolution / max(width, height))
    max_width, max_height = int(width * scale), int(height * scale)

    # Downscale if needed
    if width != max_width or height != max_height:
        return cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)

    return image
