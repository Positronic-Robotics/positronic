"""Dataset utilities for Positronic dataset visualization."""

import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Kind
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.video import VideoSignal
from positronic.utils.rerun_compat import flatten_numeric, log_series_styles, set_timeline_time

_POSE_SUFFIXES = ('.pose', '.ee_pose')
_POSE_COLORS = {
    'commands': [255, 100, 50],  # orange — commanded trajectory
    'state': [50, 200, 255],  # cyan — actual/state trajectory
    'default': [180, 180, 180],  # gray fallback
}


def _is_pose_signal(name: str, dim: int) -> bool:
    """Return True if the signal looks like a 7D ee pose (tx, ty, tz, qx, qy, qz, qw)."""
    return dim == 7 and any(name.endswith(s) for s in _POSE_SUFFIXES)


def _pose_color(name: str) -> list[int]:
    prefix = name.split('.')[0] if '.' in name else name
    for suffix, color in _POSE_COLORS.items():
        if prefix.endswith(suffix):
            return color
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
_TRAIL_UPDATE_INTERVAL = 30  # Only re-log full trail every N pose samples


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
            yield bytes(self._buffer[: self._min_bytes])
            del self._buffer[: self._min_bytes]
        # On force, yield any remaining bytes
        if force and self._buffer:
            yield bytes(self._buffer)
            self._buffer.clear()


def _log_video_signals(ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer) -> Iterator[bytes]:
    """Log video signals as AssetVideo + VideoFrameReference (columnar)."""
    for name in signals.videos:
        sig = ep.signals[name]
        if not isinstance(sig, VideoSignal):
            continue
        video_bytes = sig.video_path.read_bytes()
        asset = rr.AssetVideo(contents=video_bytes, media_type='video/mp4')
        rr.log(name, asset, static=True)

        our_ts = np.asarray(sig.keys())
        frame_pts_ns = asset.read_frame_timestamps_ns()
        rr.send_columns(
            name,
            indexes=[rr.TimeNanosColumn('time', our_ts)],
            columns=rr.VideoFrameReference.columns_nanoseconds(frame_pts_ns),
        )
        yield from drainer.drain()


def _log_numeric_signals(
    ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer
) -> Generator[bytes, None, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Log numeric time-series via send_columns. Returns pose data for 3D logging."""
    pose_set = set(signals.poses)
    pose_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for key in signals.numerics:
        timestamps = []
        values = []
        for payload, ts_ns in ep.signals[key]:
            arr = flatten_numeric(payload)
            if arr is not None:
                timestamps.append(ts_ns)
                values.append(arr)

        if not timestamps:
            continue

        ts_arr = np.array(timestamps, dtype=np.int64)
        vals = np.stack(values)
        dim = vals.shape[1] if vals.ndim > 1 else 1

        if dim == 1:
            rr.send_columns(
                f'/signals/{key}',
                indexes=[rr.TimeNanosColumn('time', ts_arr)],
                columns=rr.Scalar.columns(scalar=vals.ravel()),
            )
        else:
            for i in range(dim):
                rr.send_columns(
                    f'/signals/{key}/{i}',
                    indexes=[rr.TimeNanosColumn('time', ts_arr)],
                    columns=rr.Scalar.columns(scalar=vals[:, i]),
                )

        if key in pose_set:
            pose_data[key] = (ts_arr, vals)

        yield from drainer.drain()

    return pose_data


def _log_pose_signals(
    signals: EpisodeSignals, numeric_data: dict[str, tuple[np.ndarray, np.ndarray]], drainer: _BinaryStreamDrainer
) -> Iterator[bytes]:
    """Log 3D pose points and trajectory trails."""
    for key in signals.poses:
        if key not in numeric_data:
            continue
        ts_arr, vals = numeric_data[key]
        if vals.ndim < 2 or vals.shape[1] != 7:
            continue
        positions = vals[:, :3]
        color = _pose_color(key)
        rr.send_columns(
            f'/3d/{key}',
            indexes=[rr.TimeNanosColumn('time', ts_arr)],
            columns=rr.Points3D.columns(
                positions=positions, colors=np.tile(color, (len(ts_arr), 1)), radii=np.full(len(ts_arr), 0.01)
            ),
        )
        # Log trajectory trail at periodic intervals so it's visible when scrubbing
        pos_list = positions.tolist()
        timestamps = ts_arr.tolist()
        trail_path = f'/3d/{key}/trail'
        for i in range(_TRAIL_UPDATE_INTERVAL, len(pos_list), _TRAIL_UPDATE_INTERVAL):
            set_timeline_time('time', timestamps[i])
            _log_trajectory_trail(trail_path, pos_list[: i + 1], timestamps[: i + 1], color)
        # Final trail at last timestamp
        set_timeline_time('time', timestamps[-1])
        _log_trajectory_trail(trail_path, pos_list, timestamps, color)
        yield from drainer.drain()


@rr.recording_stream.recording_stream_generator_ctx
def stream_episode_rrd(ds: Dataset, episode_id: int) -> Iterator[bytes]:
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

        yield from _log_video_signals(ep, signals, drainer)
        pose_data = yield from _log_numeric_signals(ep, signals, drainer)
        yield from _log_pose_signals(signals, pose_data, drainer)

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
