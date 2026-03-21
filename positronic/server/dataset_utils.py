"""Dataset utilities for Positronic dataset visualization."""

import logging
import tempfile
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree

from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Kind
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.video import VideoSignal
from positronic.utils.rerun_compat import flatten_numeric, log_series_styles, set_timeline_time

# TODO: 3D visualization roles (pose_signals, joint_signal) are currently read from episode
# static data as flat keys. A cleaner long-term solution is signal-level metadata: each Signal
# would carry a `role` (e.g. 'transform3d', 'joint_position') and optionally a `robot` reference
# linking it to a robot model in static. This would:
# - Eliminate the need for pose_signals/joint_signal keys in static
# - Support multiple robots naturally (each signal references its own model)
# - Keep semantics with the signal that produces them, not in a parallel list
# - Require extending SignalMeta (currently dtype/shape/kind) with user-settable fields
#   and persisting them (parquet metadata or sidecar file)
# See: positronic/dataset/signal.py — SignalMeta, Kind

_POSE_COLORS = {
    'commands': [255, 100, 50],  # orange — commanded trajectory
    'state': [50, 200, 255],  # cyan — actual/state trajectory
    'default': [180, 180, 180],  # gray fallback
}


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


def _log_static_trail(entity_path: str, positions: np.ndarray, base_rgb: list[int]) -> None:
    """Log the full trajectory as a thin, muted static background."""
    if len(positions) < 2:
        return
    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    muted = [c // 3 + 40 for c in base_rgb]  # blend toward gray; rerun 3D doesn't do alpha
    colors = np.tile([*muted, 255], (len(segments), 1)).astype(np.uint8)
    rr.log(entity_path, rr.LineStrips3D(segments, colors=colors, radii=0.0005), static=True)


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
            mapping = {'__index__': episode_index, **ep}
            episode_data = [_format_value(mapping.get(key), formatters.get(key), defaults.get(key)) for key in keys]
            row = [episode_index, episode_data]

            # Include group metadata if available for using it in URL
            if ep.get('__meta__') and 'group' in ep['__meta__']:
                row.append(ep['__meta__']['group'])

            result.append(row)
        except Exception as e:
            raise Exception(f'Error getting episode {idx}: {ep.get("__meta__", {})}') from e
    return result


def _compute_eye_controls(signals: EpisodeSignals, ep: Episode) -> rrb.EyeControls3D | None:
    """Compute camera view orthogonal to the best-fit plane of all pose trajectories."""
    all_positions = [
        np.asarray(ep.signals[name].values(), dtype=np.float32)[:, :3] for name in signals.poses if ep.signals[name]
    ]
    if not all_positions:
        return None

    positions = np.concatenate(all_positions)
    if len(positions) < 3:
        return None
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]

    # Pick the normal direction that places the robot base (origin) behind the trajectory
    # i.e. camera on the opposite side from the base
    if np.dot(normal, centroid) < 0:
        normal = -normal

    spread = np.linalg.norm(centered, axis=1).max()
    camera_pos = centroid + normal * spread * 2.0
    return rrb.EyeControls3D(position=camera_pos.tolist(), look_target=centroid.tolist())


def _collect_signal_groups(ep: Episode) -> EpisodeSignals:
    pose_set = set(ep.static.get('pose_signals', []))
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
        if name in pose_set:
            signals.poses.append(name)
    return signals


def _group_signals_by_prefix(signals: EpisodeSignals) -> list[tuple[str, list[str]]]:
    """Group numeric signals by prefix before the first '.'. Preserves insertion order."""
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for sig in signals.numerics:
        groups[sig.split('.')[0] if '.' in sig else sig].append(sig)
    return list(groups.items())


def _build_blueprint(signals: EpisodeSignals, ep: Episode) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in signals.videos]

    def _ts_view(name: str, sig: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            name=name,
            origin=f'/signals/{sig}',
            plot_legend=rrb.PlotLegend(visible=signals.dims.get(sig, 1) > 1),
            axis_y=rrb.ScalarAxis(zoom_lock=True),
        )

    # Group time series by prefix, each group becomes a Tabs container
    series_views = []
    for group_name, sigs in _group_signals_by_prefix(signals):
        if len(sigs) == 1:
            view = _ts_view(group_name, sigs[0])
        else:
            view = rrb.Tabs(*[_ts_view(sig[len(group_name) + 1 :], sig) for sig in sigs], name=group_name)
        series_views.append(view)

    # Top row: images (big) + optional 3D (smaller)
    top_items = []
    if image_views:
        top_items.append(rrb.Grid(*image_views))
    if signals.poses:
        eye = _compute_eye_controls(signals, ep)
        top_items.append(
            rrb.Spatial3DView(
                name='3D Trajectory',
                origin='/3d',
                background=[30, 30, 30],
                line_grid=rrb.LineGrid3D(visible=True),
                eye_controls=eye or rrb.EyeControls3D(),
            )
        )

    rows = []
    row_shares = []
    if top_items:
        rows.append(top_items[0] if len(top_items) == 1 else rrb.Horizontal(*top_items, column_shares=[3, 1]))
        row_shares.append(3)
    if series_views:
        rows.append(rrb.Grid(*series_views))
        row_shares.append(1)

    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Collapsed),
        rrb.Vertical(*rows, row_shares=row_shares),
    )


def _setup_series_names(signals: EpisodeSignals, ep: Episode) -> None:
    joint_signal = ep.static.get('joint_signal')
    joint_names = ep.static.get('joint_names')
    pose_set = set(signals.poses)
    for key in signals.numerics:
        dim = signals.dims.get(key, 1)
        if key == joint_signal and joint_names:
            names = joint_names
        elif key in pose_set and dim == 7:
            names = ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        else:
            names = None
        if dim == 1:
            if names:
                log_series_styles(f'/signals/{key}', [names[0]], static=True)
        else:
            for i in range(dim):
                label = names[i] if names else str(i)
                log_series_styles(f'/signals/{key}/{i}', [label], static=True)


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


def _encode_frames_as_video(entity_path: str, sig) -> None:
    """Encode raw image frames into an H.265 video stream via pyav."""
    import av

    codec = rr.VideoCodec.H265
    container = av.open('/dev/null', 'w', format='hevc')

    first_frame = np.asarray(sig[0][0])
    h, w = first_frame.shape[:2]
    stream = container.add_stream('libx265', rate=30)
    assert isinstance(stream, av.video.stream.VideoStream)
    stream.width = w
    stream.height = h
    stream.max_b_frames = 0

    rr.log(entity_path, rr.VideoStream(codec=codec), static=True)

    for val, ts in sig:
        frame = av.VideoFrame.from_ndarray(np.asarray(val), format='rgb24')
        for packet in stream.encode(frame):
            if packet.pts is None:
                continue
            set_timeline_time('time', ts)
            rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))

    for packet in stream.encode():
        if packet.pts is not None:
            rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))


def _log_video_signals(ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer) -> Iterator[bytes]:
    """Log video signals as AssetVideo + VideoFrameReference (columnar), or as individual images."""
    for name in signals.videos:
        sig = ep.signals[name]
        if isinstance(sig, VideoSignal):
            video_bytes = sig.video_path.read_bytes()
            asset = rr.AssetVideo(contents=video_bytes, media_type='video/mp4')
            rr.log(name, asset, static=True)

            our_ts = np.asarray(sig.keys(), dtype='datetime64[ns]')
            frame_pts_ns = asset.read_frame_timestamps_nanos()
            rr.send_columns(
                name,
                indexes=[rr.TimeColumn('time', timestamp=our_ts)],
                columns=rr.VideoFrameReference.columns_nanos(frame_pts_ns),
            )
        else:
            _encode_frames_as_video(name, sig)
        yield from drainer.drain()


def _log_numeric_signals(
    ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer
) -> Generator[bytes, None, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Log numeric time-series via send_columns. Returns pose/joint data for 3D logging."""
    pose_set = set(signals.poses)
    joint_signal = ep.static.get('joint_signal')
    stash_keys = pose_set | ({joint_signal} if joint_signal else set())
    pose_data = {}

    for key in signals.numerics:
        sig = ep.signals[key]
        if len(sig) == 0:
            continue
        ts_arr = np.asarray(sig.keys(), dtype='datetime64[ns]')
        try:
            vals = np.asarray(sig.values(), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        dim = vals.shape[1]

        time_idx = [rr.TimeColumn('time', timestamp=ts_arr)]
        if dim == 1:
            rr.send_columns(f'/signals/{key}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals.ravel()))
        else:
            for i in range(dim):
                rr.send_columns(f'/signals/{key}/{i}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals[:, i]))

        if key in stash_keys:
            pose_data[key] = (ts_arr, vals)

        yield from drainer.drain()

    return pose_data


def _write_urdf_to_dir(urdf_str: str, meshes: dict[str, bytes], dest: Path) -> Path:
    """Write URDF and mesh files to a directory, rewriting mesh filenames to absolute paths."""
    root = ET.fromstring(urdf_str)
    for mesh_el in root.iter('mesh'):
        filename = mesh_el.get('filename', '')
        if filename in meshes:
            mesh_el.set('filename', str(dest / filename))
    urdf_path = dest / 'robot.urdf'
    urdf_path.write_text(ET.tostring(root, encoding='unicode'))
    for name, data in meshes.items():
        safe = Path(name).name  # strip any path components
        (dest / safe).write_bytes(data)
    return urdf_path


def _animate_joint(joint, q_column: np.ndarray, ts_arr: np.ndarray, prefix: str) -> None:
    """Compute and log transforms for a single URDF joint across all timesteps."""
    n = len(ts_arr)
    translations = np.empty((n, 3), dtype=np.float64)
    quaternions = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        t = joint.compute_transform(float(q_column[i]))
        translations[i] = t.translation.as_arrow_array().to_pylist()[0]
        quaternions[i] = t.quaternion.as_arrow_array().to_pylist()[0]
    rr.send_columns(
        f'{prefix}/{joint.child_link}',
        indexes=[rr.TimeColumn('time', timestamp=ts_arr)],
        columns=rr.Transform3D.columns(
            translation=translations,
            quaternion=quaternions,
            child_frame=[joint.child_link] * n,
            parent_frame=[joint.parent_link] * n,
        ),
    )


_URDF_ANIM_HZ = 15


def _log_urdf_robot(
    ep: Episode, numeric_data: dict[str, tuple[np.ndarray, np.ndarray]], drainer: _BinaryStreamDrainer
) -> Generator[bytes, None, str | None]:
    """Log URDF robot model with animated joint angles. Returns root frame name."""
    joint_signal = ep.static.get('joint_signal')
    joint_names = ep.static.get('joint_names')
    urdf_str = ep.static.get('urdf')
    meshes = ep.static.get('meshes')
    if not all((joint_signal, joint_names, urdf_str, meshes)) or joint_signal not in numeric_data:
        return None
    ts_arr, q_vals = numeric_data[joint_signal]
    if q_vals.shape[1] != len(joint_names):
        return None

    prefix = '/3d/robot'
    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = _write_urdf_to_dir(urdf_str, meshes, Path(tmp))
        rr.log_file_from_path(str(urdf_path), entity_path_prefix=prefix, static=True)
        tree = UrdfTree.from_file_path(str(urdf_path), entity_path_prefix=prefix)

    root_frame = tree.root_link().name
    yield from drainer.drain()

    # Downsample to ~15Hz — robot motion is smooth enough, avoids bloating the RRD
    duration_ns = int(ts_arr[-1]) - int(ts_arr[0])
    target_samples = max(1, int(_URDF_ANIM_HZ * duration_ns / 1e9))
    step = max(1, len(ts_arr) // target_samples)
    ts_ds, q_ds = ts_arr[::step], q_vals[::step]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for j_idx, name in enumerate(joint_names):
            joint = tree.get_joint_by_name(name)
            if joint is not None:
                _animate_joint(joint, q_ds[:, j_idx], ts_ds, prefix)
                yield from drainer.drain()

    return root_frame


def _log_pose_signals(
    ep: Episode,
    signals: EpisodeSignals,
    numeric_data: dict[str, tuple[np.ndarray, np.ndarray]],
    drainer: _BinaryStreamDrainer,
) -> Iterator[bytes]:
    """Log 3D pose: static full trajectory + current position ball + optional URDF robot."""
    root_frame = yield from _log_urdf_robot(ep, numeric_data, drainer)

    for key in signals.poses:
        if key not in numeric_data:
            continue
        ts_arr, vals = numeric_data[key]
        if vals.ndim < 2 or vals.shape[1] != 7:
            continue
        positions = vals[:, :3]
        color = _pose_color(key)

        # Connect pose entities to the URDF root frame so they share the same 3D space
        if root_frame:
            rr.log(f'/3d/{key}', rr.Transform3D(parent_frame=root_frame), static=True)
            rr.log(f'/3d/{key}/trail', rr.Transform3D(parent_frame=root_frame), static=True)

        _log_static_trail(f'/3d/{key}/trail', positions, color)

        rr.send_columns(
            f'/3d/{key}',
            indexes=[rr.TimeColumn('time', timestamp=ts_arr)],
            columns=[
                *rr.Points3D.columns(positions=positions).partition([1] * len(ts_arr)),
                *rr.Points3D.columns(colors=np.tile(color, (len(ts_arr), 1))).partition([1] * len(ts_arr)),
                *rr.Points3D.columns(radii=np.full(len(ts_arr), 0.01)),
            ],
        )
        yield from drainer.drain()


@rr.recording_stream.recording_stream_generator_ctx
def stream_episode_rrd(ds: Dataset, episode_id: int) -> Iterator[bytes]:
    """Yield an episode RRD as chunks while it is being generated."""

    ep = ds[episode_id]
    logging.info(f'Streaming RRD for episode {episode_id}')

    dataset_root = get_dataset_root(ds)
    dataset_name = Path(dataset_root).name if dataset_root else 'unknown'
    recording_id = f'positronic_ds_{dataset_name}_episode_{episode_id}'
    rec = rr.RecordingStream(application_id=recording_id)
    drainer = _BinaryStreamDrainer(rec.binary_stream(), min_bytes=2**20)

    with rec:
        signals = _collect_signal_groups(ep)
        rr.send_blueprint(_build_blueprint(signals, ep))
        yield from drainer.drain()

        _setup_series_names(signals, ep)
        yield from drainer.drain()

        yield from _log_video_signals(ep, signals, drainer)
        pose_data = yield from _log_numeric_signals(ep, signals, drainer)
        yield from drainer.drain(force=True)  # flush numerics to client before slow pose trails
        yield from _log_pose_signals(ep, signals, pose_data, drainer)

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
