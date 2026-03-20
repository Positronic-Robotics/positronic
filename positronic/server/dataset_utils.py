"""Dataset utilities for Positronic dataset visualization."""

import logging
import tempfile
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
    all_positions = []
    for name in signals.poses:
        sig = ep.signals[name]
        if len(sig) == 0:
            continue
        vals = np.asarray(sig.values(), dtype=np.float64)
        if vals.ndim == 2 and vals.shape[1] >= 3:
            all_positions.append(vals[:, :3])
    if not all_positions:
        return None

    positions = np.concatenate(all_positions)
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]  # smallest singular value = plane normal

    # Place camera along the normal, at a distance proportional to the trajectory spread
    spread = np.linalg.norm(centered, axis=1).max()
    camera_pos = centroid + normal * spread * 2.0

    return rrb.EyeControls3D(position=camera_pos.tolist(), look_target=centroid.tolist())


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


def _group_signals_by_prefix(signals: EpisodeSignals) -> list[tuple[str, list[str]]]:
    """Group numeric signals by prefix before the first '.'. Preserves insertion order."""
    groups: dict[str, list[str]] = {}
    for sig in signals.numerics:
        prefix = sig.split('.')[0] if '.' in sig else sig
        groups.setdefault(prefix, []).append(sig)
    return list(groups.items())


def _build_blueprint(signals: EpisodeSignals, eye_controls: rrb.EyeControls3D | None = None) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in signals.videos]

    # Show the full episode time range, unlinked from the global time cursor
    _inf = rr.datatypes.TimeRangeBoundary(inner=None, kind='infinite')
    full_range = rrb.VisibleTimeRanges(timeline='time', start=_inf, end=_inf)
    full_axis_x = rrb.TimeAxis(link=rrb.components.LinkAxis.Independent)

    # Group time series by prefix, each group becomes a Tabs container
    series_views = []
    for group_name, sigs in _group_signals_by_prefix(signals):
        if len(sigs) == 1:
            sig = sigs[0]
            series_views.append(
                rrb.TimeSeriesView(
                    name=group_name,
                    origin=f'/signals/{sig}',
                    plot_legend=rrb.PlotLegend(visible=signals.dims.get(sig, 1) > 1),
                    axis_x=full_axis_x,
                    axis_y=rrb.ScalarAxis(zoom_lock=True),
                    time_ranges=full_range,
                )
            )
        else:
            tab_views = [
                rrb.TimeSeriesView(
                    name=sig[len(group_name) + 1 :],
                    origin=f'/signals/{sig}',
                    plot_legend=rrb.PlotLegend(visible=signals.dims.get(sig, 1) > 1),
                    axis_x=full_axis_x,
                    axis_y=rrb.ScalarAxis(zoom_lock=True),
                    time_ranges=full_range,
                )
                for sig in sigs
            ]
            series_views.append(rrb.Tabs(*tab_views, name=group_name))

    # Top row: images (big) + optional 3D (smaller)
    top_items = []
    if image_views:
        top_items.append(rrb.Grid(*image_views))
    if signals.poses:
        top_items.append(
            rrb.Spatial3DView(
                name='3D Trajectory',
                origin='/3d',
                background=[30, 30, 30],
                line_grid=rrb.LineGrid3D(visible=True),
                eye_controls=eye_controls or rrb.EyeControls3D(),
            )
        )

    rows = []
    row_shares = []
    if top_items:
        if len(top_items) == 1:
            rows.append(top_items[0])
        else:
            rows.append(rrb.Horizontal(*top_items, column_shares=[3, 1]))
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
    has_joints = 'joint_names' in ep.static
    stash_keys = pose_set | ({'robot_state.q'} if has_joints else set())
    pose_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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

        if dim == 1:
            rr.send_columns(
                f'/signals/{key}',
                indexes=[rr.TimeColumn('time', timestamp=ts_arr)],
                columns=rr.Scalars.columns(scalars=vals.ravel()),
            )
        else:
            for i in range(dim):
                rr.send_columns(
                    f'/signals/{key}/{i}',
                    indexes=[rr.TimeColumn('time', timestamp=ts_arr)],
                    columns=rr.Scalars.columns(scalars=vals[:, i]),
                )

        if key in stash_keys:
            pose_data[key] = (ts_arr, vals)

        yield from drainer.drain()

    return pose_data


def _find_visual_urdf() -> Path | None:
    """Find Panda URDF with visual meshes from robosuite package data."""
    try:
        import robosuite
    except ImportError:
        return None
    base = Path(robosuite.__file__).parent / 'models/assets/bullet_data/panda_description'
    urdf_path = base / 'urdf/panda_arm.urdf'
    if not urdf_path.exists():
        return None
    return urdf_path


def _prepare_visual_urdf(urdf_path: Path) -> str:
    """Rewrite package:// mesh URIs to absolute paths."""
    import xml.etree.ElementTree as ET

    mesh_base = urdf_path.parent.parent
    tree = ET.parse(urdf_path)
    for mesh in tree.iter('mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith('package://panda_description/'):
            mesh.set('filename', str(mesh_base / filename.removeprefix('package://panda_description/')))
    return ET.tostring(tree.getroot(), encoding='unicode')


def _log_urdf_robot(
    ep: Episode, numeric_data: dict[str, tuple[np.ndarray, np.ndarray]], drainer: _BinaryStreamDrainer
) -> Generator[bytes, None, str | None]:
    """Log URDF robot model with animated joint angles. Returns root frame name."""
    from rerun.urdf import UrdfTree

    joint_names: list[str] | None = ep.static.get('joint_names')
    if not joint_names:
        return None

    q_key = 'robot_state.q'
    if q_key not in numeric_data:
        return None
    ts_arr, q_vals = numeric_data[q_key]
    if q_vals.shape[1] != len(joint_names):
        return None

    visual_urdf_path = _find_visual_urdf()
    if visual_urdf_path is None:
        return None

    urdf_str = _prepare_visual_urdf(visual_urdf_path)

    prefix = '/3d/robot'
    with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=True) as f:
        f.write(urdf_str)
        f.flush()
        rr.log_file_from_path(f.name, entity_path_prefix=prefix, static=True)
        tree = UrdfTree.from_file_path(f.name, entity_path_prefix=prefix)

    root_frame = tree.root_link().name
    yield from drainer.drain()

    # Map episode joint names (e.g. "joint1") to URDF joint names (e.g. "panda_joint1")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for j_idx, ep_joint_name in enumerate(joint_names):
            # Try exact name first, then panda_ prefix
            joint = tree.get_joint_by_name(ep_joint_name) or tree.get_joint_by_name(f'panda_{ep_joint_name}')
            if joint is None:
                continue
            entity_path = f'{prefix}/{joint.child_link}'
            n = len(ts_arr)
            translations = np.empty((n, 3), dtype=np.float64)
            quaternions = np.empty((n, 4), dtype=np.float64)
            for i in range(n):
                t = joint.compute_transform(float(q_vals[i, j_idx]))
                translations[i] = t.translation.as_arrow_array().to_pylist()[0]
                quaternions[i] = t.quaternion.as_arrow_array().to_pylist()[0]
            rr.send_columns(
                entity_path,
                indexes=[rr.TimeColumn('time', timestamp=ts_arr)],
                columns=rr.Transform3D.columns(
                    translation=translations,
                    quaternion=quaternions,
                    child_frame=[joint.child_link] * n,
                    parent_frame=[joint.parent_link] * n,
                ),
            )
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
        eye_controls = _compute_eye_controls(signals, ep)
        rr.send_blueprint(_build_blueprint(signals, eye_controls))
        yield from drainer.drain()

        _setup_series_names(signals)
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
