"""Dataset utilities for Positronic dataset visualization."""

import io
import logging
import tempfile
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from av.video.stream import VideoStream
from rerun.urdf import UrdfTree

from positronic import keys
from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Kind
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.video import VideoSignal
from positronic.utils.rerun_compat import flatten_numeric, log_series_styles, set_timeline_time

# TODO: 3D visualization roles (pose_signals, joint_signals) are currently read from episode
# static data as flat keys. A cleaner long-term solution is signal-level metadata: each Signal
# would carry a `role` (e.g. 'transform3d', 'joint_position') and optionally a `robot` reference
# linking it to a robot model in static. This would:
# - Eliminate the need for pose_signals/joint_signals keys in static
# - Support multiple robots naturally (each signal references its own model)
# - Keep semantics with the signal that produces them, not in a parallel list
# - Require extending SignalMeta (currently dtype/shape/kind) with user-settable fields
#   and persisting them (parquet metadata or sidecar file)
# See: positronic/dataset/signal.py — SignalMeta, Kind

_POSE_COLORS = {
    'command': [255, 100, 50],  # orange — commanded trajectory
    'state': [50, 200, 255],  # cyan — actual/state trajectory
    'default': [180, 180, 180],  # gray fallback
}


def _pose_color(name: str) -> list[int]:
    prefix = name.split('.')[0] if '.' in name else name
    for suffix, color in _POSE_COLORS.items():
        if prefix.endswith(suffix):
            return color
    return _POSE_COLORS['default']


# A per-element plot of a signal this wide is unreadable, and crowds the video panels out of the
# recording until they never decode.
# TODO: a view that plots a chosen few elements of a wide signal, so it stops being all-or-nothing.
_MAX_PLOTTED_WIDTH = 32


@dataclass
class EpisodeSignals:
    videos: list[str]
    numerics: list[str]
    dims: dict[str, int]
    poses: list[str]
    joints: list[str]

    @property
    def plotted(self) -> dict[str, int]:
        return {name: self.dims[name] for name in self.numerics if self.dims[name] <= _MAX_PLOTTED_WIDTH}

    @property
    def unplotted(self) -> dict[str, int]:
        return {name: dim for name, dim in self.dims.items() if dim > _MAX_PLOTTED_WIDTH}


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


_UNPLOTTED_ENTITY = '/unplotted'


def _unplotted_notice(unplotted: dict[str, int]) -> str:
    lines = '\n'.join(f'- `{name}` — {dim} values' for name, dim in sorted(unplotted.items()))
    return (
        f'### Not plotted\n\n{lines}\n\n'
        f'Wider than {_MAX_PLOTTED_WIDTH} values, so a per-element plot is unreadable and crowds out '
        'the rest of the recording. The signals are in the episode and readable through the dataset API.'
    )


# The retired singular spelling of `keys.JOINT_SIGNALS`. It lives here rather than in `keys` because nothing
# writes it any more — only released data carries it, and only until #587 converts that data.
_SINGULAR_JOINT_SIGNAL = 'joint_signal'


def _collect_signal_groups(ep: Episode) -> EpisodeSignals:
    pose_set = set(ep.static.get(keys.POSE_SIGNALS, []))
    joint_set = set(ep.static.get(keys.JOINT_SIGNALS, []))
    # TODO(#587): drop once the published PhAIL dataset carries the plural key. Its `static.json` has the
    # singular one baked in, so without this a released episode loses its arm model and joint names.
    if _SINGULAR_JOINT_SIGNAL in ep.static:
        joint_set.add(ep.static[_SINGULAR_JOINT_SIGNAL])
    signals = EpisodeSignals(videos=[], numerics=[], dims={}, poses=[], joints=[])
    for name, sig in ep.signals.items():
        if sig.kind == Kind.IMAGE:
            try:
                sig[0]
                signals.videos.append(name)
            except Exception:
                pass
            continue

        try:
            dim = _infer_dims(sig)
        except Exception:
            dim = 1
        signals.numerics.append(name)
        signals.dims[name] = dim
        if name in pose_set:
            signals.poses.append(name)
        if name in joint_set:
            signals.joints.append(name)
    return signals


def _group_signals_by_prefix(signals: EpisodeSignals) -> list[tuple[str, list[str]]]:
    """Group plotted signals by prefix before the first '.'. Preserves insertion order."""
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for sig in signals.plotted:
        groups[sig.split('.')[0] if '.' in sig else sig].append(sig)
    return list(groups.items())


def _build_blueprint(signals: EpisodeSignals, ep: Episode) -> rrb.Blueprint:
    image_views = [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in signals.videos]

    def _ts_view(sig: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            name=sig,
            origin=f'/signals/{sig}',
            plot_legend=rrb.PlotLegend(visible=signals.plotted[sig] > 1),
            axis_y=rrb.ScalarAxis(zoom_lock=True),
        )

    # Group time series by prefix, each group becomes a Tabs container
    series_views = []
    for group_name, sigs in _group_signals_by_prefix(signals):
        if len(sigs) == 1:
            view = _ts_view(sigs[0])
        else:
            view = rrb.Tabs(*[_ts_view(sig) for sig in sigs], name=group_name)
        series_views.append(view)
    if signals.unplotted:
        series_views.append(rrb.TextDocumentView(name='Not plotted', origin=_UNPLOTTED_ENTITY))

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
    joint_set = set(signals.joints)
    joint_names = ep.static.get(keys.JOINT_NAMES)
    pose_set = set(signals.poses)
    for key, dim in signals.plotted.items():
        is_joint_vel = key.endswith('.dq') and f'{key[: -len(".dq")]}.q' in joint_set
        if (key in joint_set or is_joint_vel) and joint_names:
            names = joint_names
        elif key in pose_set and dim == 7:
            # ``Serializers.transform_3d`` is scalar-first: [tx, ty, tz, qw, qx, qy, qz].
            names = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']
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


# Two pixels is the smallest side an encoder can carry: 4:2:0 chroma needs even dimensions.
_MIN_ENCODED_SIDE = 2


def _size_capped_to(width: int, height: int, max_resolution: int) -> tuple[int, int]:
    """``width`` and ``height`` on even sides, scaled down so the long side fits ``max_resolution``."""
    if max_resolution < _MIN_ENCODED_SIDE:
        raise ValueError(f'max_resolution={max_resolution} is below the {_MIN_ENCODED_SIDE}px an encoder can carry')
    scale = min(1.0, max_resolution / max(width, height))
    return max(_MIN_ENCODED_SIDE, int(width * scale) // 2 * 2), max(_MIN_ENCODED_SIDE, int(height * scale) // 2 * 2)


def _encode_frames_as_video(entity_path: str, sig, max_resolution: int) -> None:
    """Encode raw image frames into an H.265 video stream via pyav, with the long side capped."""
    codec = rr.VideoCodec.H265
    container = av.open('/dev/null', 'w', format='hevc')

    # The encoder buffers, so a packet emerges frames after the one it carries, and most of them
    # emerge from the final flush. With max_b_frames = 0 it emerges in source order, which pairs them.
    pending_times: deque[int] = deque()

    def _log_encoded(packets: Iterable[av.Packet]) -> None:
        for packet in packets:
            set_timeline_time('time', pending_times.popleft())
            rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))

    first_frame = np.asarray(sig[0][0])
    h, w = first_frame.shape[:2]
    width, height = _size_capped_to(w, h, max_resolution)
    stream = cast(VideoStream, container.add_stream('libx265', rate=30))
    stream.width = width
    stream.height = height
    stream.max_b_frames = 0

    rr.log(entity_path, rr.VideoStream(codec=codec), static=True)

    for val, ts in sig:
        frame = av.VideoFrame.from_ndarray(np.asarray(val), format='rgb24')
        if (width, height) != (w, h):
            frame = frame.reformat(width=width, height=height)
        pending_times.append(ts)
        _log_encoded(stream.encode(frame))

    _log_encoded(stream.encode())


_DOWNSCALE_OPTIONS = {'crf': '28', 'preset': 'veryfast'}


def _mp4_downscaled_to(src: Path, max_resolution: int) -> bytes:
    """Re-encode ``src`` with its long side at most ``max_resolution``, or return it unchanged if it fits.

    Frame count and presentation times survive the re-encode.
    """
    with av.open(str(src)) as inp:
        in_stream = inp.streams.video[0]
        source = (in_stream.codec_context.width, in_stream.codec_context.height)
        # An mp4 within the cap is passed through, odd sides and all: its own encoder already took them.
        if max(source) <= max_resolution:
            return src.read_bytes()
        width, height = _size_capped_to(*source, max_resolution)

        buffer = io.BytesIO()
        with av.open(buffer, 'w', format='mp4') as out:
            out_stream = out.add_stream('libx264', rate=in_stream.average_rate or 30)
            assert isinstance(out_stream, VideoStream)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = 'yuv420p'
            out_stream.time_base = in_stream.time_base
            out_stream.max_b_frames = 0
            out_stream.options = dict(_DOWNSCALE_OPTIONS)

            for frame in inp.decode(in_stream):
                scaled = frame.reformat(width=width, height=height, format='yuv420p')
                scaled.pts, scaled.time_base = frame.pts, frame.time_base
                out.mux(out_stream.encode(scaled))
            out.mux(out_stream.encode())

    return buffer.getvalue()


def _log_video_signals(
    ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer, max_resolution: int
) -> Iterator[bytes]:
    """Log video signals as AssetVideo + VideoFrameReference (columnar), or as individual images."""
    for name in signals.videos:
        sig = ep.signals[name]
        if isinstance(sig, VideoSignal):
            video_bytes = _mp4_downscaled_to(sig.video_path, max_resolution)
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
            _encode_frames_as_video(name, sig, max_resolution)
        yield from drainer.drain()


def _send_scalar_columns(key: str, ts_arr: np.ndarray, vals: np.ndarray) -> None:
    time_idx = [rr.TimeColumn('time', timestamp=ts_arr)]
    if vals.shape[1] == 1:
        rr.send_columns(f'/signals/{key}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals.ravel()))
        return
    for i in range(vals.shape[1]):
        rr.send_columns(f'/signals/{key}/{i}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals[:, i]))


# Integer nanoseconds put a source recorded at a whole multiple of the cap a hair above it.
_RATE_SLACK = 1e-6


def _decimation_indices(ts_arr: np.ndarray, max_hz: float) -> np.ndarray:
    """Indices into ``ts_arr`` whose timestamps sit at least ``1 / max_hz`` apart.

    A burst either side of a pause stays under the cap: the spacing is read off the timestamps,
    and a pause pulls the average rate below the rate inside each burst.
    """
    if max_hz < 0:
        raise ValueError(f'max_hz={max_hz} is not a rate; 0 is the opt-out')
    if max_hz == 0 or len(ts_arr) < 2:
        return np.arange(len(ts_arr))
    period = np.timedelta64(max(1, round(1e9 / max_hz * (1 - _RATE_SLACK))), 'ns')
    kept = []
    cursor = 0
    while cursor < len(ts_arr):
        kept.append(cursor)
        cursor = int(np.searchsorted(ts_arr, ts_arr[cursor] + period, side='left'))
    return np.asarray(kept, dtype=np.intp)


def _log_numeric_signals(
    ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer, max_hz: float
) -> Generator[bytes, None, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Log numeric time-series via send_columns. Returns pose/joint data for 3D logging.

    A signal too wide to plot is still read, so that a joint or pose vector of any width reaches the
    3D view.
    """
    gripper = ep.static.get(keys.GRIPPER)
    stash_keys = set(signals.poses) | set(signals.joints)
    if gripper:
        stash_keys.add(gripper['signal'])
    pose_data = {}
    unplotted = signals.unplotted

    for key in signals.numerics:
        if key in unplotted and key not in stash_keys:  # nothing would read the values
            continue
        sig = ep.signals[key]
        if len(sig) == 0:
            continue
        ts_arr = np.asarray(sig.keys(), dtype='datetime64[ns]')
        try:
            vals = np.asarray(sig.values(), dtype=np.float64)
        except (TypeError, ValueError):
            # Preserve the rest of the episode when one signal cannot be converted.
            logging.error(f'Signal {key!r} holds values that are not numeric: it is absent from the recording')
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)

        keep = _decimation_indices(ts_arr, max_hz)
        ts_arr, vals = ts_arr[keep], vals[keep]

        if key not in unplotted:
            _send_scalar_columns(key, ts_arr, vals)

        if key in stash_keys:
            pose_data[key] = (ts_arr, vals)

        yield from drainer.drain()

    return pose_data


# Robot visuals render translucent white so the pose-direction markers stay visible through the
# arm and gripper. The URDF loader turns a visual's material color into its mesh ``albedo_factor``.
_ROBOT_VISUAL_RGBA = '1 1 1 0.5'


def _write_urdf_to_dir(urdf_str: str, meshes: dict[str, bytes], dest: Path, namespace: str) -> Path:
    """Write URDF and mesh files to a directory, rewriting mesh filenames to absolute paths, tinting
    every visual translucent white, and prefixing every link and joint name with ``namespace``.

    Rerun keys a transform on the link name, so two arms driving the same model need their link names
    apart or they resolve to one another's frames.
    """
    root = ET.fromstring(urdf_str)
    for mesh_el in root.iter('mesh'):
        filename = mesh_el.get('filename', '')
        if filename in meshes:
            mesh_el.set('filename', str(dest / filename))
    for visual_el in root.iter('visual'):
        material_el = ET.SubElement(visual_el, 'material', name='viewer_translucent')
        ET.SubElement(material_el, 'color', rgba=_ROBOT_VISUAL_RGBA)
    for el in root.iter():
        if el.tag in ('link', 'joint'):
            el.set('name', namespace + el.get('name', ''))
        elif el.tag in ('parent', 'child'):
            el.set('link', namespace + el.get('link', ''))
    urdf_path = dest / 'robot.urdf'
    urdf_path.write_text(ET.tostring(root, encoding='unicode'))
    for name, data in meshes.items():
        safe = Path(name).name  # strip any path components
        (dest / safe).write_bytes(data)
    return urdf_path


def _animate_joint(joint, q_column: np.ndarray, ts_arr: np.ndarray, entity_path: str) -> None:
    """Compute and log transforms for a single URDF joint across all timesteps."""
    n = len(ts_arr)
    translations = np.empty((n, 3), dtype=np.float64)
    quaternions = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        t = joint.compute_transform(float(q_column[i]))
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


_URDF_ANIM_HZ = 15


def _log_urdf_robot(
    ep: Episode, joint_sig: str, numeric_data: dict[str, tuple[np.ndarray, np.ndarray]], drainer: _BinaryStreamDrainer
) -> Iterator[bytes]:
    """Log the episode's robot model, its joints animated by `joint_sig`."""
    joint_names = ep.static.get(keys.JOINT_NAMES)
    urdf_str = ep.static.get(keys.URDF)
    meshes = ep.static.get('meshes')
    if not (joint_names and urdf_str and meshes):
        return
    ts_arr, q_vals = numeric_data[joint_sig]
    if q_vals.shape[1] != len(joint_names):
        logging.warning(
            f'{joint_sig} carries {q_vals.shape[1]} angles for {len(joint_names)} model joints; skipping its model'
        )
        return
    mount = ep.static.get(keys.MOUNTS, {}).get(joint_sig)
    namespace = f'{joint_sig}.'
    prefix = f'/3d/robot/{joint_sig}'

    def link_path(joint) -> str:
        return f'{prefix}/{joint.child_link.removeprefix(namespace)}'

    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = _write_urdf_to_dir(urdf_str, meshes, Path(tmp), namespace)
        rr.log_file_from_path(str(urdf_path), entity_path_prefix=prefix, static=True)
        tree = UrdfTree.from_file_path(str(urdf_path), entity_path_prefix=prefix)

    # An unattached root link frame shares no space with the poses, and the loader leaves it that way.
    rr.log(prefix, rr.Transform3D(translation=mount or np.zeros(3), child_frame=tree.root_link().name), static=True)
    yield from drainer.drain()

    # Robot motion is smooth enough that the model reads well below the plot rate.
    keep = _decimation_indices(ts_arr, _URDF_ANIM_HZ)
    ts_ds, q_ds = ts_arr[keep], q_vals[keep]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for j_idx, name in enumerate(joint_names):
            joint = tree.get_joint_by_name(namespace + name)
            if joint is not None:
                _animate_joint(joint, q_ds[:, j_idx], ts_ds, link_path(joint))
                yield from drainer.drain()

        # A single ``grip`` signal in [0, 1] drives the gripper joints, each joint's axis sign setting
        # its direction; recordings can overshoot slightly, so clip before scaling by ``travel``.
        # TODO: the spec names one signal, so every model grips with it. Arms that grip independently
        # need it pluralized the way `joint_signals` is.
        gripper = ep.static.get(keys.GRIPPER)
        if gripper and gripper['signal'] in numeric_data:
            grip_ts, grip_vals = numeric_data[gripper['signal']]
            grip_keep = _decimation_indices(grip_ts, _URDF_ANIM_HZ)
            finger_pos = np.clip(grip_vals[grip_keep, 0], 0.0, 1.0) * gripper['travel']
            for name in gripper['joints']:
                joint = tree.get_joint_by_name(namespace + name)
                if joint is not None:
                    _animate_joint(joint, finger_pos, grip_ts[grip_keep], link_path(joint))
                    yield from drainer.drain()


def _log_pose_signals(
    ep: Episode,
    signals: EpisodeSignals,
    numeric_data: dict[str, tuple[np.ndarray, np.ndarray]],
    drainer: _BinaryStreamDrainer,
) -> Iterator[bytes]:
    """Log 3D pose: static full trajectory + current position ball + a URDF model per joint signal."""
    for joint_sig in signals.joints:
        if joint_sig in numeric_data:
            yield from _log_urdf_robot(ep, joint_sig, numeric_data, drainer)

    for key in signals.poses:
        if key not in numeric_data:
            continue
        ts_arr, vals = numeric_data[key]
        if vals.ndim < 2 or vals.shape[1] != 7:
            continue
        positions = vals[:, :3]
        color = _pose_color(key)

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


DEFAULT_MAX_HZ = 30.0
DEFAULT_MAX_RESOLUTION = 640


@rr.recording_stream.recording_stream_generator_ctx
def stream_episode_rrd(
    ds: Dataset, episode_id: int, max_hz: float = DEFAULT_MAX_HZ, max_resolution: int = DEFAULT_MAX_RESOLUTION
) -> Iterator[bytes]:
    """Yield an episode RRD as chunks while it is being generated.

    ``max_hz`` thins every numeric signal, ``max_resolution`` caps the long side of each video.
    Both cost fidelity to cut transfer size; pass ``max_hz=0`` and a resolution above the source
    to keep the recording as it was captured.
    """

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
        if signals.unplotted:
            logging.warning(f'Episode {episode_id}: not plotting {signals.unplotted}')
            notice = _unplotted_notice(signals.unplotted)
            rr.log(_UNPLOTTED_ENTITY, rr.TextDocument(notice, media_type=rr.MediaType.MARKDOWN), static=True)
        yield from drainer.drain()

        _setup_series_names(signals, ep)
        yield from drainer.drain()

        yield from _log_video_signals(ep, signals, drainer, max_resolution)
        pose_data = yield from _log_numeric_signals(ep, signals, drainer, max_hz)
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
