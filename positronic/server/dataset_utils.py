"""Dataset utilities for Positronic dataset visualization."""

import io
import logging
import math
import tempfile
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from av.video.stream import VideoStream
from rerun.blueprint.datatypes import TextLogColumn, TextLogColumnKind
from rerun.urdf import UrdfTree

from pimm import Time
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from positronic import keys
from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import RECORDED_TIME, Kind, Signal
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.video import VideoSignal
from positronic.drivers.camera import CAPTURE_TIME
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.eval import keys as eval_keys
from positronic.utils.rerun_compat import flatten_numeric, log_series_styles

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


# A plot of more elements, or more distinct text values, than this is unreadable, and crowds the video
# panels out of the recording until they never decode.
# TODO: a view that plots a chosen few elements of a wide signal, so it stops being all-or-nothing.
_MAX_PLOTTED_WIDTH = 32


@dataclass
class EpisodeSignals:
    videos: list[str]
    numerics: list[str]
    dims: dict[str, int]
    poses: list[str]
    joints: list[str]
    # Each text signal's distinct values, in the order they first appear.
    texts: dict[str, list[str]] = field(default_factory=dict)
    # Width over height of each camera's frames.
    camera_aspects: dict[str, float] = field(default_factory=dict)
    neither_numeric_nor_text: list[str] = field(default_factory=list)

    @property
    def plotted(self) -> dict[str, int]:
        return {name: self.dims[name] for name in self.numerics if self.dims[name] <= _MAX_PLOTTED_WIDTH}

    @property
    def plotted_texts(self) -> dict[str, list[str]]:
        return {name: values for name, values in self.texts.items() if len(values) <= _MAX_PLOTTED_WIDTH}

    @property
    def unplotted(self) -> dict[str, str]:
        """Each signal left out of the plots, and what it holds that a plot cannot show."""
        wide = {name: f'{dim} values' for name, dim in self.dims.items() if dim > _MAX_PLOTTED_WIDTH}
        texts = {
            name: f'{len(values)} distinct text values'
            for name, values in self.texts.items()
            if name not in self.plotted_texts
        }
        other = dict.fromkeys(self.neither_numeric_nor_text, 'values that are not numbers or text')
        return wide | texts | other


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


def _unplotted_notice(unplotted: dict[str, str]) -> str:
    lines = '\n'.join(f'- `{name}` — {holds}' for name, holds in sorted(unplotted.items()))
    return (
        f'### Not plotted\n\n{lines}\n\n'
        f'A plot of more than {_MAX_PLOTTED_WIDTH} values or distinct text values is unreadable and crowds out '
        'the rest of the recording. A text signal is in the text log. The signals are in the episode and '
        'readable through the dataset API.'
    )


# The retired singular spelling of `JOINT_SIGNALS`. It lives here rather than in `keys` because nothing
# writes it any more — only released data carries it, and only until #587 converts that data.
_SINGULAR_JOINT_SIGNAL = 'joint_signal'


def _collect_signal_groups(ep: Episode) -> EpisodeSignals:
    pose_set = set(ep.static.get(eval_keys.POSE_SIGNALS, []))
    joint_set = set(ep.static.get(eval_keys.JOINT_SIGNALS, []))
    # TODO(#587): drop once the published PhAIL dataset carries the plural key. Its `static.json` has the
    # singular one baked in, so without this a released episode loses its arm model and joint names.
    if _SINGULAR_JOINT_SIGNAL in ep.static:
        joint_set.add(ep.static[_SINGULAR_JOINT_SIGNAL])
    signals = EpisodeSignals(videos=[], numerics=[], dims={}, poses=[], joints=[])
    for name, sig in ep.signals.items():
        if sig.kind == Kind.IMAGE:
            try:
                height, width = np.asarray(sig[0][0]).shape[:2]
                signals.videos.append(name)
                signals.camera_aspects[name] = width / height
            except Exception:
                logging.exception(f'Image signal {name!r} has no readable first frame: it is absent from the recording')
            continue

        first = sig[0][0] if len(sig) else 0.0
        if isinstance(first, str):
            signals.texts[name] = list(dict.fromkeys(str(value) for value in sig.values()))
            continue
        if flatten_numeric(first) is None:
            signals.neither_numeric_nor_text.append(name)
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


def _group_signals_by_prefix(names: Iterable[str]) -> list[tuple[str, list[str]]]:
    """Group signals by prefix before the first '.'. Preserves insertion order."""
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for sig in names:
        groups[sig.split('.')[0] if '.' in sig else sig].append(sig)
    return list(groups.items())


_TEXT_LOG_ENTITY = '/text'


def _text_log_view(sig: str) -> rrb.TextLogView:
    hidden = [TextLogColumn(kind, visible=False) for kind in (TextLogColumnKind.EntityPath, TextLogColumnKind.LogLevel)]
    columns = rrb.TextLogColumns(text_log_columns=[*hidden, TextLogColumn(TextLogColumnKind.Body)])
    return rrb.TextLogView(name=sig, origin=f'{_TEXT_LOG_ENTITY}/{sig}', columns=columns)


# The layout is sized for a viewer this many times wider than tall: a wide browser window under the page header.
_VIEWER_ASPECT = 2.4
# Width over height a signal plot reads best at.
_PLOT_CELL_ASPECT = 2.0
# The cameras' share of the top row, and the 3D view's.
_TOP_ROW_SHARES = [3, 1]
_NO_CAMERA_TOP_SHARE = 0.75


def _camera_row_share(signals: EpisodeSignals) -> float:
    """The share of the viewer's height that shows every camera, side by side, without black bands."""
    width = _TOP_ROW_SHARES[0] / sum(_TOP_ROW_SHARES) if signals.poses else 1.0
    return float(np.clip(width / sum(signals.camera_aspects.values()) * _VIEWER_ASPECT, 0.2, 0.75))


def _series_columns(cells: int, height_share: float) -> int:
    """The column count that brings a grid of ``cells`` closest to plot-shaped cells."""
    area_aspect = _VIEWER_ASPECT / height_share

    def miss(columns: int) -> float:
        return abs(math.log(area_aspect * math.ceil(cells / columns) / columns / _PLOT_CELL_ASPECT))

    return min(range(1, cells + 1), key=miss)


@dataclass(frozen=True)
class ReplayLayout:
    """The shares of a replay's rows and of its top row, and the signals each chart of the bottom row plots.

    The top row holds the 3D view and, on its right, the camera grid. ``charts`` holds the bottom row in order. A key
    ``Group/Chart`` puts the chart as a tab in that group, where the group first appears; any other key is a chart
    of its own. A list names each line by its signal, and a dict by its key. A chart with no signal in the episode
    is left out. With ``show_unnamed_signals``, the signals that no chart plots follow, grouped by name prefix.
    """

    row_shares: tuple[float, float]  # top row, bottom row
    top_shares: tuple[float, float]  # 3D view, camera grid
    charts: dict[str, list[str] | dict[str, str]]
    show_unnamed_signals: bool = True


_POSE_VALUE_NAMES = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']  # ``Serializers.transform_3d`` is scalar-first


def _value_names(key: str, signals: EpisodeSignals, ep: Episode) -> list[str] | None:
    """The names of the values of the numeric signal ``key``, where the episode gives them."""
    joint_names = ep.static.get(roboarm_keys.JOINT_NAMES)
    is_joint_vel = (
        key.endswith(keys.JOINT_VEL_SUFFIX)
        and key.removesuffix(keys.JOINT_VEL_SUFFIX) + keys.JOINTS_SUFFIX in signals.joints
    )
    if (key in signals.joints or is_joint_vel) and joint_names:
        return joint_names
    if key in signals.poses and signals.dims[key] == len(_POSE_VALUE_NAMES):
        return _POSE_VALUE_NAMES
    return None


def _signal_views(signals: EpisodeSignals, placed: set[str]) -> list[rrb.View | rrb.Container]:
    """A view of each signal that is not in ``placed``: a group of signals that share a prefix shows as tabs."""

    def _ts_view(sig: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            name=sig,
            origin=f'/signals/{sig}',
            plot_legend=rrb.PlotLegend(visible=signals.plotted[sig] > 1),
            axis_y=rrb.ScalarAxis(zoom_lock=True),
        )

    def _steps_view(sig: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            name=sig,
            origin=f'/signals/{sig}',
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(-0.5, len(signals.texts[sig]) - 0.5), zoom_lock=True),
        )

    def _view(sig: str) -> rrb.TimeSeriesView:
        return _steps_view(sig) if sig in signals.plotted_texts else _ts_view(sig)

    # Each group becomes a Tabs container that opens on its first text signal.
    # A text signal's log is a cell of its own beside its group: a share of one grid cell is too narrow to read.
    unplaced = [sig for sig in [*signals.plotted, *signals.plotted_texts] if sig not in placed]
    views: list[rrb.View | rrb.Container] = []
    for group_name, sigs in _group_signals_by_prefix(unplaced):
        if len(sigs) == 1:
            view = _view(sigs[0])
        else:
            texts = [index for index, sig in enumerate(sigs) if sig in signals.plotted_texts]
            view = rrb.Tabs(*[_view(sig) for sig in sigs], name=group_name, active_tab=texts[0] if texts else None)
        views.append(view)
        views.extend(_text_log_view(sig) for sig in sigs if sig in signals.plotted_texts)
    views.extend(_text_log_view(sig) for sig in signals.texts if sig not in signals.plotted_texts)
    if signals.unplotted:
        views.append(rrb.TextDocumentView(name='Not plotted', origin=_UNPLOTTED_ENTITY))
    return views


def _image_views(signals: EpisodeSignals) -> list[rrb.Spatial2DView]:
    return [rrb.Spatial2DView(name=k, origin=f'/{k}') for k in signals.videos]


def _trajectory_view(signals: EpisodeSignals, ep: Episode) -> rrb.Spatial3DView:
    eye = _compute_eye_controls(signals, ep)
    return rrb.Spatial3DView(
        name='3D Trajectory',
        origin='/3d',
        background=[30, 30, 30],
        line_grid=rrb.LineGrid3D(visible=True),
        eye_controls=eye or rrb.EyeControls3D(),
    )


def _lines(signals: list[str] | dict[str, str]) -> list[tuple[str | None, str]]:
    """Each line of a chart: its label, None where a list names it by its signal, and its signal."""
    return list(signals.items()) if isinstance(signals, dict) else [(None, signal) for signal in signals]


def _chart_view(
    name: str, lines: list[tuple[str | None, str]], signals: EpisodeSignals, ep: Episode
) -> rrb.TimeSeriesView | None:
    shown = [(label, signal) for label, signal in lines if signal in signals.plotted]
    if not shown:
        return None
    line_names: dict[str, str] = {}
    for label, signal in shown:
        path = f'/signals/{signal}'
        width = signals.plotted[signal]
        if width == 1:
            line_names[path] = signal if label is None else label
        elif label is not None:
            names = _value_names(signal, signals, ep)
            line_names.update({f'{path}/{i}': f'{label} {names[i] if names else i}' for i in range(width)})
    return rrb.TimeSeriesView(
        name=name,
        origin='/signals',
        contents=[f'/signals/{signal}/**' for _, signal in shown],
        overrides={path: rr.SeriesLines(names=[line]) for path, line in line_names.items()},
        plot_legend=rrb.PlotLegend(visible=sum(signals.plotted[signal] for _, signal in shown) > 1),
        axis_y=rrb.ScalarAxis(zoom_lock=True),
    )


def _chart_cells(layout: ReplayLayout, signals: EpisodeSignals, ep: Episode) -> list[rrb.View | rrb.Container]:
    """The bottom row's cells: each group of charts, and each chart in no group."""
    cells: dict[tuple[str, bool], list[rrb.TimeSeriesView]] = {}
    for key, chart_signals in layout.charts.items():
        group, slash, chart = key.partition('/')
        view = _chart_view(chart if slash else key, _lines(chart_signals), signals, ep)
        if view is not None:
            cells.setdefault((group if slash else key, bool(slash)), []).append(view)
    # The viewer titles a tab by its child's name and draws no container's own name, so the outer tab carries it.
    return [rrb.Tabs(rrb.Tabs(*views, name=name)) if grouped else views[0] for (name, grouped), views in cells.items()]


def _layout_root(signals: EpisodeSignals, ep: Episode, layout: ReplayLayout) -> rrb.Vertical:
    trajectory = _trajectory_view(signals, ep) if signals.poses else None
    cameras = rrb.Grid(*_image_views(signals)) if signals.videos else None
    top = [
        (view, share) for view, share in zip((trajectory, cameras), layout.top_shares, strict=True) if view is not None
    ]
    bottom = _chart_cells(layout, signals, ep)
    if layout.show_unnamed_signals:
        charted = {signal for lines in layout.charts.values() for _, signal in _lines(lines)}
        bottom.extend(_signal_views(signals, placed=charted & signals.plotted.keys()))

    top_share, bottom_share = layout.row_shares
    rows: list[rrb.View | rrb.Container] = []
    row_shares = []
    if top:
        views = [view for view, _ in top]
        rows.append(views[0] if len(views) == 1 else rrb.Horizontal(*views, column_shares=[s for _, s in top]))
        row_shares.append(top_share)
    if bottom:
        rows.append(rrb.Horizontal(*bottom))
        row_shares.append(bottom_share)
    return rrb.Vertical(*rows, row_shares=row_shares)


def _default_root(signals: EpisodeSignals, ep: Episode) -> rrb.Vertical:
    """Cameras side by side beside the 3D view, over a grid of every signal."""
    series_views = _signal_views(signals, placed=set())

    top_items: list[rrb.View | rrb.Container] = []
    if signals.videos:
        # Widths in proportion to the aspect ratios give every camera one height.
        aspects = [signals.camera_aspects[k] for k in signals.videos]
        top_items.append(rrb.Horizontal(*_image_views(signals), column_shares=aspects))
    if signals.poses:
        top_items.append(_trajectory_view(signals, ep))

    rows = []
    row_shares = []
    top_share = _camera_row_share(signals) if signals.videos else _NO_CAMERA_TOP_SHARE
    if top_items:
        rows.append(top_items[0] if len(top_items) == 1 else rrb.Horizontal(*top_items, column_shares=_TOP_ROW_SHARES))
        row_shares.append(top_share)
    if series_views:
        series_share = 1 - top_share if top_items else 1.0
        rows.append(rrb.Grid(*series_views, grid_columns=_series_columns(len(series_views), series_share)))
        row_shares.append(series_share)
    return rrb.Vertical(*rows, row_shares=row_shares)


# In preference order for playback and rate limiting. Unrecognized timelines retain integer units.
_DURATION_TIMELINES = (RECEIVED_WORLD, RECORDED_TIME, EMITTED_WORLD, RECEIVED_WALL, EMITTED_WALL, CAPTURE_TIME)


def duration_timeline(timelines: Iterable[str]) -> str | None:
    """The preferred nanosecond clock, if the viewer recognizes one."""
    names = set(timelines)
    return next((name for name in _DURATION_TIMELINES if name in names), None)


def _build_blueprint(
    signals: EpisodeSignals, ep: Episode, layout: ReplayLayout | None, timeline: str | None = None
) -> rrb.Blueprint:
    if timeline is not None and timeline not in ep.timelines:
        raise KeyError(timeline)
    if timeline is None:
        timeline = duration_timeline(ep.timelines) or next(iter(ep.timelines), None)
    return rrb.Blueprint(
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TopPanel(state=rrb.PanelState.Expanded),
        rrb.TimePanel(state=rrb.PanelState.Collapsed, timeline=timeline),
        _default_root(signals, ep) if layout is None else _layout_root(signals, ep, layout),
    )


def _setup_series_names(signals: EpisodeSignals, ep: Episode) -> None:
    for key, dim in signals.plotted.items():
        names = _value_names(key, signals, ep)
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


# 4:2:0 chroma needs even dimensions.
_MIN_ENCODED_SIDE = 2

# A packet's pts is its frame's index: a 1/1 base is unrescaled.
_FRAME_INDEX_TIME_BASE = Fraction(1, 1)


def _size_capped_to(width: int, height: int, max_resolution: int) -> tuple[int, int]:
    """``width`` and ``height`` on even sides, scaled down so the long side fits ``max_resolution``."""
    if max_resolution < _MIN_ENCODED_SIDE:
        raise ValueError(f'max_resolution={max_resolution} is below the {_MIN_ENCODED_SIDE}px an encoder can carry')
    scale = min(1.0, max_resolution / max(width, height))
    return max(_MIN_ENCODED_SIDE, int(width * scale) // 2 * 2), max(_MIN_ENCODED_SIDE, int(height * scale) // 2 * 2)


def _timestamp_columns(sig: Signal) -> dict[str, np.ndarray]:
    return {name: np.asarray(sig.timestamps(name), dtype=np.int64) for name in sig.timelines}


def _rerun_indexes(timestamps: dict[str, np.ndarray]) -> list[rr.TimeColumn]:
    return [
        rr.TimeColumn(name, duration=values.astype('timedelta64[ns]'))
        if name in _DURATION_TIMELINES
        else rr.TimeColumn(name, sequence=values)
        for name, values in timestamps.items()
    ]


def _sample_indices(timestamps: dict[str, np.ndarray], max_hz: float) -> np.ndarray:
    if max_hz < 0:
        raise ValueError(f'max_hz={max_hz} is not a rate; 0 is the opt-out')
    timeline = duration_timeline(timestamps)
    if timeline is None:
        return np.arange(len(next(iter(timestamps.values()))))
    return _decimation_indices(timestamps[timeline].astype('timedelta64[ns]'), max_hz)


def _encode_frames_as_video(entity_path: str, sig: Signal, max_resolution: int, max_hz: float) -> None:
    codec = rr.VideoCodec.H265
    # A frame may produce 0, 1 or more packets, and so may the final flush.
    times_by_pts: dict[int, Time] = {}

    def _log_encoded(packets: Iterable[av.Packet]) -> None:
        for packet in packets:
            assert packet.pts is not None
            timestamps = {name: np.array([value], dtype=np.int64) for name, value in times_by_pts[packet.pts].items()}
            rr.send_columns(
                entity_path, indexes=_rerun_indexes(timestamps), columns=rr.VideoStream.columns(sample=[bytes(packet)])
            )

    first_frame = np.asarray(sig[0][0])
    h, w = first_frame.shape[:2]
    width, height = _size_capped_to(w, h, max_resolution)
    kept = set(_sample_indices(_timestamp_columns(sig), max_hz).tolist())
    with av.open('/dev/null', 'w', format='hevc') as container:
        stream = cast(VideoStream, container.add_stream('libx265', rate=30))
        stream.width = width
        stream.height = height
        stream.max_b_frames = 0
        stream.codec_context.time_base = _FRAME_INDEX_TIME_BASE

        rr.log(entity_path, rr.VideoStream(codec=codec), static=True)

        for position, (val, ts) in enumerate(sample for index, sample in enumerate(sig) if index in kept):
            frame = av.VideoFrame.from_ndarray(np.asarray(val), format='rgb24')
            if (width, height) != (w, h):
                frame = frame.reformat(width=width, height=height)
            frame.pts, frame.time_base = position, _FRAME_INDEX_TIME_BASE
            times_by_pts[position] = ts
            _log_encoded(stream.encode(frame))

        _log_encoded(stream.encode())


_DOWNSCALE_OPTIONS = {'crf': '28', 'preset': 'veryfast'}


def _mp4_reduced_to(src: Path, max_resolution: int, kept: np.ndarray | None = None) -> bytes:
    """``src`` re-encoded to at most ``max_resolution`` on the long side and to the frames at the indexes in
    ``kept``; unchanged when it fits and ``kept`` is None."""
    with av.open(str(src)) as inp:
        in_stream = inp.streams.video[0]
        source = (in_stream.codec_context.width, in_stream.codec_context.height)
        # Odd sides pass through; evening them here would re-encode every source that already fits.
        if max(source) <= max_resolution and kept is None:
            return src.read_bytes()
        wanted = None if kept is None else set(kept.tolist())
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

            for index, frame in enumerate(inp.decode(in_stream)):
                if wanted is not None and index not in wanted:
                    continue
                scaled = frame.reformat(width=width, height=height, format='yuv420p')
                scaled.pts, scaled.time_base = frame.pts, frame.time_base
                out.mux(out_stream.encode(scaled))
            out.mux(out_stream.encode())

    return buffer.getvalue()


def _log_video_signals(
    ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer, max_resolution: int, max_hz: float
) -> Iterator[bytes]:
    """Log stored videos as frame references and raw images as a video stream."""
    for name in signals.videos:
        sig = ep.signals[name]
        if not len(sig):
            continue
        if isinstance(sig, VideoSignal):
            timestamps = _timestamp_columns(sig)
            kept = _sample_indices(timestamps, max_hz)
            video_bytes = _mp4_reduced_to(sig.video_path, max_resolution, kept if len(kept) < len(sig) else None)
            asset = rr.AssetVideo(contents=video_bytes, media_type='video/mp4')
            rr.log(name, asset, static=True)

            frame_pts_ns = asset.read_frame_timestamps_nanos()
            rr.send_columns(
                name,
                indexes=_rerun_indexes({name: values[kept] for name, values in timestamps.items()}),
                columns=rr.VideoFrameReference.columns_nanos(frame_pts_ns),
            )
        else:
            _encode_frames_as_video(name, sig, max_resolution, max_hz)
        yield from drainer.drain()


def _send_scalar_columns(key: str, timestamps: dict[str, np.ndarray], vals: np.ndarray) -> None:
    time_idx = _rerun_indexes(timestamps)
    if vals.shape[1] == 1:
        rr.send_columns(f'/signals/{key}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals.ravel()))
        return
    for i in range(vals.shape[1]):
        rr.send_columns(f'/signals/{key}/{i}', indexes=time_idx, columns=rr.Scalars.columns(scalars=vals[:, i]))


# Integer nanoseconds put a source recorded at the cap a hair above it.
_RATE_SLACK = 1e-6


def _decimation_indices(ts_arr: np.ndarray, max_hz: float) -> np.ndarray:
    """Indices into ``ts_arr`` whose timestamps sit at least ``1 / max_hz`` apart."""
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
) -> Generator[bytes, None, dict[str, tuple[dict[str, np.ndarray], np.ndarray]]]:
    """Log numeric time-series via send_columns. Returns pose/joint data for 3D logging.

    A signal too wide to plot is still read, so that a joint or pose vector of any width reaches the
    3D view.
    """
    gripper = ep.static.get(roboarm_keys.GRIPPER)
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
        timestamps = _timestamp_columns(sig)
        try:
            vals = np.asarray(sig.values(), dtype=np.float64)
        except (TypeError, ValueError):
            # Preserve the rest of the episode when one signal cannot be converted.
            logging.error(f'Signal {key!r} holds values that are not numeric: it is absent from the recording')
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)

        keep = _sample_indices(timestamps, max_hz)
        timestamps, vals = {name: values[keep] for name, values in timestamps.items()}, vals[keep]

        if key not in unplotted:
            _send_scalar_columns(key, timestamps, vals)

        if key in stash_keys:
            pose_data[key] = (timestamps, vals)

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


def _animate_joint(joint, q_column: np.ndarray, timestamps: dict[str, np.ndarray], entity_path: str) -> None:
    """Compute and log transforms for a single URDF joint across all timesteps."""
    n = len(q_column)
    translations = np.empty((n, 3), dtype=np.float64)
    quaternions = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        t = joint.compute_transform(float(q_column[i]))
        translations[i] = t.translation.as_arrow_array().to_pylist()[0]
        quaternions[i] = t.quaternion.as_arrow_array().to_pylist()[0]
    rr.send_columns(
        entity_path,
        indexes=_rerun_indexes(timestamps),
        columns=rr.Transform3D.columns(
            translation=translations,
            quaternion=quaternions,
            child_frame=[joint.child_link] * n,
            parent_frame=[joint.parent_link] * n,
        ),
    )


_URDF_ANIM_HZ = 15


def _log_urdf_robot(
    ep: Episode,
    joint_sig: str,
    numeric_data: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    drainer: _BinaryStreamDrainer,
) -> Iterator[bytes]:
    """Log the episode's robot model, its joints animated by `joint_sig`."""
    joint_names = ep.static.get(roboarm_keys.JOINT_NAMES)
    urdf_str = ep.static.get(roboarm_keys.URDF)
    meshes = ep.static.get('meshes')
    if not (joint_names and urdf_str and meshes):
        return
    timestamps, q_vals = numeric_data[joint_sig]
    if q_vals.shape[1] != len(joint_names):
        logging.warning(
            f'{joint_sig} carries {q_vals.shape[1]} angles for {len(joint_names)} model joints; skipping its model'
        )
        return
    mount = ep.static.get(eval_keys.MOUNTS, {}).get(joint_sig)
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

    keep = _sample_indices(timestamps, _URDF_ANIM_HZ)
    ts_ds, q_ds = {name: values[keep] for name, values in timestamps.items()}, q_vals[keep]

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
        gripper = ep.static.get(roboarm_keys.GRIPPER)
        if gripper and gripper['signal'] in numeric_data:
            grip_ts, grip_vals = numeric_data[gripper['signal']]
            grip_keep = _sample_indices(grip_ts, _URDF_ANIM_HZ)
            grip_times = {name: values[grip_keep] for name, values in grip_ts.items()}
            finger_pos = np.clip(grip_vals[grip_keep, 0], 0.0, 1.0) * gripper['travel']
            for name in gripper['joints']:
                joint = tree.get_joint_by_name(namespace + name)
                if joint is not None:
                    _animate_joint(joint, finger_pos, grip_times, link_path(joint))
                    yield from drainer.drain()


def _log_pose_signals(
    ep: Episode,
    signals: EpisodeSignals,
    numeric_data: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    drainer: _BinaryStreamDrainer,
) -> Iterator[bytes]:
    """Log 3D pose: static full trajectory + current position ball + a URDF model per joint signal."""
    for joint_sig in signals.joints:
        if joint_sig in numeric_data:
            yield from _log_urdf_robot(ep, joint_sig, numeric_data, drainer)

    for key in signals.poses:
        if key not in numeric_data:
            continue
        timestamps, vals = numeric_data[key]
        if vals.ndim < 2 or vals.shape[1] != 7:
            continue
        positions = vals[:, :3]
        color = _pose_color(key)

        _log_static_trail(f'/3d/{key}/trail', positions, color)

        rr.send_columns(
            f'/3d/{key}',
            indexes=_rerun_indexes(timestamps),
            columns=[
                *rr.Points3D.columns(positions=positions).partition([1] * len(vals)),
                *rr.Points3D.columns(colors=np.tile(color, (len(vals), 1))).partition([1] * len(vals)),
                *rr.Points3D.columns(radii=np.full(len(vals), 0.01)),
            ],
        )
        yield from drainer.drain()


def _changes(values: np.ndarray) -> np.ndarray:
    """Indices where ``values`` differ from the sample before, the first sample included."""
    return np.flatnonzero(np.concatenate([[True], values[1:] != values[:-1]]))


def _log_text_signals(ep: Episode, signals: EpisodeSignals, drainer: _BinaryStreamDrainer) -> Iterator[bytes]:
    """Log each text value to the text log, and a plotted text signal as a step plot of its value indices.

    A text signal is logged where its value changes rather than thinned to a rate, so no short-lived value drops out.
    """
    plotted = signals.plotted_texts
    ep_signals = ep.signals  # `Episode.signals` builds a new dict on every read
    for key in signals.texts:
        sig = ep_signals[key]
        if not len(sig):
            continue
        timestamps = _timestamp_columns(sig)
        texts = np.asarray([str(value) for value in sig.values()], dtype=object)
        changes = _changes(texts)
        time_idx = _rerun_indexes({name: values[changes] for name, values in timestamps.items()})
        rr.send_columns(f'{_TEXT_LOG_ENTITY}/{key}', indexes=time_idx, columns=rr.TextLog.columns(text=texts[changes]))

        if key in plotted:
            values = plotted[key]
            label = ', '.join(f'{index} {value}' for index, value in enumerate(values))
            style = rr.SeriesLines(names=[label], interpolation_mode=rr.components.InterpolationMode.StepAfter)
            rr.log(f'/signals/{key}', style, static=True)
            # The last sample holds the final value to the end of the episode.
            shown = np.union1d(changes, [len(texts) - 1])
            index_of = {value: index for index, value in enumerate(values)}
            indices = np.asarray([index_of[text] for text in texts[shown]], dtype=np.float64)
            _send_scalar_columns(
                key, {name: values[shown] for name, values in timestamps.items()}, indices.reshape(-1, 1)
            )
        yield from drainer.drain()


DEFAULT_MAX_HZ = 30.0
DEFAULT_MAX_RESOLUTION = 640


@rr.recording_stream.recording_stream_generator_ctx
def stream_episode_rrd(
    ds: Dataset,
    episode_id: int,
    max_hz: float = DEFAULT_MAX_HZ,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
    layout: ReplayLayout | None = None,
    timeline: str | None = None,
) -> Iterator[bytes]:
    """Yield an episode RRD as chunks while it is being generated.

    Videos and numeric signals with a recognized nanosecond clock are thinned to ``max_hz``.
    ``max_hz=0`` with a resolution above the source keeps the recording as it was captured.
    Without a ``layout``, the replay shows the cameras beside
    the 3D view over a grid of every signal. ``timeline`` selects the initial view; all timelines are exported.
    """

    ep = ds[episode_id]
    assert isinstance(ep, Episode)
    logging.info(f'Streaming RRD for episode {episode_id}')

    dataset_root = get_dataset_root(ds)
    dataset_name = Path(dataset_root).name if dataset_root else 'unknown'
    recording_id = f'positronic_ds_{dataset_name}_episode_{episode_id}'
    rec = rr.RecordingStream(application_id=recording_id)
    drainer = _BinaryStreamDrainer(rec.binary_stream(), min_bytes=2**20)

    with rec:
        signals = _collect_signal_groups(ep)
        rr.send_blueprint(_build_blueprint(signals, ep, layout, timeline))
        if signals.unplotted:
            logging.warning(f'Episode {episode_id}: not plotting {signals.unplotted}')
            notice = _unplotted_notice(signals.unplotted)
            rr.log(_UNPLOTTED_ENTITY, rr.TextDocument(notice, media_type=rr.MediaType.MARKDOWN), static=True)
        yield from drainer.drain()

        _setup_series_names(signals, ep)
        yield from drainer.drain()

        yield from _log_video_signals(ep, signals, drainer, max_resolution, max_hz)
        pose_data = yield from _log_numeric_signals(ep, signals, drainer, max_hz)
        yield from _log_text_signals(ep, signals, drainer)
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
