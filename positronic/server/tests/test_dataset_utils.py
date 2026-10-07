import io
import typing
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import numpy as np
import psutil
import pyarrow as pa
import pytest
import rerun.blueprint as rrb
import rerun.recording as rr_recording

from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from positronic import geom, keys
from positronic.cfg.server import single_arm_replay_layout
from positronic.dataset import Time
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter, LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers, expand_suffixed
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.eval import keys as eval_keys
from positronic.server import dataset_utils
from positronic.server.dataset_utils import (
    _MAX_PLOTTED_WIDTH,
    ReplayLayout,
    _build_blueprint,
    _collect_signal_groups,
    _decimation_indices,
    _mp4_reduced_to,
    _size_capped_to,
    _unplotted_notice,
    _write_urdf_to_dir,
    stream_episode_rrd,
)


def _episode(ep_dir, widths: dict[str, int], static: dict[str, Any] | None = None) -> DiskEpisode:
    with DiskEpisodeWriter(ep_dir) as writer:
        for name, width in widths.items():
            writer.append(name, np.zeros(width, dtype=np.float32), Time(**{RECORDED_TIME: 1000}))
            writer.append(name, np.ones(width, dtype=np.float32), Time(**{RECORDED_TIME: 2000}))
        for name, value in (static or {}).items():
            writer.set_static(name, value)
    return DiskEpisode(ep_dir)


def test_narrow_signals_are_plotted(tmp_path):
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7, keys.GRIP: 1}))

    assert signals.plotted == {keys.JOINTS: 7, keys.GRIP: 1}
    assert signals.unplotted == {}


def test_wide_signal_is_named_instead_of_plotted(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7, 'wide_signal': width}))

    assert signals.plotted == {keys.JOINTS: 7}
    assert signals.unplotted == {'wide_signal': f'{width} values'}


def test_wide_signal_still_reaches_the_3d_view(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: width}))

    assert signals.numerics == [keys.JOINTS]
    assert signals.dims == {keys.JOINTS: width}


def test_every_joint_signal_the_episode_records_is_collected(tmp_path):
    widths = {'robot_state.left.q': 6, 'robot_state.right.q': 6, keys.GRIP: 1}
    static = {eval_keys.JOINT_SIGNALS: ['robot_state.left.q', 'robot_state.right.q', 'robot_state.absent.q']}

    signals = _collect_signal_groups(_episode(tmp_path / 'ep', widths, static))

    assert sorted(signals.joints) == ['robot_state.left.q', 'robot_state.right.q']


def test_released_episodes_singular_joint_signal_still_counts(tmp_path):
    # TODO(#587): delete with the bridge in `_collect_signal_groups`.
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: 7}, {'joint_signal': keys.JOINTS}))

    assert signals.joints == [keys.JOINTS]


def test_urdf_link_and_joint_names_carry_the_namespace(tmp_path):
    urdf = """<robot name="toy">
      <link name="base"/>
      <link name="arm"/>
      <joint name="shoulder" type="revolute">
        <parent link="base"/>
        <child link="arm"/>
      </joint>
    </robot>"""

    urdf_path = _write_urdf_to_dir(urdf, {}, tmp_path, 'robot_state.left.q.')

    root = ET.fromstring(urdf_path.read_text())
    assert [el.get('name') for el in root.iter('link')] == ['robot_state.left.q.base', 'robot_state.left.q.arm']
    assert [el.get('name') for el in root.iter('joint')] == ['robot_state.left.q.shoulder']
    assert [el.get('link') for el in root.iter('parent')] == ['robot_state.left.q.base']
    assert [el.get('link') for el in root.iter('child')] == ['robot_state.left.q.arm']


def test_notice_names_every_unplotted_signal_and_its_width():
    notice = _unplotted_notice({'wide_signal': '866 values', 'wider_signal': '120 values'})

    assert '`wide_signal` — 866 values' in notice
    assert '`wider_signal` — 120 values' in notice


_STATES = ['floating', 'floating', 'reaching', 'contact', 'reaching', 'reaching', 'at-target']


def _text_episode(ep_dir, texts: dict[str, list[Any]]) -> DiskEpisode:
    with DiskEpisodeWriter(ep_dir) as writer:
        for name, values in texts.items():
            for i, value in enumerate(values):
                writer.append(name, value, Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
    return DiskEpisode(ep_dir)


def test_a_text_signal_is_plotted_by_its_values_in_order_of_first_appearance(tmp_path):
    signals = _collect_signal_groups(_text_episode(tmp_path / 'ep', {'progress.state': _STATES}))

    assert signals.numerics == []
    assert signals.plotted_texts == {'progress.state': ['floating', 'reaching', 'contact', 'at-target']}
    assert signals.unplotted == {}


def test_a_text_signal_with_too_many_values_is_named_instead_of_plotted(tmp_path):
    count = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_text_episode(tmp_path / 'ep', {'prompt': [f'p{i}' for i in range(count)]}))

    assert signals.plotted_texts == {}
    assert list(signals.texts) == ['prompt']
    assert signals.unplotted == {'prompt': f'{count} distinct text values'}


def test_an_array_of_text_is_named_instead_of_plotted(tmp_path):
    words = [np.array(['a', 'b']), np.array(['c', 'd'])]
    signals = _collect_signal_groups(_text_episode(tmp_path / 'ep', {'words': words}))

    assert signals.numerics == []
    assert signals.texts == {}
    assert signals.unplotted == {'words': 'values that are not numbers or text'}


def test_a_text_signal_reaches_the_recording_as_a_plot_and_a_text_log(tmp_path):
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as dataset_writer, dataset_writer.new_episode() as writer:
        for i, state in enumerate(_STATES):
            writer.append('progress.state', state, Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
    rrd = tmp_path / 'ep.rrd'
    rrd.write_bytes(b''.join(stream_episode_rrd(LocalDataset(root), 0)))

    columns = rr_recording.load_recording(str(rrd)).schema().component_columns()
    archetypes = {(column.entity_path, column.archetype) for column in columns}
    assert ('/signals/progress.state', 'rerun.archetypes.Scalars') in archetypes
    assert ('/text/progress.state', 'rerun.archetypes.TextLog') in archetypes


def _numeric_dataset(root: Path) -> LocalDataset:
    with LocalDatasetWriter(root) as dataset_writer, dataset_writer.new_episode() as writer:
        for i in range(3):
            writer.append('x', np.array([i, 1.0]), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
    return LocalDataset(root)


@pytest.mark.parametrize('read_all', [True, False])
def test_an_ended_stream_releases_its_recording_threads(tmp_path, read_all):
    # A leaked recording keeps two threads; other tests in the worker may start or end a few of their own.
    streams = 10
    ds = _numeric_dataset(tmp_path / 'ds')
    process = psutil.Process()
    b''.join(stream_episode_rrd(ds, 0))
    threads = process.num_threads()

    for _ in range(streams):
        stream = stream_episode_rrd(ds, 0)
        next(stream)
        if read_all:
            b''.join(stream)
        stream.close()

    assert process.num_threads() < threads + streams


def test_a_streamed_recording_carries_its_blueprint(tmp_path):
    rrd = b''.join(stream_episode_rrd(_numeric_dataset(tmp_path / 'ds'), 0))

    assert rrb.archetypes.ContainerBlueprint.archetype().encode() in rrd


def _null_drainer() -> dataset_utils._BinaryStreamDrainer:
    return dataset_utils._BinaryStreamDrainer(dataset_utils.rr.RecordingStream('test').binary_stream(), min_bytes=1)


@pytest.mark.parametrize('selected', [None, 'tick'])
def test_numeric_pose_and_video_samples_keep_their_coordinates_after_thinning(tmp_path, monkeypatch, selected):
    sent = {}
    send = dataset_utils.rr.send_columns

    def send_columns(path, indexes, columns):
        sent[path] = {index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes}
        send(path, indexes=indexes, columns=columns)

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as ds, ds.new_episode() as writer:
        writer.set_static(eval_keys.POSE_SIGNALS, ['pose'])
        for i in range(6):
            ts = Time(**{RECEIVED_WORLD: i * 10_000_000, RECEIVED_WALL: 2**53 + i + 1, 'tick': i})
            writer.append('pose', np.array([i, 0, 0, 0, 0, 0, 1], dtype=float), ts)
            writer.append('camera', np.full((16, 16, 3), i * 30, dtype=np.uint8), ts)
            writer.append('other', i, Time(device=i))
    rrd = tmp_path / 'ep.rrd'
    rrd.write_bytes(b''.join(stream_episode_rrd(LocalDataset(root), 0, max_hz=30, timeline=selected)))

    kept = {RECEIVED_WORLD: [0, 40_000_000], RECEIVED_WALL: [2**53 + 1, 2**53 + 5], 'tick': [0, 4]}
    assert sent['/signals/pose/0'] == kept
    assert sent['/3d/pose'] == kept
    assert sent['camera'] == kept
    assert sent['/signals/other'] == {'device': list(range(6))}
    schema = rr_recording.load_recording(str(rrd)).schema()
    assert {column.name for column in schema.index_columns()} == {*kept, 'device'}


def test_timeline_kinds_preserve_integer_precision():
    names = [EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD, RECORDED_TIME, 'tick']
    values = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    indexes = dataset_utils._rerun_indexes(dict.fromkeys(names, values))
    for index in indexes:
        array = index.as_arrow_array()
        assert array.cast(pa.int64()).to_pylist() == values.tolist()
        assert array.type == (pa.int64() if index.timeline_name() == 'tick' else pa.duration('ns'))


@pytest.mark.parametrize(
    'names, expected',
    [
        ((RECORDED_TIME,), RECORDED_TIME),
        ((EMITTED_WALL, RECEIVED_WORLD, RECORDED_TIME), RECEIVED_WORLD),
        (('tick',), 'tick'),
    ],
)
def test_initial_timeline_uses_world_then_legacy_clock_then_an_available_axis(tmp_path, names, expected):
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        writer.append('value', 1, Time(**dict.fromkeys(names, 1)))
    ep = DiskEpisode(tmp_path / 'ep')
    blueprint = _build_blueprint(_collect_signal_groups(ep), ep, None)
    assert blueprint.time_panel.timeline == expected


def test_a_text_signal_is_logged_where_its_value_changes(tmp_path, monkeypatch):
    sent: dict[str, tuple[list[int], list[Any]]] = {}
    styles: dict[str, Any] = {}

    def send_columns(path, indexes, columns):
        times = indexes[0].as_arrow_array().cast(pa.int64()).to_pylist()
        sent[path] = (times, [value for column in columns for value in column.as_arrow_array().to_pylist()])

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda path, value, static=False: styles.__setitem__(path, value))
    ep = _text_episode(tmp_path / 'ep', {'progress.state': _STATES})

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    second = 1_000_000_000
    changes = [second, 3 * second, 4 * second, 5 * second, 7 * second]
    texts = [['floating'], ['reaching'], ['contact'], ['reaching'], ['at-target']]
    assert sent['/text/progress.state'] == (changes, texts)
    assert sent['/signals/progress.state'] == (changes, [[0.0], [1.0], [2.0], [1.0], [3.0]])
    names = styles['/signals/progress.state'].names.as_arrow_array().to_pylist()
    assert names == ['0 floating, 1 reaching, 2 contact, 3 at-target']


def test_a_text_log_entry_keeps_all_its_coordinates(tmp_path, monkeypatch):
    sent: dict[str, dict[str, list[int]]] = {}

    def send_columns(path, indexes, columns):
        sent[path] = {index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes}

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    machine_clock = 1_011_234_567_890_123  # nanoseconds since boot, far from the epoch
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        writer.append('robot.q', np.zeros(2), Time(**{RECORDED_TIME: machine_clock}))
        writer.append(
            'progress.state', 'floating', Time(**{RECEIVED_WALL: machine_clock, RECEIVED_WORLD: 1000, 'tick': 4})
        )
        writer.append(
            'progress.state', 'reaching', Time(**{RECEIVED_WALL: machine_clock + 1000, RECEIVED_WORLD: 2000, 'tick': 5})
        )
    ep = DiskEpisode(tmp_path / 'ep')

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    assert sent['/text/progress.state'] == {
        RECEIVED_WALL: [machine_clock, machine_clock + 1000],
        RECEIVED_WORLD: [1000, 2000],
        'tick': [4, 5],
    }


def test_disjoint_text_timelines_are_exported(tmp_path, monkeypatch):
    sent: dict[str, dict[str, list[int]]] = {}

    def send_columns(path, indexes, columns):
        sent[path] = {index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes}

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        writer.append('progress.state', 'floating', Time(**{RECORDED_TIME: 4_000_000_000}))
        writer.append('device.state', 'ready', Time(device=7))
    ep = DiskEpisode(tmp_path / 'ep')

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    assert sent['/text/progress.state'] == {RECORDED_TIME: [4_000_000_000]}
    assert sent['/text/device.state'] == {'device': [7]}


def test_a_text_log_displays_all_timelines():
    columns = dataset_utils._text_log_view('progress.state').properties['TextLogColumns']
    assert isinstance(columns, rrb.TextLogColumns) and columns.timeline_columns is None


def _tabs(container: Any) -> list[rrb.Tabs]:
    found = [container] if isinstance(container, rrb.Tabs) else []
    return found + [tabs for child in getattr(container, 'contents', None) or [] for tabs in _tabs(child)]


def test_a_tab_group_opens_on_its_text_signal(tmp_path):
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        for i, state in enumerate(_STATES):
            writer.append('device.level', float(i), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
            writer.append('device.state', state, Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
            writer.append('robot.q', np.zeros(2), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
            writer.append('robot.dq', np.zeros(2), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
    ep = DiskEpisode(tmp_path / 'ep')

    blueprint = _build_blueprint(_collect_signal_groups(ep), ep, None)
    tabs = {tab.name: tab.active_tab for tab in _tabs(blueprint.root_container)}

    assert (tabs['device'], tabs['robot']) == (1, None)


def _layout(charts: dict[str, list[str] | dict[str, str]], show_unnamed_signals: bool = True) -> ReplayLayout:
    return ReplayLayout(
        split_shares=(3, 1), view_shares=(1, 3), charts=charts, show_unnamed_signals=show_unnamed_signals
    )


def _root(signals: dataset_utils.EpisodeSignals, ep: DiskEpisode, layout: ReplayLayout | None) -> Any:
    return _build_blueprint(signals, ep, layout).root_container


def _bottom_row(ep: DiskEpisode, layout: ReplayLayout) -> list[Any]:
    return _root(_collect_signal_groups(ep), ep, layout).contents[-1].contents


_GRIP: dict[str, list[str] | dict[str, str]] = {'Grip': {'Target': keys.TARGET_GRIP, 'Current': keys.GRIP}}


def test_a_layout_puts_the_3d_view_left_of_the_cameras_at_its_shares_over_the_charts(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.EE_POSE: 7, keys.GRIP: 1}, {eval_keys.POSE_SIGNALS: [keys.EE_POSE]})
    signals = replace(_collect_signal_groups(ep), videos=['camera'])

    root = _root(signals, ep, _layout(_GRIP))

    top, bottom = root.contents
    assert list(root.row_shares) == [3, 1]
    assert [type(view) for view in top.contents] == [rrb.Spatial3DView, rrb.Grid]
    assert list(top.column_shares) == [1, 3]
    assert [type(view) for view in top.contents[1].contents] == [rrb.Spatial2DView]
    assert isinstance(bottom, rrb.Horizontal)


def test_a_layout_with_the_3d_view_in_the_camera_grid_puts_it_after_the_cameras(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.EE_POSE: 7, keys.GRIP: 1}, {eval_keys.POSE_SIGNALS: [keys.EE_POSE]})
    signals = replace(_collect_signal_groups(ep), videos=['left', 'right'])

    root = _root(signals, ep, replace(_layout(_GRIP), trajectory_in_camera_grid=True))

    top, bottom = root.contents
    assert list(root.row_shares) == [3, 1]
    assert isinstance(top, rrb.Grid)
    assert [(type(view), view.name) for view in top.contents] == [
        (rrb.Spatial2DView, 'left'),
        (rrb.Spatial2DView, 'right'),
        (rrb.Spatial3DView, '3D Trajectory'),
    ]
    assert isinstance(bottom, rrb.Horizontal)


def test_a_layout_with_the_charts_beside_stacks_them_right_of_the_views_at_its_split(tmp_path):
    ep = _episode(
        tmp_path / 'ep', {keys.EE_POSE: 7, keys.JOINTS: 7, keys.GRIP: 1}, {eval_keys.POSE_SIGNALS: [keys.EE_POSE]}
    )
    signals = replace(_collect_signal_groups(ep), videos=['left', 'right'])
    charts = {'Robot State/Joints': [keys.JOINTS], **_GRIP}
    layout = replace(_layout(charts, False), trajectory_in_camera_grid=True, charts_beside=True)

    root = _root(signals, ep, layout)

    media, column = root.contents
    assert isinstance(root, rrb.Horizontal)
    assert np.asarray(root.column_shares).tolist() == [3, 1]
    assert isinstance(media, rrb.Grid)
    assert isinstance(column, rrb.Vertical)
    assert [type(chart) for chart in column.contents] == [rrb.Tabs, rrb.TimeSeriesView]


def test_a_top_view_with_no_signal_to_show_is_left_out(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.GRIP: 1})

    assert [type(row) for row in _root(_collect_signal_groups(ep), ep, _layout(_GRIP)).contents] == [rrb.Horizontal]


def test_a_group_shows_its_charts_as_tabs_under_its_name_where_it_first_appears(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.JOINTS: 7, keys.JOINT_VEL: 7, keys.GRIP: 1, keys.TARGET_GRIP: 1})
    charts = {'Robot State/Joints': [keys.JOINTS], **_GRIP, 'Robot State/Joints Vel': [keys.JOINT_VEL]}

    group, grip = _bottom_row(ep, _layout(charts, show_unnamed_signals=False))

    (named,) = group.contents
    assert named.name == 'Robot State'
    assert [chart.name for chart in named.contents] == ['Joints', 'Joints Vel']
    assert isinstance(grip, rrb.TimeSeriesView)
    assert grip.name == 'Grip'


def _line_names(view: Any) -> dict[str, list[str]]:
    return {str(path): lines.names.as_arrow_array().to_pylist() for path, lines in view.visualizer_overrides.items()}


def test_a_dict_names_each_line_by_its_key_and_a_list_by_its_signal(tmp_path):
    static = {eval_keys.JOINT_SIGNALS: [keys.JOINTS], roboarm_keys.JOINT_NAMES: ['j1', 'j2']}
    widths = {keys.GRIP: 1, keys.TARGET_GRIP: 1, keys.JOINTS: 2, keys.TARGET_JOINTS: 2}
    ep = _episode(tmp_path / 'ep', widths, static)
    charts = {
        'Labelled': {'Current': keys.GRIP, 'State': keys.JOINTS},
        'Listed': [keys.TARGET_GRIP, keys.TARGET_JOINTS],
    }

    labelled, listed = _bottom_row(ep, _layout(charts, show_unnamed_signals=False))

    assert _line_names(labelled) == {
        f'/signals/{keys.GRIP}': ['Current'],
        f'/signals/{keys.JOINTS}/0': ['State j1'],
        f'/signals/{keys.JOINTS}/1': ['State j2'],
    }
    assert _line_names(listed) == {f'/signals/{keys.TARGET_GRIP}': [keys.TARGET_GRIP]}


def test_the_single_arm_layout_charts_the_motion_signals_of_each_kind_of_arm_command():
    one_of_each_kind = (
        command.CartesianPosition(geom.Transform3D.identity),
        command.CartesianDelta(geom.Transform3D.identity),
        command.JointPosition(np.zeros(7)),
        command.JointDelta(np.zeros(7)),
    )
    assert {type(cmd) for cmd in one_of_each_kind} == set(typing.get_args(command.CommandType))
    recorded = {
        name
        for cmd in one_of_each_kind
        for name, _ in expand_suffixed(keys.ROBOT_COMMAND, Serializers.robot_command(cmd))
    }
    charted = {
        signal
        for signals in single_arm_replay_layout.instantiate().charts.values()
        for signal in (signals.values() if isinstance(signals, dict) else signals)
    }

    assert recorded <= charted


def test_a_chart_plots_the_signals_the_episode_records_and_a_chart_with_none_is_left_out(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.GRIP: 1})
    charts = {'Commands/Joints': [keys.TARGET_JOINTS], **_GRIP}

    (view,) = _bottom_row(ep, _layout(charts, show_unnamed_signals=False))

    assert view.contents == [f'/signals/{keys.GRIP}/**']


def test_the_signals_no_chart_plots_follow_the_charts_in_one_group(tmp_path):
    widths = {keys.GRIP: 1, keys.TARGET_GRIP: 1, 'progress.delivered': 1, 'progress.placed': 1, 'device.level': 1}
    ep = _episode(tmp_path / 'ep', widths)

    grip, other = _bottom_row(ep, _layout(_GRIP))

    (group,) = other.contents
    assert grip.name == 'Grip'
    assert group.name == 'Other'
    assert [view.name for view in group.contents] == ['progress', 'device.level']


def test_with_unnamed_signals_off_only_the_charts_show(tmp_path):
    ep = _episode(tmp_path / 'ep', {keys.GRIP: 1, keys.TARGET_GRIP: 1, 'progress.delivered': 1, 'wide': 33})

    assert [view.name for view in _bottom_row(ep, _layout(_GRIP, show_unnamed_signals=False))] == ['Grip']


def test_a_chart_plots_a_text_signal_by_the_index_of_its_value_and_keeps_its_legend(tmp_path):
    ep = _text_episode(tmp_path / 'ep', {'progress.state': _STATES})

    (view,) = _bottom_row(ep, _layout({'Progress': ['progress.state']}))

    assert view.contents == ['/signals/progress.state/**']
    assert _line_names(view) == {}


def _default_column(ep: DiskEpisode) -> Any:
    return _root(_collect_signal_groups(ep), ep, None).contents[-1]


def _title(cell: Any) -> str:
    """The name on a cell: its group's, where it is a group of tabs."""
    named = next(iter(cell.contents)) if isinstance(cell, rrb.Tabs) else cell
    return str(named.name)


def test_the_default_puts_the_views_in_one_grid_and_the_arm_charts_in_a_column_on_its_right(tmp_path):
    widths = {keys.EE_POSE: 7, keys.JOINTS: 7, keys.GRIP: 1, keys.TARGET_GRIP: 1, keys.TARGET_EE_POSE: 7}
    ep = _episode(tmp_path / 'ep', widths, {eval_keys.POSE_SIGNALS: [keys.EE_POSE]})
    signals = replace(_collect_signal_groups(ep), videos=['wrist'])

    root = _root(signals, ep, None)

    grid, column = root.contents
    assert isinstance(root, rrb.Horizontal)
    assert np.asarray(root.column_shares).tolist() == [3, 1]
    assert [(type(view), view.name) for view in grid.contents] == [
        (rrb.Spatial2DView, 'wrist'),
        (rrb.Spatial3DView, '3D Trajectory'),
    ]
    assert [_title(cell) for cell in column.contents] == ['Robot State', 'Grip', 'Robot Commands – End Effector']


def test_the_default_gives_each_arm_of_a_two_arm_robot_its_charts_as_a_tab(tmp_path):
    widths = {f'robot_state.{arm}.q': 6 for arm in ('left', 'right')}
    widths |= {f'{grip}.{arm}': 1 for grip in (keys.GRIP, keys.TARGET_GRIP) for arm in ('left', 'right')}
    ep = _episode(tmp_path / 'ep', {**widths, 'device.level': 1})

    column = _default_column(ep)

    arms, other = column.contents
    assert list(column.row_shares) == [2, 1]
    assert [arm.name for arm in arms.contents] == ['Left', 'Right']
    assert [[_title(cell) for cell in arm.contents] for arm in arms.contents] == [['Robot State', 'Grip']] * 2
    assert [arm.contents[1].contents for arm in arms.contents] == [
        ['/signals/target_grip.left/**', '/signals/grip.left/**'],
        ['/signals/target_grip.right/**', '/signals/grip.right/**'],
    ]
    assert _title(other) == 'Other'


def test_an_arm_is_found_by_its_state_and_no_other_signal_makes_one():
    names = [keys.JOINTS, 'robot_state.right.ee_pose', 'robot_state.left.dq', keys.ROBOT_STATUS]
    names += ['robot_state.left.status', 'robot_command.mid.pose', 'grip.wrist']

    assert dataset_utils._recorded_arms(names) == [None, 'left', 'right']


def test_the_default_shows_the_signals_of_a_recording_with_no_arm_in_other(tmp_path):
    ep = _episode(tmp_path / 'ep', {'sim_state': _MAX_PLOTTED_WIDTH + 1, 'device.level': 1, 'device.speed': 1})

    (other,) = _default_column(ep).contents

    (group,) = other.contents
    assert group.name == 'Other'
    assert [view.name for view in group.contents] == ['device', 'Not plotted']


def test_the_default_charts_the_progress_marks_and_leaves_them_out_of_other(tmp_path):
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        for i, state in enumerate(_STATES):
            writer.append('progress.state', state, Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
            writer.append('progress.delivered', float(i), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
            writer.append('device.level', float(i), Time(**{RECORDED_TIME: 1_000_000_000 * (i + 1)}))
    ep = DiskEpisode(tmp_path / 'ep')

    progress, other = _default_column(ep).contents

    assert progress.name == 'Progress'
    assert sorted(progress.contents) == ['/signals/progress.delivered/**', '/signals/progress.state/**']
    assert _line_names(progress) == {'/signals/progress.delivered': ['progress.delivered']}
    assert [view.name for view in other.contents[0].contents] == ['device.level']


def test_a_text_signal_holds_its_last_value_to_the_last_sample(tmp_path, monkeypatch):
    sent: dict[str, list[int]] = {}
    monkeypatch.setattr(
        dataset_utils.rr,
        'send_columns',
        lambda path, indexes, columns: sent.__setitem__(path, indexes[0].as_arrow_array().cast(pa.int64()).to_pylist()),
    )
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    ep = _text_episode(tmp_path / 'ep', {'progress.state': ['floating', 'reaching', 'reaching']})

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    assert sent['/text/progress.state'] == [1_000_000_000, 2_000_000_000]
    assert sent['/signals/progress.state'] == [1_000_000_000, 2_000_000_000, 3_000_000_000]


def _timestamps_ns(hz: float, seconds: float) -> np.ndarray:
    step_ns = int(1e9 / hz)
    return np.arange(0, int(seconds * 1e9), step_ns, dtype='int64').astype('datetime64[ns]')


def _thinned(ts: np.ndarray, max_hz: float) -> np.ndarray:
    return ts[_decimation_indices(ts, max_hz)]


def test_a_signal_above_the_cap_is_thinned_to_it():
    thinned = _thinned(_timestamps_ns(hz=300, seconds=10), max_hz=30)

    seconds = (int(thinned[-1]) - int(thinned[0])) / 1e9
    assert 29 <= len(thinned) / seconds <= 31


def test_a_rate_that_is_not_a_whole_multiple_of_the_cap_thins_to_below_it():
    thinned = _thinned(_timestamps_ns(hz=100, seconds=9.99), max_hz=30)

    seconds = (int(thinned[-1]) - int(thinned[0])) / 1e9
    assert (len(thinned) - 1) / seconds <= 30


def test_a_burst_beside_a_gap_thins_to_the_cap():
    burst = _timestamps_ns(hz=100, seconds=1)
    ts = np.concatenate([burst, np.array([int(1.38e9)], dtype='int64').astype('datetime64[ns]')])

    thinned = _thinned(ts, max_hz=30)

    spacing_s = np.diff(thinned).astype('int64') / 1e9
    assert spacing_s.min() >= 1 / 30 * (1 - 1e-5)


def test_a_signal_recorded_at_the_cap_keeps_every_sample():
    ts = _timestamps_ns(hz=30, seconds=10)

    assert len(_thinned(ts, max_hz=30)) == len(ts)


def test_a_rate_cap_that_is_not_a_rate_is_refused():
    with pytest.raises(ValueError):
        _decimation_indices(_timestamps_ns(hz=100, seconds=1), max_hz=-30)


def test_a_signal_below_the_cap_keeps_every_sample():
    ts = _timestamps_ns(hz=10, seconds=10)

    assert len(_thinned(ts, max_hz=30)) == len(ts)
    assert len(_thinned(ts, max_hz=0)) == len(ts)
    assert len(_decimation_indices(ts[:1], max_hz=30)) == 1
    assert len(_decimation_indices(np.array([], dtype='datetime64[ns]'), max_hz=30)) == 0


def test_every_encoded_frame_keeps_its_own_episode_time(monkeypatch):
    times = [i * 33_000_000 for i in range(12)]
    frames = [np.full((64, 64, 3), i * 20 % 256, dtype=np.uint8) for i in range(12)]
    logged = []

    def send_columns(path, indexes, columns):
        logged.append({index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes})

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    signal = DummySignal(list(zip(times, range(12), strict=True)), frames, timelines=(RECORDED_TIME, 'tick'))
    dataset_utils._encode_frames_as_video('/video', signal, max_resolution=640, max_hz=0)
    dataset_utils._encode_frames_as_video(
        '/other', DummySignal([7], frames[:1], timelines=('device',)), max_resolution=640, max_hz=0
    )

    assert logged == [{RECORDED_TIME: [ts], 'tick': [i]} for i, ts in enumerate(times)] + [{'device': [7]}]


def test_frames_past_the_rate_cap_are_left_out_of_the_encoding(monkeypatch):
    times = [i * 10_000_000 for i in range(12)]
    frames = [np.full((64, 64, 3), i * 20 % 256, dtype=np.uint8) for i in range(12)]
    logged: list[int] = []
    monkeypatch.setattr(
        dataset_utils.rr,
        'send_columns',
        lambda path, indexes, columns: logged.extend(indexes[0].as_arrow_array().cast(pa.int64()).to_pylist()),
    )
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)

    dataset_utils._encode_frames_as_video('/video', DummySignal(times, frames), max_resolution=640, max_hz=30)

    kept = _decimation_indices(np.asarray(times, dtype='datetime64[ns]'), max_hz=30)
    assert 1 < len(kept) < len(times)
    assert logged == [times[index] for index in kept]


def _write_mp4(path: Path, width: int, height: int, frames: int) -> Path:
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        for i in range(frames):
            picture = np.full((height, width, 3), i * 8 % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')))
        container.mux(stream.encode())
    return path


def _frame_times(data: bytes) -> list[float]:
    with av.open(io.BytesIO(data), 'r') as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        return [float(frame.pts * time_base) for frame in container.decode(stream) if frame.pts is not None]


def test_a_frame_size_within_the_cap_is_left_alone():
    assert _size_capped_to(320, 240, 640) == (320, 240)
    assert _size_capped_to(640, 480, 640) == (640, 480)


def test_an_odd_frame_side_within_the_cap_is_still_evened():
    assert _size_capped_to(301, 240, 640) == (300, 240)
    assert _size_capped_to(320, 241, 640) == (320, 240)


def test_a_frame_size_above_the_cap_fits_it_on_even_sides():
    assert _size_capped_to(1280, 720, 640) == (640, 360)
    assert _size_capped_to(720, 1280, 640) == (360, 640)
    assert _size_capped_to(1000, 999, 640) == (640, 638)


def test_a_cap_an_encoder_cannot_carry_is_refused():
    assert _size_capped_to(1280, 720, 2) == (2, 2)
    for cap in (1, 0, -640):
        with pytest.raises(ValueError):
            _size_capped_to(1280, 720, cap)


def test_a_video_within_the_cap_is_embedded_as_recorded(tmp_path):
    src = _write_mp4(tmp_path / 'small.mp4', width=320, height=240, frames=12)

    assert _mp4_reduced_to(src, max_resolution=640) == src.read_bytes()


def test_a_video_keeps_the_frames_named_at_their_own_times(tmp_path):
    src = _write_mp4(tmp_path / 'small.mp4', width=320, height=240, frames=12)
    kept = np.array([0, 3, 6, 9])

    thinned = _mp4_reduced_to(src, max_resolution=640, kept=kept)

    source_times = _frame_times(src.read_bytes())
    assert _frame_times(thinned) == pytest.approx([source_times[index] for index in kept], abs=1e-4)
    assert thinned != src.read_bytes()


def test_a_larger_video_is_re_encoded_frame_for_frame(tmp_path):
    src = _write_mp4(tmp_path / 'big.mp4', width=1280, height=720, frames=12)

    downscaled = _mp4_reduced_to(src, max_resolution=640)

    with av.open(io.BytesIO(downscaled), 'r') as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (640, 360)
    assert len(downscaled) < len(src.read_bytes())
    assert _frame_times(downscaled) == pytest.approx(_frame_times(src.read_bytes()), abs=1e-4)
