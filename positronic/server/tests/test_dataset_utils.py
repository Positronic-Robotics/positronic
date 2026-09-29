import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun.blueprint as rrb
import rerun.recording as rr_recording

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter, LocalDataset, LocalDatasetWriter
from positronic.eval import keys as eval_keys
from positronic.server import dataset_utils
from positronic.server.dataset_utils import (
    _MAX_PLOTTED_WIDTH,
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
            writer.append(name, np.zeros(width, dtype=np.float32), 1000)
            writer.append(name, np.ones(width, dtype=np.float32), 2000)
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
                writer.append(name, value, 1_000_000_000 * (i + 1))
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
            writer.append('progress.state', state, 1_000_000_000 * (i + 1))
    rrd = tmp_path / 'ep.rrd'
    rrd.write_bytes(b''.join(stream_episode_rrd(LocalDataset(root), 0)))

    columns = rr_recording.load_recording(str(rrd)).schema().component_columns()
    archetypes = {(column.entity_path, column.archetype) for column in columns}
    assert ('/signals/progress.state', 'rerun.archetypes.Scalars') in archetypes
    assert ('/text/progress.state', 'rerun.archetypes.TextLog') in archetypes


def _null_drainer() -> dataset_utils._BinaryStreamDrainer:
    return dataset_utils._BinaryStreamDrainer(dataset_utils.rr.RecordingStream('test').binary_stream(), min_bytes=1)


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


def test_a_text_log_entry_carries_its_time_from_the_start_of_the_recording(tmp_path, monkeypatch):
    sent: dict[str, dict[str, list[int]]] = {}

    def send_columns(path, indexes, columns):
        sent[path] = {index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes}

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    machine_clock = 1_011_234_567_890_123  # nanoseconds since boot, far from the epoch
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        writer.append('robot.q', np.zeros(2), machine_clock)
        writer.append('progress.state', 'floating', machine_clock + 1_503_456_789)
        writer.append('progress.state', 'reaching', machine_clock + 62_250_000_000)
    ep = DiskEpisode(tmp_path / 'ep')

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    assert sent['/text/progress.state'][dataset_utils._TIME_FROM_START] == [1_500_000_000, 62_250_000_000]


def test_a_signal_the_recording_leaves_out_does_not_move_the_text_log_origin(tmp_path, monkeypatch):
    sent: dict[str, dict[str, list[int]]] = {}

    def send_columns(path, indexes, columns):
        sent[path] = {index.timeline_name(): index.as_arrow_array().cast(pa.int64()).to_pylist() for index in indexes}

    monkeypatch.setattr(dataset_utils.rr, 'send_columns', send_columns)
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        writer.append('words', np.array(['a', 'b']), 1_000_000_000)
        writer.append('robot.q', np.zeros(2), 3_000_000_000)
        writer.append('progress.state', 'floating', 4_000_000_000)
    ep = DiskEpisode(tmp_path / 'ep')

    list(dataset_utils._log_text_signals(ep, _collect_signal_groups(ep), _null_drainer()))

    assert sent['/text/progress.state'][dataset_utils._TIME_FROM_START] == [1_000_000_000]


def test_a_text_log_shows_its_time_from_the_start_and_not_the_clock_time():
    columns = dataset_utils._text_log_view('progress.state').properties['TextLogColumns']
    assert isinstance(columns, rrb.TextLogColumns) and columns.timeline_columns is not None
    timelines = columns.timeline_columns.as_arrow_array().to_pylist()

    shown = [column['timeline'] for column in timelines if column['visible']]
    assert shown == [dataset_utils._TIME_FROM_START]


def _signals_with_cameras(aspects: list[float], with_3d: bool) -> dataset_utils.EpisodeSignals:
    cameras = {f'camera_{i}': aspect for i, aspect in enumerate(aspects)}
    poses = ['pose'] if with_3d else []
    return dataset_utils.EpisodeSignals(
        videos=list(cameras), numerics=[], dims={}, poses=poses, joints=[], camera_aspects=cameras
    )


@pytest.mark.parametrize('aspects', [[16 / 9] * 3, [4 / 3] * 3, [16 / 9] * 4, [4 / 3, 16 / 9, 16 / 9]])
@pytest.mark.parametrize('with_3d', [True, False])
def test_the_camera_row_is_as_tall_as_its_frames(aspects, with_3d):
    share = dataset_utils._camera_row_share(_signals_with_cameras(aspects, with_3d))

    row_width = dataset_utils._VIEWER_ASPECT * (0.75 if with_3d else 1.0)
    for aspect in aspects:
        assert row_width * aspect / sum(aspects) / share == pytest.approx(aspect)


def test_one_camera_leaves_the_signals_a_quarter_of_the_height():
    assert dataset_utils._camera_row_share(_signals_with_cameras([16 / 9], with_3d=False)) == 0.75


def test_eight_signal_cells_under_three_cameras_wrap_to_two_rows_of_four():
    share = dataset_utils._camera_row_share(_signals_with_cameras([16 / 9] * 3, with_3d=True))

    assert dataset_utils._series_columns(8, 1 - share) == 4


def _tabs(container: Any) -> list[rrb.Tabs]:
    if isinstance(container, rrb.Tabs):
        return [container]
    return [tabs for child in getattr(container, 'contents', None) or [] for tabs in _tabs(child)]


def test_a_tab_group_opens_on_its_text_signal(tmp_path):
    with DiskEpisodeWriter(tmp_path / 'ep') as writer:
        for i, state in enumerate(_STATES):
            writer.append('progress.delivered', float(i), 1_000_000_000 * (i + 1))
            writer.append('progress.state', state, 1_000_000_000 * (i + 1))
            writer.append('robot.q', np.zeros(2), 1_000_000_000 * (i + 1))
            writer.append('robot.dq', np.zeros(2), 1_000_000_000 * (i + 1))
    ep = DiskEpisode(tmp_path / 'ep')

    tabs = {tab.name: tab.active_tab for tab in _tabs(_build_blueprint(_collect_signal_groups(ep), ep).root_container)}

    assert tabs == {'progress': 1, 'robot': None}


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


class _RawFrameSignal:
    def __init__(self, frames: list[np.ndarray], times: list[int]):
        self._frames, self._times = frames, times

    def __getitem__(self, index):
        return self._frames[index], self._times[index]

    def __iter__(self):
        return iter(zip(self._frames, self._times, strict=True))

    def keys(self):
        return np.asarray(self._times, dtype=np.int64)


def test_every_encoded_frame_keeps_its_own_episode_time(monkeypatch):
    times = [i * 33_000_000 for i in range(12)]
    frames = [np.full((64, 64, 3), i * 20 % 256, dtype=np.uint8) for i in range(12)]
    logged: list[int] = []
    monkeypatch.setattr(dataset_utils, 'set_timeline_time', lambda _timeline, ts: logged.append(ts))
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)

    dataset_utils._encode_frames_as_video('/video', _RawFrameSignal(frames, times), max_resolution=640, max_hz=0)

    assert logged == times


def test_frames_past_the_rate_cap_are_left_out_of_the_encoding(monkeypatch):
    times = [i * 10_000_000 for i in range(12)]
    frames = [np.full((64, 64, 3), i * 20 % 256, dtype=np.uint8) for i in range(12)]
    logged: list[int] = []
    monkeypatch.setattr(dataset_utils, 'set_timeline_time', lambda _timeline, ts: logged.append(ts))
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)

    dataset_utils._encode_frames_as_video('/video', _RawFrameSignal(frames, times), max_resolution=640, max_hz=30)

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
