import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.server import dataset_utils
from positronic.server.dataset_utils import (
    _MAX_PLOTTED_WIDTH,
    _collect_signal_groups,
    _decimation_indices,
    _mp4_downscaled_to,
    _size_capped_to,
    _unplotted_notice,
    _write_urdf_to_dir,
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
    assert signals.unplotted == {'wide_signal': width}


def test_wide_signal_still_reaches_the_3d_view(tmp_path):
    width = _MAX_PLOTTED_WIDTH + 1
    signals = _collect_signal_groups(_episode(tmp_path / 'ep', {keys.JOINTS: width}))

    assert signals.numerics == [keys.JOINTS]
    assert signals.dims == {keys.JOINTS: width}


def test_every_joint_signal_the_episode_records_is_collected(tmp_path):
    widths = {'robot_state.left.q': 6, 'robot_state.right.q': 6, keys.GRIP: 1}
    static = {keys.JOINT_SIGNALS: ['robot_state.left.q', 'robot_state.right.q', 'robot_state.absent.q']}

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
    notice = _unplotted_notice({'wide_signal': 866, 'wider_signal': 120})

    assert '`wide_signal` — 866 values' in notice
    assert '`wider_signal` — 120 values' in notice


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


def test_every_encoded_frame_keeps_its_own_episode_time(monkeypatch):
    times = [i * 33_000_000 for i in range(12)]
    frames = [np.full((64, 64, 3), i * 20 % 256, dtype=np.uint8) for i in range(12)]
    logged: list[int] = []
    monkeypatch.setattr(dataset_utils, 'set_timeline_time', lambda _timeline, ts: logged.append(ts))
    monkeypatch.setattr(dataset_utils.rr, 'log', lambda *args, **kwargs: None)

    dataset_utils._encode_frames_as_video('/video', _RawFrameSignal(frames, times), max_resolution=640)

    assert logged == times


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

    assert _mp4_downscaled_to(src, max_resolution=640) == src.read_bytes()


def test_a_larger_video_is_re_encoded_frame_for_frame(tmp_path):
    src = _write_mp4(tmp_path / 'big.mp4', width=1280, height=720, frames=12)

    downscaled = _mp4_downscaled_to(src, max_resolution=640)

    with av.open(io.BytesIO(downscaled), 'r') as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (640, 360)
    assert len(downscaled) < len(src.read_bytes())
    assert _frame_times(downscaled) == pytest.approx(_frame_times(src.read_bytes()), abs=1e-4)
