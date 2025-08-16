import numpy as np
import pytest

from positronic.dataset.vector import SimpleSignal, SimpleSignalWriter
from positronic.dataset.video import VideoSignal, VideoSignalWriter


def test_vector_start_last_ts_basic(tmp_path):
    fp = tmp_path / "sig.parquet"
    w = SimpleSignalWriter(fp)
    w.append(1, 1000)
    w.append(2, 2000)
    w.append(3, 3000)
    w.finish()

    s = SimpleSignal(fp)
    assert s.start_ts == 1000
    assert s.last_ts == 3000


def test_vector_start_last_ts_empty_raises(tmp_path):
    fp = tmp_path / "empty.parquet"
    w = SimpleSignalWriter(fp)
    w.finish()
    s = SimpleSignal(fp)
    with pytest.raises(ValueError):
        _ = s.start_ts
    with pytest.raises(ValueError):
        _ = s.last_ts


def create_frame(value=0, shape=(16, 16, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_video_start_last_ts_basic(tmp_path):
    video = tmp_path / "test.mp4"
    frames_idx = tmp_path / "frames.parquet"
    w = VideoSignalWriter(video, frames_idx, gop_size=5, fps=30)
    w.append(create_frame(10), 1000)
    w.append(create_frame(20), 2000)
    w.append(create_frame(30), 4000)
    w.finish()

    s = VideoSignal(video, frames_idx)
    assert s.start_ts == 1000
    assert s.last_ts == 4000


def test_video_start_last_ts_empty_raises(tmp_path):
    video = tmp_path / "empty.mp4"
    frames_idx = tmp_path / "frames.parquet"
    w = VideoSignalWriter(video, frames_idx)
    w.finish()
    s = VideoSignal(video, frames_idx)
    with pytest.raises(ValueError):
        _ = s.start_ts
    with pytest.raises(ValueError):
        _ = s.last_ts

