import numpy as np
import pytest

from positronic.dataset.video import VideoSignal, VideoSignalWriter
from positronic.offboard.frame_codec_cost import H264, Jpeg, bounded_frames, costs, sampled_span
from positronic.policy.codec import RestrictImageSize

SEC = 1_000_000_000
PERIOD_30HZ = SEC // 30


def _camera(tmp_path, name: str, first_ts: int, count: int) -> VideoSignal:
    """``count`` flat frames at 30 Hz from ``first_ts``, frame ``i`` filled with ``10 * i``."""
    video, index = tmp_path / f'{name}.mp4', tmp_path / f'{name}.frames.parquet'
    with VideoSignalWriter(video, index) as writer:
        for i in range(count):
            writer.append(np.full((16, 16, 3), 10 * i, dtype=np.uint8), first_ts + i * PERIOD_30HZ)
    return VideoSignal(video, index)


def _frame_ids(frames: list[np.ndarray]) -> list[int]:
    return [round(float(np.median(frame)) / 10) for frame in frames]


def test_sampling_follows_the_recorded_timestamps_not_the_video_rate(tmp_path):
    camera = _camera(tmp_path, 'cam', SEC, 8)
    frames = bounded_frames(camera, sampled_span([camera]), 15.0, RestrictImageSize(64, 64), 0)
    assert _frame_ids(frames) == [0, 2, 4, 6]


def test_every_camera_samples_one_grid_over_the_span_all_cover(tmp_path):
    long = _camera(tmp_path, 'long', SEC, 12)
    short = _camera(tmp_path, 'short', SEC + PERIOD_30HZ, 6)
    span = sampled_span([long, short])
    bound = RestrictImageSize(64, 64)
    assert _frame_ids(bounded_frames(long, span, 15.0, bound, 0)) == [1, 3, 5]
    assert _frame_ids(bounded_frames(short, span, 15.0, bound, 0)) == [0, 2, 4]


def test_count_caps_the_frames_decoded(tmp_path):
    camera = _camera(tmp_path, 'cam', SEC, 8)
    frames = bounded_frames(camera, sampled_span([camera]), 15.0, RestrictImageSize(64, 64), 3)
    assert _frame_ids(frames) == [0, 2, 4]


def test_costs_refuses_fewer_frames_than_one_window(tmp_path):
    camera = _camera(tmp_path, 'cam', SEC, 4)
    frames = bounded_frames(camera, sampled_span([camera]), 15.0, RestrictImageSize(64, 64), 0)
    with pytest.raises(ValueError, match='cannot fill one 4-frame window'):
        costs(frames, 'cam', 4, [Jpeg()])


def test_h264_encodes_a_window_with_odd_sides():
    size, encode_ms, decode_ms = H264('ultrafast', 20).cost(np.zeros((3, 15, 17, 3), dtype=np.uint8))
    assert size > 0 and encode_ms > 0 and decode_ms > 0
