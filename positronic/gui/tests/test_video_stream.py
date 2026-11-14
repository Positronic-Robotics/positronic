import io
import queue

import av
import numpy as np

from positronic.gui.video_stream import AnnexBEncoder, VideoStream, codec_string


def _frame(i: int, height: int = 480, width: int = 640) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : (i * 37) % width] = (i * 11) % 256
    return frame


def _fragments(subscriber: queue.Queue[bytes]) -> list[bytes]:
    fragments = []
    while not subscriber.empty():
        fragments.append(subscriber.get_nowait())
    return fragments


def test_a_subscriber_gets_fragments_that_play_after_the_init_segment():
    stream = VideoStream(fps=15, width=320, keyframe_interval=4, bitrate=500_000)
    subscriber = stream.subscribe()
    for i in range(20):
        stream.push(_frame(i))
    stream.close()

    fragments = _fragments(subscriber)
    assert fragments and all(fragment[4:8] == b'moof' for fragment in fragments)
    assert codec_string(stream.init_segment).startswith('avc1.42')  # the baseline profile
    with av.open(io.BytesIO(stream.init_segment + b''.join(fragments)), mode='r') as container:
        decoded = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    assert len(decoded) == 20
    assert decoded[0].shape == (240, 320, 3)


def test_a_late_subscriber_plays_from_the_init_segment_and_its_first_fragment():
    stream = VideoStream(fps=15, width=320, keyframe_interval=4, bitrate=500_000)
    for i in range(10):
        stream.push(_frame(i))
    late = stream.subscribe()
    for i in range(10, 20):
        stream.push(_frame(i))
    stream.close()

    with av.open(io.BytesIO(stream.init_segment + b''.join(_fragments(late))), mode='r') as container:
        assert sum(1 for _ in container.decode(video=0)) > 0


def test_the_init_segment_is_empty_until_the_first_fragment():
    stream = VideoStream(fps=15, width=320, keyframe_interval=30, bitrate=500_000)
    stream.push(_frame(0))
    assert stream.init_segment == b''
    stream.close()


def test_a_subscriber_that_falls_behind_loses_its_oldest_fragments():
    stream = VideoStream(fps=2, width=320, keyframe_interval=1, bitrate=500_000)
    subscriber = stream.subscribe()
    for i in range(10):
        stream.push(_frame(i))
    assert subscriber.qsize() == 2
    stream.close()


def _decoded(units: list[bytes]) -> list[np.ndarray]:
    decoder = av.CodecContext.create('h264', 'r')
    return [frame.to_ndarray(format='rgb24') for unit in units for frame in decoder.decode(av.Packet(unit))]


def test_the_annex_b_stream_decodes_to_the_colors_of_its_frames():
    encoder = AnnexBEncoder(fps=30, keyframe_interval=30, bitrate=2_000_000)
    red, blue = np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64, 3), np.uint8)
    red[..., 0], blue[..., 2] = 255, 255
    units = [unit for frame in (red, blue) for unit in encoder.encode(frame)]
    assert len(units) == 2, 'each frame gives its access unit at once'
    first, second = _decoded([unit.data for unit in units])
    np.testing.assert_allclose(first.mean(axis=(0, 1)), [255, 0, 0], atol=12)
    np.testing.assert_allclose(second.mean(axis=(0, 1)), [0, 0, 255], atol=12)


def test_only_a_key_frame_carries_the_decoder_config():
    encoder = AnnexBEncoder(fps=30, keyframe_interval=3, bitrate=2_000_000)
    units = [unit for i in range(4) for unit in encoder.encode(_frame(i, 48, 64))]
    assert [unit.config is not None for unit in units] == [True, False, False, True]
    config = units[0].config
    assert config is not None and (config.width, config.height) == (64, 48)
    assert config.codec.startswith('avc1.42C0'), 'the constrained baseline profile, at the level of the SPS'


def test_a_frame_of_a_new_size_starts_a_new_stream_at_a_key_frame():
    encoder = AnnexBEncoder(fps=30, keyframe_interval=30, bitrate=2_000_000)
    encoder.encode(_frame(0, 48, 64))
    encoder.encode(_frame(1, 48, 64))
    (unit,) = encoder.encode(_frame(2, 96, 128))
    assert unit.config is not None and (unit.config.width, unit.config.height) == (128, 96)
    (decoded,) = _decoded([unit.data])
    assert decoded.shape == (96, 128, 3)


def test_a_frame_with_an_odd_side_is_encoded_one_pixel_larger():
    (unit,) = AnnexBEncoder(fps=30, keyframe_interval=30, bitrate=2_000_000).encode(_frame(0, 47, 63))
    assert unit.config is not None and (unit.config.width, unit.config.height) == (64, 48)
