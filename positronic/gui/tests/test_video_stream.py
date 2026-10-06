import io
import queue

import av
import numpy as np

from positronic.gui.video_stream import VideoStream, codec_string


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
