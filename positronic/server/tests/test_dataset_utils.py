import io

import av
import numpy as np
import pytest

from positronic.server.dataset_utils import playable_video_bytes


def _write_video(path, n_frames, max_b_frames):
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('h264', rate=30)
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = 'yuv420p'
        if max_b_frames is not None:
            stream.max_b_frames = max_b_frames
        for i in range(n_frames):
            img = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _pict_types(data):
    with av.open(data) as container:
        return [int(frame.pict_type) for frame in container.decode(video=0)]


def test_playable_video_bytes_strips_b_frames(tmp_path):
    """B-frame recordings are re-encoded for the viewer (regression: rerun's web viewer stalls on them)."""
    src = tmp_path / 'bframes.mp4'
    _write_video(src, n_frames=60, max_b_frames=None)
    b = int(av.video.frame.PictureType.B)
    src_types = _pict_types(str(src))
    if b not in src_types:
        pytest.skip("host encoder emits no B-frames; nothing to strip")

    out = playable_video_bytes(src)
    out_types = _pict_types(io.BytesIO(out))
    assert b not in out_types
    assert len(out_types) == len(src_types)


def test_playable_video_bytes_passthrough(tmp_path):
    """A B-frame-free recording is served byte-identical."""
    src = tmp_path / 'plain.mp4'
    _write_video(src, n_frames=20, max_b_frames=0)
    assert playable_video_bytes(src) == src.read_bytes()
