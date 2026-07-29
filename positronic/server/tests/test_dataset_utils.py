import io
from pathlib import Path

import av
import numpy as np
import pytest

from positronic.server import positronic_server
from positronic.server.dataset_utils import RRD_FORMAT_VERSION, playable_video_bytes


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


def test_rrd_cache_path_versions_and_drops_stale(tmp_path, monkeypatch):
    """A generation bump changes the cache key and removes the previous generation's entry
    (regression: RRDs cached before the B-frame re-encode were served stale forever)."""
    class _Ep:
        meta = {'uid': 'abc123'}

    monkeypatch.setitem(positronic_server.app_state, 'dataset', {0: _Ep()})
    monkeypatch.setitem(positronic_server.app_state, 'cache_dir', str(tmp_path))
    monkeypatch.setitem(positronic_server.app_state, 'root', str(tmp_path / 'ds'))

    path = positronic_server._get_rrd_cache_path(0)
    assert f'.v{RRD_FORMAT_VERSION}.rrd' in path

    old = Path(path).parent / 'abc123.rrd'
    old.write_bytes(b'stale')
    assert positronic_server._get_rrd_cache_path(0) == path
    assert not old.exists()


def test_cache_while_streaming_commits_only_complete_streams(tmp_path):
    """A failed RRD stream leaves no cache entry (regression: a mid-build crash persisted an empty
    file that was then served as a valid cache hit), and a complete stream commits atomically."""
    cache_path = str(tmp_path / 'ep.rrd')

    def failing():
        yield b'partial'
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        list(positronic_server._cache_while_streaming(failing(), cache_path))
    assert list(tmp_path.iterdir()) == []

    assert list(positronic_server._cache_while_streaming(iter([b'a', b'b']), cache_path)) == [b'a', b'b']
    assert Path(cache_path).read_bytes() == b'ab'
