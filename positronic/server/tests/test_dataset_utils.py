import io
import os
import time
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


def test_playable_video_bytes_preserves_seek_points(tmp_path):
    """The re-encode keeps the source's keyframes (regression: an encoder-default keyframe interval
    longer than the clip left one seek point, so the viewer could not seek at all)."""
    src = tmp_path / 'bframes.mp4'
    _write_video(src, n_frames=90, max_b_frames=None)
    if int(av.video.frame.PictureType.B) not in _pict_types(str(src)):
        pytest.skip('host encoder emits no B-frames; nothing to re-encode')

    with av.open(str(src)) as container:
        src_keys = sum(1 for frame in container.decode(video=0) if frame.key_frame)

    with av.open(io.BytesIO(playable_video_bytes(src))) as container:
        out_keys = sum(1 for frame in container.decode(video=0) if frame.key_frame)
    assert out_keys == src_keys


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
    positronic_server._drop_unservable_cache_entries(str(Path(path).parent))
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


def test_rrd_cache_sweep_keeps_current_generation_entries(tmp_path, monkeypatch):
    """The sweep drops what the current generation cannot serve and keeps what it can: another
    episode's current entry and a partial a builder is still streaming into both survive."""
    class _Ep:
        def __init__(self, uid):
            self.meta = {'uid': uid}

    # 'foo.victim' extends 'foo' past a dot, the separator before '.v<N>.rrd' (regression: the sweep
    # for 'foo' unlinked the sibling's entry).
    episodes = {0: _Ep('foo'), 1: _Ep('foo.victim')}
    monkeypatch.setitem(positronic_server.app_state, 'dataset', episodes)
    monkeypatch.setitem(positronic_server.app_state, 'cache_dir', str(tmp_path))
    monkeypatch.setitem(positronic_server.app_state, 'root', str(tmp_path / 'ds'))

    path = Path(positronic_server._get_rrd_cache_path(0))
    sibling = Path(positronic_server._get_rrd_cache_path(1))
    assert sibling != path
    sibling.write_bytes(b'other episode')
    previous_generation = path.parent / 'foo.rrd'
    previous_generation.write_bytes(b'stale')
    live_partial = path.parent / f'{path.name}.deadbeef.partial'
    live_partial.write_bytes(b'streaming right now')
    abandoned = path.parent / f'{path.name}.cafe.partial'
    abandoned.write_bytes(b'crashed builder')
    old_ts = time.time() - 2 * positronic_server._ABANDONED_PARTIAL_AGE_S
    os.utime(abandoned, (old_ts, old_ts))

    positronic_server._drop_unservable_cache_entries(str(path.parent))
    assert sibling.exists()
    assert live_partial.exists()
    assert not previous_generation.exists()
    assert not abandoned.exists()


@pytest.mark.parametrize('uid', ['../shared', 'weird[uid]', 'foo.victim', 'é' * 100])
def test_rrd_cache_key_bounds_hostile_uids(tmp_path, monkeypatch, uid):
    """Uids reach the writer unvalidated, so every one of them has to key a single bounded filename
    inside the cache directory (regressions: '../shared' traversed out of it, 'weird[uid]' globbed a
    sibling away, 'foo.victim' shared a sweep prefix with 'foo', and percent-encoding a long
    non-ASCII uid overflowed the 255-byte filename component limit)."""
    class _Ep:
        meta = {'uid': uid}

    cache_root = tmp_path / 'cache'
    monkeypatch.setitem(positronic_server.app_state, 'dataset', {0: _Ep()})
    monkeypatch.setitem(positronic_server.app_state, 'cache_dir', str(cache_root))
    monkeypatch.setitem(positronic_server.app_state, 'root', str(tmp_path / 'ds'))

    outside = cache_root / f'shared.v{RRD_FORMAT_VERSION}.rrd'
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b'foreign dataset entry')

    path = Path(positronic_server._get_rrd_cache_path(0))
    assert path.parent.parent == cache_root
    assert path.resolve().is_relative_to(cache_root.resolve())
    assert len(path.name.encode()) <= 255
    assert outside.exists()


def test_cache_while_streaming_concurrent_builders(tmp_path):
    """Two builders for the same episode never corrupt each other's partials; the committed
    file is one complete stream."""
    cache_path = str(tmp_path / 'ep.rrd')
    a = positronic_server._cache_while_streaming(iter([b'a1', b'a2']), cache_path)
    b = positronic_server._cache_while_streaming(iter([b'b1', b'b2']), cache_path)
    assert next(a) == b'a1'
    assert next(b) == b'b1'
    assert list(a) == [b'a2']
    assert list(b) == [b'b2']
    assert Path(cache_path).read_bytes() in (b'a1a2', b'b1b2')
    assert [p.name for p in tmp_path.iterdir()] == ['ep.rrd']
