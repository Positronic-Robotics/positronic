import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from positronic.dataset import Episode
from positronic.dataset.local_dataset import UNFINISHED_MARKER, DiskEpisode, DiskEpisodeWriter
from positronic.dataset.tests.test_video import assert_frames_equal, create_frame


def test_episode_writer_and_reader_basic(tmp_path):
    ep_dir = tmp_path / 'ep1'
    with DiskEpisodeWriter(ep_dir) as w:
        w.append('a', 1, 1000)
        w.append('a', 2, 2000)
        w.append('b', 10, 1500)
        w.append('b', 20, 2500)

    # Files written
    assert (ep_dir / 'a.parquet').exists()
    assert (ep_dir / 'b.parquet').exists()

    ep = DiskEpisode(ep_dir)
    # Keys present
    assert set(ep.keys()) == {'a', 'b'}

    # Access signals and basic data
    a = ep['a']
    b = ep['b']
    assert len(a) == 2
    assert len(b) == 2
    assert a[0] == (1, 1000)
    assert b[1] == (20, 2500)


def test_episode_start_last_ts(tmp_path):
    ep_dir = tmp_path / 'ep2'
    with DiskEpisodeWriter(ep_dir) as w:
        # a: starts 1000, last 2000
        w.append('a', 1, 1000)
        w.append('a', 2, 2000)
        # b: starts 1500, last 2500
        w.append('b', 10, 1500)
        w.append('b', 20, 2500)

    ep = DiskEpisode(ep_dir)
    # Current Episode implementation uses max of starts and max of lasts
    assert ep.start_ts == 1500
    assert ep.last_ts == 2500


def test_episode_getitem_returns_signal(tmp_path):
    ep_dir = tmp_path / 'ep3'
    with DiskEpisodeWriter(ep_dir) as w:
        w.append('x', np.array([1, 2]), 1000)
        w.append('x', np.array([3, 4]), 2000)

    ep = DiskEpisode(ep_dir)
    x = ep['x']
    assert len(x) == 2
    v0, t0 = x[0]
    v1, t1 = x[1]
    assert t0 == 1000 and t1 == 2000
    # vector contents preserved
    assert list(v0) == [1, 2]
    assert list(v1) == [3, 4]


def test_episode_static_items_json(tmp_path):
    ep_dir = tmp_path / 'ep_static'
    with DiskEpisodeWriter(ep_dir) as w:
        # write static metadata (single file static.json)
        w.set_static('task', 'pick_place')
        w.set_static('version', 1)
        w.set_static('params', {'speed': 0.5})
        w.set_static('tags', ['demo', 'test'])
        # also write a dynamic signal
        w.append('a', 42, 1000)
        w.append('a', 43, 2000)

    # Files written
    assert (ep_dir / 'static.json').exists()
    assert (ep_dir / 'a.parquet').exists()

    ep = DiskEpisode(ep_dir)
    # keys include both static and dynamic names
    assert set(ep.keys()) == {'task', 'version', 'params', 'tags', 'a'}

    # static access returns the value directly
    assert ep['task'] == 'pick_place'
    assert ep['version'] == 1
    assert ep['params'] == {'speed': 0.5}
    assert ep['tags'] == ['demo', 'test']

    # dynamic access still returns a Signal
    a = ep['a']
    assert len(a) == 2
    assert a[0] == (42, 1000)


def test_episode_meta_written_and_exposed(tmp_path):
    ep_dir = tmp_path / 'ep_meta'
    with DiskEpisodeWriter(ep_dir) as w:
        # also write a dynamic signal and static
        w.append('a', 1, 1000)
        w.set_static('user_key', 'value')

    # Files present
    assert (ep_dir / 'static.json').exists()
    assert (ep_dir / 'meta.json').exists()

    ep = DiskEpisode(ep_dir)
    m = ep.meta
    assert isinstance(m, dict)
    assert 'schema_version' in m and m['schema_version'] == 1
    assert 'created_ts_ns' in m and isinstance(m['created_ts_ns'], int)
    assert 'writer' in m and isinstance(m['writer'], dict)
    assert m['writer'].get('name') == 'positronic.dataset.local_dataset.DiskEpisodeWriter'
    expected_path = str(ep_dir.expanduser().absolute())
    assert m.get('path') == expected_path
    assert 'size_mb' in m and isinstance(m['size_mb'], float)
    assert m['size_mb'] > 0
    formatted_size = f'{m["size_mb"]:.2f}'
    assert isinstance(formatted_size, str)
    assert formatted_size.replace('.', '', 1).isdigit()
    # git info present when running inside a git repo; skip strict assertions otherwise
    if 'git' in m['writer']:
        git = m['writer']['git']
        assert isinstance(git, dict)
        assert {'commit', 'branch', 'dirty'}.issubset(git.keys())


def test_episode_writer_marks_unfinished_and_clears_on_close(tmp_path):
    ep_dir = tmp_path / 'ep_unfinished'
    with DiskEpisodeWriter(ep_dir) as w:
        marker = ep_dir / UNFINISHED_MARKER
        assert marker.exists()
        w.set_static('id', 1)

    assert not (ep_dir / UNFINISHED_MARKER).exists()


def test_episode_reader_rejects_unfinished(tmp_path):
    ep_dir = tmp_path / 'ep_incomplete'
    ep_dir.mkdir()
    (ep_dir / UNFINISHED_MARKER).write_text('unfinished', encoding='utf-8')
    with pytest.raises(ValueError, match='unfinished episode'):
        DiskEpisode(ep_dir)


def test_episode_static_numpy_arrays_rejected(tmp_path):
    ep_dir = tmp_path / 'ep_static_np'
    with DiskEpisodeWriter(ep_dir) as w:
        arr_i32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        arr_f32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        nested = {'cam': {'K': arr_f32.reshape(3, 1), 'shape': [480, 640]}, 'list': [arr_i32]}
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('arr_i32', arr_i32)
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('arr_f32', arr_f32)
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('nested', nested)


def test_episode_static_accepts_valid_json_structures(tmp_path):
    ep_dir = tmp_path / 'ep_static_json_ok'
    payload = {
        'task': 'stack',
        'ok': True,
        'version': 2,
        'params': {'k': 1, 'names': ['a', 'b'], 'thresholds': [0.1, 0.2]},
        'nested': [{'v': 1}, {'v': 2, 'flag': False}],
    }
    with DiskEpisodeWriter(ep_dir) as w:
        for k, v in payload.items():
            w.set_static(k, v)

    ep = DiskEpisode(ep_dir)
    for k, v in payload.items():
        assert ep[k] == v


def test_episode_static_rejects_non_string_keys(tmp_path):
    ep_dir = tmp_path / 'ep_static_bad_key'
    with DiskEpisodeWriter(ep_dir) as w:
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('bad', {1: 'a'})


def test_episode_static_accepts_tuple_but_rejects_set(tmp_path):
    ep_dir = tmp_path / 'ep_static_bad_types'
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_static('coords', (1, 2))
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('labels', {'a', 'b'})

    ep = DiskEpisode(ep_dir)
    assert ep['coords'] == [1, 2]


def test_episode_static_rejects_none(tmp_path):
    ep_dir = tmp_path / 'ep_static_none'
    with DiskEpisodeWriter(ep_dir) as w:
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('maybe', None)


def test_episode_writer_abort_cleans_up_and_blocks_further_use(tmp_path):
    ep_dir = tmp_path / 'ep_abort'
    with DiskEpisodeWriter(ep_dir) as w:
        # Append some data to create resources
        w.append('a', 1, 1000)
        w.set_static('k', 1)
        assert ep_dir.exists()

        # Abort should remove the directory and prevent further actions
        w.abort()
        assert not ep_dir.exists()

        with pytest.raises(RuntimeError):
            w.append('a', 2, 2000)
        with pytest.raises(RuntimeError):
            w.set_static('z', 2)


def test_episode_writer_context_aborts_on_exception(tmp_path):
    ep_dir = tmp_path / 'ep_context_abort'
    with pytest.raises(RuntimeError, match='boom'):
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            raise RuntimeError('boom')
    assert not ep_dir.exists()


def test_episode_writer_set_static_twice_raises(tmp_path):
    ep_dir = tmp_path / 'ep_static_dup'
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_static('info', {'ok': True})
        with np.testing.assert_raises_regex(ValueError, 'already set'):
            w.set_static('info', {'ok': False})


def test_episode_writer_prevents_signal_name_conflicting_with_static(tmp_path):
    ep_dir = tmp_path / 'ep_conflict_static_then_signal'
    with DiskEpisodeWriter(ep_dir) as w:
        # Set a static item first
        w.set_static('conflict_key', {'foo': 1})
        # Appending a signal with the same name should raise
        with np.testing.assert_raises_regex(ValueError, "Static item 'conflict_key' already set"):
            w.append('conflict_key', 123, 1000)


def test_episode_writer_prevents_static_name_conflicting_with_signal(tmp_path):
    ep_dir = tmp_path / 'ep_conflict_signal_then_static'
    with DiskEpisodeWriter(ep_dir) as w:
        # Write a signal first
        w.append('conflict_key', 1, 1000)
        # Setting a static item with the same name should raise
        with np.testing.assert_raises_regex(ValueError, "Signal 'conflict_key' already exists"):
            w.set_static('conflict_key', {'foo': 2})


class TestEpisodeVideoIntegration:
    def test_episode_writer_routes_images_to_video(self, tmp_path):
        ep_dir = tmp_path / 'ep_video'
        with DiskEpisodeWriter(ep_dir) as w:
            # Append a few frames under the same signal name
            frames = [create_frame(30), create_frame(120), create_frame(200)]
            ts = [1000, 2000, 4000]
            for f, t in zip(frames, ts, strict=False):
                w.append('cam', f, t)

        # Files created for a video signal
        assert (ep_dir / 'cam.mp4').exists()
        assert (ep_dir / 'cam.frames.parquet').exists()

        # Reader exposes the signal under the same key
        ep = DiskEpisode(ep_dir)
        cam = ep['cam']
        assert len(cam) == 3

        # Retrieve frames back and compare approximately (accounting for compression)
        for i in range(3):
            frame, t = cam[i]
            assert t == ts[i]
            assert_frames_equal(frame, frames[i], tolerance=25)

    def test_episode_mixed_vector_and_video(self, tmp_path):
        ep_dir = tmp_path / 'ep_mixed'
        with DiskEpisodeWriter(ep_dir) as w:
            # Vector signal
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)
            # Video signal
            w.append('cam', create_frame(10), 1500)
            w.append('cam', create_frame(20), 2500)

        ep = DiskEpisode(ep_dir)
        # Keys include both signals
        assert set(ep.keys()) == {'a', 'cam'}

        # Time snapshot merges both; dynamic values only
        snap = ep.time[2000]
        assert snap['a'] == 2
        cam_frame = snap['cam']
        assert_frames_equal(cam_frame, create_frame(10), tolerance=25)


class TestCoreEpisodeTime:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        ep_dir = tmp_path / 'ep_time_fixture'
        with DiskEpisodeWriter(ep_dir) as w:
            # Static items
            w.set_static('task', 'stack')
            w.set_static('version', 2)
            w.set_static('params', {'k': 1})
            # Dynamic signals
            # a: 1000->1, 2000->2, 3000->3
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)
            w.append('a', 3, 3000)
            # b: 1500->5, 2500->7, 3500->9
            w.append('b', 5, 1500)
            w.append('b', 7, 2500)
            w.append('b', 9, 3500)
        self.ep = DiskEpisode(ep_dir)

    def test_int_includes_static(self):
        snap = self.ep.time[2000]
        expected = {'task': 'stack', 'version': 2, 'params': {'k': 1}, 'a': 2, 'b': 5}
        assert snap == expected

    def test_array_preserves_static_and_samples(self):
        ts = [1500, 2500, 3000]
        sub = self.ep.time[ts]
        expected = {'task': 'stack', 'version': 2, 'params': {'k': 1}, 'a': [1, 2, 3], 'b': [5, 7, 7]}
        sub_norm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sub.items()}
        assert sub_norm == expected

    def test_no_step_slice_raises_keyerror(self):
        with pytest.raises(KeyError):
            _ = self.ep.time[1500:3000]

    def test_stepped_slice_without_end_defaults_to_episode_last_ts(self):
        start = self.ep.start_ts
        step = 500
        sub = self.ep.time[start::step]
        a_vals = sub['a']
        b_vals = sub['b']
        expected_len = len(np.arange(start, self.ep.last_ts + 1, step))
        assert len(a_vals) == len(b_vals) == expected_len


def test_disk_episode_implements_abc(tmp_path):
    ep_dir = tmp_path / 'ep_abc'
    with DiskEpisodeWriter(ep_dir) as w:
        w.append('a', 1, 1000)
        w.append('a', 2, 2000)
        w.set_static('task', 'stack')

    ep = DiskEpisode(ep_dir)

    assert isinstance(ep, Episode)

    view = ep.time[1000:3000:1000]
    assert isinstance(view, dict)


def test_episode_writer_with_extra_timelines(tmp_path):
    """Test that EpisodeWriter passes extra timelines to signal writers."""

    ep_dir = tmp_path / 'ep_extra_timelines'
    with DiskEpisodeWriter(ep_dir) as w:
        w.append('state', np.array([1.0, 2.0]), 1000, extra_ts={'producer': 900, 'consumer': 1100})
        w.append('state', np.array([3.0, 4.0]), 2000, extra_ts={'producer': 1900, 'consumer': 2100})
        w.append('image', create_frame(50), 1500, extra_ts={'producer': 1400, 'consumer': 1600})
        w.append('image', create_frame(100), 2500, extra_ts={'producer': 2400, 'consumer': 2600})

    # Verify vector signal has extra timelines
    state_table = pq.read_table(ep_dir / 'state.parquet')
    assert {'timestamp', 'ts_ns.producer', 'ts_ns.consumer'} <= set(state_table.column_names)
    assert state_table['ts_ns.producer'].to_pylist() == [900, 1900]
    assert state_table['ts_ns.consumer'].to_pylist() == [1100, 2100]

    # Verify video signal has extra timelines
    frames_table = pq.read_table(ep_dir / 'image.frames.parquet')
    assert {'ts_ns', 'ts_ns.producer', 'ts_ns.consumer'} == set(frames_table.column_names)
    assert frames_table['ts_ns.producer'].to_pylist() == [1400, 2400]
    assert frames_table['ts_ns.consumer'].to_pylist() == [1600, 2600]


class TestLazyMetaProperties:
    """Tests for lazy computation of expensive meta properties."""

    def test_duration_ns_stored_in_meta_json(self, tmp_path):
        """Test that duration_ns is written to meta.json on episode close."""

        ep_dir = tmp_path / 'ep_duration'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)
            w.append('b', 10, 1500)
            w.append('b', 20, 5000)  # last timestamp

        # Verify meta.json contains duration_ns
        with (ep_dir / 'meta.json').open('r') as f:
            stored_meta = json.load(f)
        assert 'duration_ns' in stored_meta
        assert stored_meta['duration_ns'] == 5000 - 1000  # last - first

    def test_duration_ns_read_from_meta(self, tmp_path):
        """Test that duration_ns is read from meta (fast path)."""
        ep_dir = tmp_path / 'ep_duration_read'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 3000)

        ep = DiskEpisode(ep_dir)
        # Access duration_ns - should come from meta, not compute from signals
        assert ep.duration_ns == 2000
        assert ep.meta['duration_ns'] == 2000

    def test_duration_ns_fallback_for_old_episodes(self, tmp_path):
        """Test that duration_ns falls back to computing from signals for old episodes."""

        ep_dir = tmp_path / 'ep_old'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 4000)

        # Remove duration_ns from meta.json to simulate old episode
        meta_path = ep_dir / 'meta.json'
        with meta_path.open('r') as f:
            meta = json.load(f)
        del meta['duration_ns']
        with meta_path.open('w') as f:
            json.dump(meta, f)

        # Read episode - should compute duration from signals
        ep = DiskEpisode(ep_dir)
        assert ep.duration_ns == 3000  # 4000 - 1000

    def test_size_mb_computed_lazily(self, tmp_path):
        """Test that size_mb is only computed when accessed."""
        ep_dir = tmp_path / 'ep_lazy_size'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)

        ep = DiskEpisode(ep_dir)

        # Access meta - should not compute size_mb yet
        meta = ep.meta
        # size_mb should be available (in keys) but lazy
        assert 'size_mb' in meta

        # Explicitly access size_mb
        size = meta['size_mb']
        assert isinstance(size, float)
        assert size > 0

    def test_meta_copy_preserves_laziness(self, tmp_path):
        """Test that meta.copy() preserves lazy evaluation."""

        ep_dir = tmp_path / 'ep_lazy_copy'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)

        # Remove duration_ns to test lazy fallback
        meta_path = ep_dir / 'meta.json'
        with meta_path.open('r') as f:
            meta = json.load(f)
        del meta['duration_ns']
        with meta_path.open('w') as f:
            json.dump(meta, f)

        ep = DiskEpisode(ep_dir)
        # Get a copy of meta
        meta_copy = ep.meta

        # Keys should include lazy properties
        assert 'size_mb' in meta_copy
        assert 'duration_ns' in meta_copy

        # Accessing should work
        assert meta_copy['size_mb'] > 0
        assert meta_copy['duration_ns'] == 1000  # 2000 - 1000

    def test_multiple_meta_accesses_cache_lazy_values(self, tmp_path):
        """Test that lazy values are cached after first computation."""
        ep_dir = tmp_path / 'ep_cache'
        with DiskEpisodeWriter(ep_dir) as w:
            w.append('a', 1, 1000)
            w.append('a', 2, 2000)

        ep = DiskEpisode(ep_dir)

        # First access
        meta1 = ep.meta
        size1 = meta1['size_mb']

        # Second access - should use cached internal meta
        meta2 = ep.meta
        size2 = meta2['size_mb']

        assert size1 == size2

    def test_created_ts_ns_preserved(self, tmp_path):
        """Test that created_ts_ns parameter is preserved in meta.json."""
        ep_dir = tmp_path / 'ep_created'
        original_ts = 1234567890123456789

        with DiskEpisodeWriter(ep_dir, created_ts_ns=original_ts) as w:
            w.append('a', 1, 1000)

        ep = DiskEpisode(ep_dir)
        assert ep.meta['created_ts_ns'] == original_ts

    def test_duration_computed_from_scanned_files(self, tmp_path):
        """Test that duration is computed by scanning parquet files for raw writes."""
        ep_dir = tmp_path / 'ep_scan'

        with DiskEpisodeWriter(ep_dir):
            # Write a parquet file directly (simulating migration's raw write)
            timestamps = [1000, 2000, 5000]
            table = pa.table({'timestamp': timestamps, 'value': [1, 2, 3]})
            pq.write_table(table, ep_dir / 'signal.parquet')

        # Duration should be computed from scanned file
        ep = DiskEpisode(ep_dir)
        assert ep.duration_ns == 4000  # 5000 - 1000

    def test_duration_computed_from_frames_parquet(self, tmp_path):
        """Test that duration is computed from .frames.parquet files (video signals)."""
        ep_dir = tmp_path / 'ep_video_scan'

        with DiskEpisodeWriter(ep_dir):
            # Write a frames parquet file directly (simulating video migration)
            timestamps = [2000, 3000, 8000]
            table = pa.table({'ts_ns': timestamps})
            pq.write_table(table, ep_dir / 'cam.frames.parquet')
            # Also need to create dummy video file for completeness
            (ep_dir / 'cam.mp4').write_bytes(b'dummy')

        ep = DiskEpisode(ep_dir)
        assert ep.duration_ns == 6000  # 8000 - 2000
