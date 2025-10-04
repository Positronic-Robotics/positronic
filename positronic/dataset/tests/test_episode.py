import numpy as np
import pytest

from positronic.dataset.episode import DiskEpisode, DiskEpisodeWriter
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
    assert set(ep.keys) == {'a', 'b'}

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


def test_episode_vector_names_roundtrip(tmp_path):
    ep_dir = tmp_path / 'ep_vec_names'
    feature_names = ['px', 'py']
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_signal_meta('vec', names=feature_names)
        w.append('vec', np.array([1.0, 2.0], dtype=np.float32), 1000)
        w.append('vec', np.array([3.0, 4.0], dtype=np.float32), 2000)

    sig = DiskEpisode(ep_dir)['vec']
    assert sig.names == feature_names


def test_episode_vector_names_mismatch_raises(tmp_path):
    ep_dir = tmp_path / 'ep_vec_names_mismatch'
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_signal_meta('vec', names=['px', 'py'])
        with pytest.raises(ValueError):
            w.set_signal_meta('vec', names=['ax', 'ay'])
        w.append('vec', np.array([1.0, 2.0], dtype=np.float32), 1000)
        w.append('vec', np.array([3.0, 4.0], dtype=np.float32), 2000)


def test_episode_video_names_not_supported(tmp_path):
    ep_dir = tmp_path / 'ep_video_names'
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_signal_meta('cam', names=['r', 'g', 'b'])
        with pytest.raises(ValueError, match='Custom names are not supported'):
            w.append('cam', create_frame(10), 1000)


def test_set_signal_meta_without_append(tmp_path):
    ep_dir = tmp_path / 'ep_meta_only'
    with DiskEpisodeWriter(ep_dir) as w:
        w.set_signal_meta('vec', names=['a', 'b'])

    assert not (ep_dir / 'vec.parquet').exists()
    ep = DiskEpisode(ep_dir)
    assert 'vec' not in ep.keys


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
    assert set(ep.keys) == {'task', 'version', 'params', 'tags', 'a'}

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
    assert m['writer'].get('name') == 'positronic.dataset.episode.DiskEpisodeWriter'
    # git info present when running inside a git repo; skip strict assertions otherwise
    if 'git' in m['writer']:
        git = m['writer']['git']
        assert isinstance(git, dict)
        assert {'commit', 'branch', 'dirty'}.issubset(git.keys())

    # Print meta.json content for debugging
    with open(ep_dir / 'meta.json') as f:
        import json

        meta_content = json.load(f)
        print('meta.json content:', meta_content)


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


def test_episode_static_rejects_tuple_and_set(tmp_path):
    ep_dir = tmp_path / 'ep_static_bad_types'
    with DiskEpisodeWriter(ep_dir) as w:
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('coords', (1, 2))
        with np.testing.assert_raises_regex(ValueError, 'JSON-serializable'):
            w.set_static('labels', {'a', 'b'})


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
        assert ep_dir.exists() and (ep_dir / 'meta.json').exists()

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
        assert set(ep.keys) == {'a', 'cam'}

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
        expected = {
            'task': 'stack',
            'version': 2,
            'params': {'k': 1},
            'a': 2,
            'b': 5,
        }
        assert snap == expected

    def test_array_preserves_static_and_samples(self):
        ts = [1500, 2500, 3000]
        sub = self.ep.time[ts]
        # Expected full dictionary comparison (static + sampled values)
        expected = {
            'task': 'stack',
            'version': 2,
            'params': {'k': 1},
            'a': [1, 2, 3],
            'b': [5, 7, 7],
        }
        import numpy as np

        sub_norm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sub.items()}
        assert sub_norm == expected

    def test_no_step_slice_raises_keyerror(self):
        import pytest

        with pytest.raises(KeyError):
            _ = self.ep.time[1500:3000]

    def test_stepped_slice_without_end_defaults_to_episode_last_ts(self):
        start = self.ep.start_ts  # common start across signals
        step = 500
        sub = self.ep.time[start::step]
        a_vals = sub['a']
        b_vals = sub['b']
        import numpy as np

        expected_len = len(np.arange(start, self.ep.last_ts + 1, step))
        assert len(a_vals) == len(b_vals) == expected_len


def test_disk_episode_implements_abc(tmp_path):
    ep_dir = tmp_path / 'ep_abc'
    with DiskEpisodeWriter(ep_dir) as w:
        w.append('a', 1, 1000)
        w.append('a', 2, 2000)
        w.set_static('task', 'stack')

    ep = DiskEpisode(ep_dir)
    from positronic.dataset import Episode

    assert isinstance(ep, Episode)

    view = ep.time[1000:3000:1000]
    assert isinstance(view, dict)
