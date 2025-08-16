import numpy as np
import pytest

from positronic.dataset.episode import DiskEpisode, DiskEpisodeWriter
from positronic.dataset.core import Episode
from positronic.dataset.tests.test_video import create_frame, assert_frames_equal


def test_episode_writer_and_reader_basic(tmp_path):
    ep_dir = tmp_path / "ep1"
    w = DiskEpisodeWriter(ep_dir)
    w.append("a", 1, 1000)
    w.append("a", 2, 2000)
    w.append("b", 10, 1500)
    w.append("b", 20, 2500)
    w.finish()

    # Files written
    assert (ep_dir / "a.parquet").exists()
    assert (ep_dir / "b.parquet").exists()

    ep = DiskEpisode(ep_dir)
    # Keys present
    assert set(ep.keys) == {"a", "b"}

    # Access signals and basic data
    a = ep["a"]
    b = ep["b"]
    assert len(a) == 2
    assert len(b) == 2
    assert a[0] == (1, 1000)
    assert b[1] == (20, 2500)


def test_episode_start_last_ts(tmp_path):
    ep_dir = tmp_path / "ep2"
    w = DiskEpisodeWriter(ep_dir)
    # a: starts 1000, last 2000
    w.append("a", 1, 1000)
    w.append("a", 2, 2000)
    # b: starts 1500, last 2500
    w.append("b", 10, 1500)
    w.append("b", 20, 2500)
    w.finish()

    ep = DiskEpisode(ep_dir)
    # Current Episode implementation uses max of starts and max of lasts
    assert ep.start_ts == 1500
    assert ep.last_ts == 2500


def test_episode_getitem_returns_signal(tmp_path):
    ep_dir = tmp_path / "ep3"
    w = DiskEpisodeWriter(ep_dir)
    w.append("x", np.array([1, 2]), 1000)
    w.append("x", np.array([3, 4]), 2000)
    w.finish()

    ep = DiskEpisode(ep_dir)
    x = ep["x"]
    assert len(x) == 2
    v0, t0 = x[0]
    v1, t1 = x[1]
    assert t0 == 1000 and t1 == 2000
    # vector contents preserved
    assert list(v0) == [1, 2]
    assert list(v1) == [3, 4]


def test_episode_static_items_json(tmp_path):
    ep_dir = tmp_path / "ep_static"
    w = DiskEpisodeWriter(ep_dir)
    # write static metadata (single file episode.json)
    w.set_static("task", "pick_place")
    w.set_static("version", 1)
    w.set_static("params", {"speed": 0.5})
    w.set_static("tags", ["demo", "test"])
    # also write a dynamic signal
    w.append("a", 42, 1000)
    w.append("a", 43, 2000)
    w.finish()

    # Files written
    assert (ep_dir / "episode.json").exists()
    assert (ep_dir / "a.parquet").exists()

    ep = DiskEpisode(ep_dir)
    # keys include both static and dynamic names
    assert set(ep.keys) == {"task", "version", "params", "tags", "a"}

    # static access returns the value directly
    assert ep["task"] == "pick_place"
    assert ep["version"] == 1
    assert ep["params"] == {"speed": 0.5}
    assert ep["tags"] == ["demo", "test"]

    # dynamic access still returns a Signal
    a = ep["a"]
    assert len(a) == 2
    assert a[0] == (42, 1000)


def test_episode_static_numpy_arrays_roundtrip(tmp_path):
    ep_dir = tmp_path / "ep_static_np"
    w = DiskEpisodeWriter(ep_dir)
    arr_i32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    arr_f32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    nested = {"cam": {"K": arr_f32.reshape(3, 1), "shape": [480, 640]}, "list": [arr_i32]}
    w.set_static("arr_i32", arr_i32)
    w.set_static("arr_f32", arr_f32)
    w.set_static("nested", nested)
    w.finish()

    ep = DiskEpisode(ep_dir)
    out_i32 = ep["arr_i32"]
    out_f32 = ep["arr_f32"]
    out_nested = ep["nested"]

    assert isinstance(out_i32, np.ndarray) and out_i32.dtype == np.int32
    assert isinstance(out_f32, np.ndarray) and out_f32.dtype == np.float32
    np.testing.assert_array_equal(out_i32, arr_i32)
    np.testing.assert_array_equal(out_f32, arr_f32)
    assert isinstance(out_nested["cam"]["K"], np.ndarray)
    np.testing.assert_array_equal(out_nested["cam"]["K"], arr_f32.reshape(3, 1))


def test_episode_writer_set_static_twice_raises(tmp_path):
    ep_dir = tmp_path / "ep_static_dup"
    w = DiskEpisodeWriter(ep_dir)
    w.set_static("info", {"ok": True})
    with np.testing.assert_raises_regex(ValueError, "already set"):
        w.set_static("info", {"ok": False})


def test_episode_writer_prevents_signal_name_conflicting_with_static(tmp_path):
    ep_dir = tmp_path / "ep_conflict_static_then_signal"
    w = DiskEpisodeWriter(ep_dir)
    # Set a static item first
    w.set_static("conflict_key", {"foo": 1})
    # Appending a signal with the same name should raise
    with np.testing.assert_raises_regex(ValueError, "Static item 'conflict_key' already set"):
        w.append("conflict_key", 123, 1000)


def test_episode_writer_prevents_static_name_conflicting_with_signal(tmp_path):
    ep_dir = tmp_path / "ep_conflict_signal_then_static"
    w = DiskEpisodeWriter(ep_dir)
    # Write a signal first
    w.append("conflict_key", 1, 1000)
    # Setting a static item with the same name should raise
    with np.testing.assert_raises_regex(ValueError, "Signal 'conflict_key' already exists"):
        w.set_static("conflict_key", {"foo": 2})


class TestEpisodeTimeAccessor:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        ep_dir = tmp_path / "ep_time_fixture"
        w = DiskEpisodeWriter(ep_dir)
        # Static items
        w.set_static("task", "stack")
        w.set_static("version", 2)
        w.set_static("params", {"k": 1})
        # Dynamic signals
        # a: 1000->1, 2000->2, 3000->3
        w.append("a", 1, 1000)
        w.append("a", 2, 2000)
        w.append("a", 3, 3000)
        # b: 1500->5, 2500->7, 3500->9
        w.append("b", 5, 1500)
        w.append("b", 7, 2500)
        w.append("b", 9, 3500)
        w.finish()

        self.ep = DiskEpisode(ep_dir)


    def test_int_includes_static(self):
        snap = self.ep.time[2000]
        # includes static keys
        assert snap["task"] == "stack"
        assert snap["version"] == 2
        # includes dynamic samples at-or-before timestamp
        assert snap["a"] == (2, 2000)
        assert snap["b"] == (5, 1500)


    def test_array_preserves_static_and_samples(self):
        ts = [1500, 2500, 3000]
        sub = self.ep.time[ts]
        # static preserved
        assert sub["task"] == "stack"
        # dynamic are signals sampled at requested timestamps, length equals len(ts)
        a = sub["a"]
        b = sub["b"]
        assert len(a) == 3 and len(b) == 3
        # timestamps of sampled signals equal requested timestamps
        assert a[0][1] == 1500 and a[1][1] == 2500 and a[2][1] == 3000
        assert b[0][1] == 1500 and b[1][1] == 2500 and b[2][1] == 3000
        # values correspond to last at-or-before
        assert a[0][0] == 1 and a[1][0] == 2 and a[2][0] == 3
        assert b[0][0] == 5 and b[1][0] == 7 and b[2][0] == 7


    def test_slice_preserves_static_and_window(self):
        # slice with step produces requested timestamps [1500, 2500, 3500)
        sub = self.ep.time[1500:3501:1000]
        assert sub["params"] == {"k": 1}
        a = sub["a"]
        b = sub["b"]
        assert len(a) == 3 and len(b) == 3
        assert [a[i][1] for i in range(3)] == [1500, 2500, 3500]
        assert [b[i][1] for i in range(3)] == [1500, 2500, 3500]


def test_disk_episode_implements_abc(tmp_path):
    ep_dir = tmp_path / "ep_abc"
    w = DiskEpisodeWriter(ep_dir)
    w.append("a", 1, 1000)
    w.append("a", 2, 2000)
    w.set_static("task", "stack")
    w.finish()

    ep = DiskEpisode(ep_dir)
    assert isinstance(ep, Episode)

    view = ep.time[1000:3000]
    assert isinstance(view, Episode)


class TestEpisodeVideoIntegration:
    def test_episode_writer_routes_images_to_video(self, tmp_path):
        ep_dir = tmp_path / "ep_video"
        w = DiskEpisodeWriter(ep_dir)

        # Append a few frames under the same signal name
        frames = [create_frame(30), create_frame(120), create_frame(200)]
        ts = [1000, 2000, 4000]
        for f, t in zip(frames, ts):
            w.append("cam", f, t)
        w.finish()

        # Files created for a video signal
        assert (ep_dir / "cam.mp4").exists()
        assert (ep_dir / "cam.frames.parquet").exists()

        # Reader exposes the signal under the same key
        ep = DiskEpisode(ep_dir)
        cam = ep["cam"]
        assert len(cam) == 3

        # Retrieve frames back and compare approximately (accounting for compression)
        for i in range(3):
            frame, t = cam[i]
            assert t == ts[i]
            assert_frames_equal(frame, frames[i], tolerance=25)

    def test_episode_mixed_vector_and_video(self, tmp_path):
        ep_dir = tmp_path / "ep_mixed"
        w = DiskEpisodeWriter(ep_dir)
        # Vector signal
        w.append("a", 1, 1000)
        w.append("a", 2, 2000)
        # Video signal
        w.append("cam", create_frame(10), 1500)
        w.append("cam", create_frame(20), 2500)
        w.finish()

        ep = DiskEpisode(ep_dir)
        # Keys include both signals
        assert set(ep.keys) == {"a", "cam"}

        # Time snapshot merges both
        snap = ep.time[2000]
        assert snap["a"] == (2, 2000)
        cam_frame, cam_ts = snap["cam"]
        assert cam_ts == 1500
        assert_frames_equal(cam_frame, create_frame(10), tolerance=25)
