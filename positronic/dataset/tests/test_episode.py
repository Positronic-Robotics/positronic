from pathlib import Path

import numpy as np

from positronic.dataset.episode import Episode, EpisodeWriter


def test_episode_writer_and_reader_basic(tmp_path):
    ep_dir = tmp_path / "ep1"
    w = EpisodeWriter(ep_dir)
    w.append("a", 1, 1000)
    w.append("a", 2, 2000)
    w.append("b", 10, 1500)
    w.append("b", 20, 2500)
    w.finish()

    # Files written
    assert (ep_dir / "a.parquet").exists()
    assert (ep_dir / "b.parquet").exists()

    ep = Episode(ep_dir)
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
    w = EpisodeWriter(ep_dir)
    # a: starts 1000, last 2000
    w.append("a", 1, 1000)
    w.append("a", 2, 2000)
    # b: starts 1500, last 2500
    w.append("b", 10, 1500)
    w.append("b", 20, 2500)
    w.finish()

    ep = Episode(ep_dir)
    # Current Episode implementation uses max of starts and max of lasts
    assert ep.start_ts == 1500
    assert ep.last_ts == 2500


def test_episode_getitem_returns_signal(tmp_path):
    ep_dir = tmp_path / "ep3"
    w = EpisodeWriter(ep_dir)
    w.append("x", np.array([1, 2]), 1000)
    w.append("x", np.array([3, 4]), 2000)
    w.finish()

    ep = Episode(ep_dir)
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
    w = EpisodeWriter(ep_dir)
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

    ep = Episode(ep_dir)
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
    w = EpisodeWriter(ep_dir)
    arr_i32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    arr_f32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    nested = {"cam": {"K": arr_f32.reshape(3, 1), "shape": [480, 640]}, "list": [arr_i32]}
    w.set_static("arr_i32", arr_i32)
    w.set_static("arr_f32", arr_f32)
    w.set_static("nested", nested)
    w.finish()

    ep = Episode(ep_dir)
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
    w = EpisodeWriter(ep_dir)
    w.set_static("info", {"ok": True})
    with np.testing.assert_raises_regex(ValueError, "already set"):
        w.set_static("info", {"ok": False})
