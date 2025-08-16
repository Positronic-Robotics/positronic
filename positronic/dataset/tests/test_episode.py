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

