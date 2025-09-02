import pytest

from positronic.dataset.episode import DiskEpisode, DiskEpisodeWriter


class TestCoreEpisodeTime:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        ep_dir = tmp_path / "ep_time_fixture"
        with DiskEpisodeWriter(ep_dir) as w:
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


def test_disk_episode_implements_abc(tmp_path):
    ep_dir = tmp_path / "ep_abc"
    with DiskEpisodeWriter(ep_dir) as w:
        w.append("a", 1, 1000)
        w.append("a", 2, 2000)
        w.set_static("task", "stack")

    ep = DiskEpisode(ep_dir)
    from positronic.dataset import Episode
    assert isinstance(ep, Episode)

    view = ep.time[1000:3000]
    assert isinstance(view, Episode)
