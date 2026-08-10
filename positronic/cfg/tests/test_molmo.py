import json
import logging

import pytest

from positronic.cfg.eval.sim.molmo import _TIMEOUT_MARGIN_SEC, benchmark

_HORIZON_SEC = 30.0


@pytest.fixture
def benchmark_dir(tmp_path):
    """A two-episode benchmark declaring one horizon — enough for the config, which never spawns the server."""
    spec = {'task': {'task_horizon_sec': _HORIZON_SEC}}
    (tmp_path / 'benchmark.json').write_text(json.dumps([spec, spec]))
    return str(tmp_path)


def test_timeout_defaults_to_the_benchmark_horizon_plus_a_margin(benchmark_dir):
    ev = benchmark.override(benchmark_dir=benchmark_dir).instantiate()
    assert ev.task.timeout == _HORIZON_SEC + _TIMEOUT_MARGIN_SEC


def test_explicit_timeout_can_only_lower_the_deadline(benchmark_dir, caplog):
    with caplog.at_level(logging.WARNING):
        short = benchmark.override(benchmark_dir=benchmark_dir, timeout=20.0).instantiate()
        long = benchmark.override(benchmark_dir=benchmark_dir, timeout=999.0).instantiate()
    assert short.task.timeout == 20.0
    assert long.task.timeout == _HORIZON_SEC + _TIMEOUT_MARGIN_SEC
    assert len(caplog.records) == 2, 'both directions differ from the backstop, so both warn'


def test_timeout_matching_the_backstop_is_silent(benchmark_dir, caplog):
    with caplog.at_level(logging.WARNING):
        ev = benchmark.override(benchmark_dir=benchmark_dir, timeout=_HORIZON_SEC + _TIMEOUT_MARGIN_SEC).instantiate()
    assert ev.task.timeout == _HORIZON_SEC + _TIMEOUT_MARGIN_SEC
    assert not caplog.records


def test_benchmark_without_a_declared_horizon_is_rejected(tmp_path):
    (tmp_path / 'benchmark.json').write_text(json.dumps([{'task': {}}]))
    with pytest.raises(ValueError, match='task_horizon_sec'):
        benchmark.override(benchmark_dir=str(tmp_path)).instantiate()
