"""Which episodes a MolmoSpaces eval runs: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed, since a real episode list needs the MolmoSpaces venv;
``tests/e2e.py`` runs the same command against the real benchmark.
"""

import logging
from typing import Any

import pytest

from positronic.cfg.eval.sim.molmo import _TIMEOUT_MARGIN_SEC, benchmark
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.molmo_spaces import keys as molmo_keys

_HORIZON_SEC = 30.0


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every spec is answered with two episodes of one horizon."""
    specs: list[Any] = []

    def tasks(self, selection: Any) -> list[dict[str, Any]]:
        specs.append(selection)
        return [
            {
                eval_keys.TASK: 'put the banana in the bowl',
                molmo_keys.EPISODE_INDEX: 0,
                molmo_keys.TASK_HORIZON: _HORIZON_SEC,
            },
            {
                eval_keys.TASK: 'put the mug on the shelf',
                molmo_keys.EPISODE_INDEX: 1,
                molmo_keys.TASK_HORIZON: _HORIZON_SEC,
            },
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_env_answers_which_episodes_the_sweep_runs(asked):
    """The sweep is asked for when the run starts; an unset seed leaves the episode's own seed in force."""
    ev = benchmark.override(benchmark_dir='unused', trial_count=2).instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{}]
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[molmo_keys.EPISODE_INDEX] for scene in scenes] == [0, 0, 1, 1]
    assert all(eval_keys.SEED not in scene for scene in scenes)
    assert [trial.meta[eval_keys.TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]


def test_an_episode_selection_rides_the_spec(asked):
    ev = benchmark.override(benchmark_dir='unused', episodes=[0, 1]).instantiate()
    ev.tasks()
    assert asked == [{'episodes': [0, 1]}]


def test_an_explicit_seed_sweeps_each_episode(asked):
    ev = benchmark.override(benchmark_dir='unused', seed=5, trial_count=2).instantiate()
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in ev.tasks()]
    assert [scene[eval_keys.SEED] for scene in scenes] == [5, 6, 5, 6]


def test_a_non_positive_trial_count_is_refused():
    with pytest.raises(ValueError, match='trial_count'):
        benchmark.override(benchmark_dir='unused', trial_count=0).instantiate()


def test_timeout_defaults_to_the_benchmark_horizon_plus_a_margin(asked):
    ev = benchmark.override(benchmark_dir='unused').instantiate()
    trials = list(ev.tasks())
    assert [trial.timeout_sec for trial in trials] == [_HORIZON_SEC + _TIMEOUT_MARGIN_SEC] * 2


def test_explicit_timeout_can_only_lower_the_deadline(asked, caplog):
    with caplog.at_level(logging.WARNING):
        short = benchmark.override(benchmark_dir='unused', timeout=20.0).instantiate().tasks()
        long = benchmark.override(benchmark_dir='unused', timeout=999.0).instantiate().tasks()
    assert short[0].timeout_sec == 20.0
    assert long[0].timeout_sec == _HORIZON_SEC + _TIMEOUT_MARGIN_SEC
    assert len(caplog.records) == 2, 'both directions differ from the backstop, so both warn'


def test_timeout_matching_the_backstop_is_silent(asked, caplog):
    backstop = _HORIZON_SEC + _TIMEOUT_MARGIN_SEC
    with caplog.at_level(logging.WARNING):
        trials = benchmark.override(benchmark_dir='unused', timeout=backstop).instantiate().tasks()
    assert trials[0].timeout_sec == backstop
    assert not caplog.records
