"""Which episodes a MolmoSpaces eval runs: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed, since a real episode list needs the MolmoSpaces venv;
``tests/e2e.py`` runs the same command against the real benchmark.
"""

import logging
from typing import Any

import pytest

from positronic.cfg.eval.sim.molmo import _TIMEOUT_MARGIN_SEC, benchmarks
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.molmo_spaces import keys as molmo_keys

_PICK = {
    molmo_keys.SUITE: 'molmospaces-bench-v1',
    molmo_keys.SCENE_DATASET: 'procthor-10k',
    molmo_keys.TASK_CONFIG: 'FrankaPickDroidMiniBench',
    molmo_keys.BENCHMARK: 'pick_20251231',
}
_PLACE = {**_PICK, molmo_keys.TASK_CONFIG: 'FrankaPickandPlaceDroidMiniBench', molmo_keys.BENCHMARK: 'pnp_20260111'}
_PICK_HORIZON_SEC = 30.0
_PLACE_HORIZON_SEC = 60.0


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every spec is answered with one episode of each of two benchmarks."""
    specs: list[Any] = []

    def tasks(self, selection: Any) -> list[dict[str, Any]]:
        specs.append(selection)
        return [
            {
                **_PICK,
                eval_keys.TASK: 'put the banana in the bowl',
                molmo_keys.EPISODE_INDEX: 0,
                molmo_keys.TASK_HORIZON: _PICK_HORIZON_SEC,
            },
            {
                **_PLACE,
                eval_keys.TASK: 'put the mug on the shelf',
                molmo_keys.EPISODE_INDEX: 0,
                molmo_keys.TASK_HORIZON: _PLACE_HORIZON_SEC,
            },
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_env_answers_which_episodes_the_sweep_runs(asked):
    """The sweep is asked for when the run starts; an unset seed leaves the episode's own seed in force."""
    ev = benchmarks.override(trial_count=2).instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{}]
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[molmo_keys.TASK_CONFIG] for scene in scenes] == [
        'FrankaPickDroidMiniBench',
        'FrankaPickDroidMiniBench',
        'FrankaPickandPlaceDroidMiniBench',
        'FrankaPickandPlaceDroidMiniBench',
    ]
    assert all(eval_keys.SEED not in scene for scene in scenes)
    assert [trial.meta[eval_keys.TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]
    assert trials[0].meta[molmo_keys.BENCHMARK] == 'pick_20251231'


def test_the_benchmark_dimensions_and_the_episode_selection_ride_the_spec(asked):
    ev = benchmarks.override(suite='molmospaces-bench-v2', task_config=['A', 'B'], episodes=[0, 1]).instantiate()
    ev.tasks()
    assert asked == [{'suite': 'molmospaces-bench-v2', 'task_config': ['A', 'B'], 'episodes': [0, 1]}]


def test_an_explicit_seed_sweeps_each_episode(asked):
    ev = benchmarks.override(seed=5, trial_count=2).instantiate()
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in ev.tasks()]
    assert [scene[eval_keys.SEED] for scene in scenes] == [5, 6, 5, 6]


def test_a_non_positive_trial_count_is_refused():
    with pytest.raises(ValueError, match='trial_count'):
        benchmarks.override(trial_count=0).instantiate()


def test_timeout_defaults_to_each_benchmark_horizon_plus_a_margin(asked, caplog):
    with caplog.at_level(logging.WARNING):
        trials = list(benchmarks.instantiate().tasks())
    expected = [_PICK_HORIZON_SEC + _TIMEOUT_MARGIN_SEC, _PLACE_HORIZON_SEC + _TIMEOUT_MARGIN_SEC]
    assert [trial.timeout_sec for trial in trials] == expected
    assert not caplog.records


def test_an_explicit_timeout_replaces_the_backstop_and_warns(asked, caplog):
    with caplog.at_level(logging.WARNING):
        short = benchmarks.override(timeout=20.0).instantiate().tasks()
        long = benchmarks.override(timeout=999.0).instantiate().tasks()
    assert [trial.timeout_sec for trial in short] == [20.0, 20.0]
    assert [trial.timeout_sec for trial in long] == [999.0, 999.0]
    assert len(caplog.records) == 2
