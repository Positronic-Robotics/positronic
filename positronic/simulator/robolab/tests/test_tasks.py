"""Which tasks a RoboLab eval runs: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed here, so the env, the socket and the adapter stay out of the
config tests — reading RoboLab's task metadata needs its own Isaac Lab interpreter, which the normal suite
has no access to. The adapter's own mapping is tested below, and ``e2e.py`` runs the same command against
the real benchmark.
"""

from typing import Any

import pytest

from positronic import keys
from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.robolab.adapter import RobolabAdapter


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The selections the eval sends the proxy; every selection is answered with two tasks of different
    budgets."""
    specs: list[Any] = []

    def tasks(self, selection: Any) -> list[dict[str, Any]]:
        specs.append(selection)
        return [
            {keys.EVAL_TASK: 'BananaInBowlTask', keys.EVAL_EPISODE_LENGTH: 50.0},
            {keys.EVAL_TASK: 'CleanUpToysTask', keys.EVAL_EPISODE_LENGTH: 300.0},
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    """A record carries the name and the budget, and the name is what the token needs to build the task."""
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    params = adapter.task_params([{'name': 'AnimalsInBinTask', 'episode_length_s': 90.0}])

    assert params == [{keys.EVAL_TASK: 'AnimalsInBinTask', keys.EVAL_EPISODE_LENGTH: 90.0}]
    token = adapter.reset_token({**params[0], keys.EVAL_INSTRUCTION_TYPE: 'vague'})
    assert token == {'task': 'AnimalsInBinTask', 'instruction_type': 'vague'}


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, not when the config is built, so a live env reports its own
    tasks. The instruction phrasing is the config's, and joins every task the env names."""
    ev = robolab_cfg.visual.override(trial_count=2, instruction_type='specific').instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{'task': 'visual'}]
    scenes = [trial.prepare_args[keys.SCENE] for trial in trials]
    assert [scene[keys.EVAL_TASK] for scene in scenes] == ['BananaInBowlTask'] * 2 + ['CleanUpToysTask'] * 2
    assert [trial.meta[keys.EVAL_TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]
    assert scenes[0] == {
        keys.EVAL_TASK: 'BananaInBowlTask',
        keys.EVAL_EPISODE_LENGTH: 50.0,
        keys.EVAL_INSTRUCTION_TYPE: 'specific',
    }


def test_a_spec_that_names_no_task_narrows_nothing(asked):
    """The whole benchmark binds no task, so the spec carries no key and the env selects every task."""
    list(robolab_cfg.benchmark.instantiate().tasks())

    assert asked == [{}]


def test_each_task_runs_under_the_budget_the_benchmark_gives_it(asked):
    """RoboLab gives each task its own episode length and truncates there, so one deadline for the whole sweep
    would cut a long task short."""
    trials = list(robolab_cfg.benchmark.instantiate().tasks())

    assert [trial.timeout_sec for trial in trials] == [60.0, 310.0]


def test_a_pinned_timeout_overrides_every_task(asked):
    """``--eval.timeout`` is one deadline for the whole sweep, whatever the benchmark budgets."""
    trials = list(robolab_cfg.benchmark.override(timeout=12.0).instantiate().tasks())

    assert [trial.timeout_sec for trial in trials] == [12.0, 12.0]


def test_the_spec_reaches_the_env_unmapped(asked):
    """A category is the benchmark's own vocabulary, so the env resolves it and positronic maps nothing."""
    list(robolab_cfg.procedural.instantiate().tasks())
    list(robolab_cfg.rubiks_cube_and_banana.instantiate().tasks())

    assert asked == [{'task': 'procedural'}, {'task': 'RubiksCubeAndBananaTask'}]
