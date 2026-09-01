"""Which tasks a LIBERO eval runs: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed here, so the env, the socket and the adapter stay out of the
config tests — a task list needs LIBERO's own 3.10 interpreter, which the normal suite has no access to. The
adapter's own mapping is tested below, and ``test_e2e.py`` runs the same command against the real benchmark.
"""

from typing import Any

import pytest

from positronic import keys
from positronic.cfg.eval.sim import libero as libero_cfg
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.libero.adapter import LiberoAdapter


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every suite a spec names is answered with two tasks."""
    specs: list[Any] = []

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        specs.append(spec)
        suites = [spec['suite']] if isinstance(spec['suite'], str) else spec['suite']
        return [
            {keys.EVAL_TASK: f'{suite}_{i}', keys.EVAL_SUITE: suite, keys.EVAL_TASK_ID: i}
            for suite in suites
            for i in (0, 1)
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    """A record carries the suite and the index, which is what the token needs to name the task again."""
    adapter = LiberoAdapter({keys.EXTERIOR_IMAGE: 'agentview_image'})

    params = adapter.task_params([{'suite': 'libero_goal', 'task_id': 4, 'name': 'KITCHEN_SCENE_open_the_drawer'}])

    assert params[0][keys.EVAL_TASK] == 'KITCHEN_SCENE_open_the_drawer', 'the episode records this id'

    token = adapter.reset_token({
        **params[0],
        keys.EVAL_CAMERA_RESOLUTION: 128,
        keys.EVAL_CONTROL_MODE: 'ee',
        keys.EVAL_SETTLE_STEPS: 0,
    })
    assert (token['suite'], token['task_id']) == ('libero_goal', 4)


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, not when the config is built, so a live env reports its own
    tasks. The render and control settings are the config's, and join each task the env names."""
    ev = libero_cfg.spatial.override(
        seed=7, trial_count=2, camera_resolution=64, control_mode='joint', settle_steps=3
    ).instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{'suite': 'libero_spatial'}], 'an unbound task_id is absent, so the env narrows nothing'
    scenes = [trial.prepare_args[keys.SCENE] for trial in trials]
    assert [(scene[keys.EVAL_TASK_ID], scene[keys.EVAL_SEED]) for scene in scenes] == [(0, 7), (0, 8), (1, 7), (1, 8)]
    assert scenes[0] == {
        keys.EVAL_TASK: 'libero_spatial_0',
        keys.EVAL_SUITE: 'libero_spatial',
        keys.EVAL_TASK_ID: 0,
        keys.EVAL_CAMERA_RESOLUTION: 64,
        keys.EVAL_CONTROL_MODE: 'joint',
        keys.EVAL_SETTLE_STEPS: 3,
        keys.EVAL_SEED: 7,
    }


def test_a_pinned_task_id_reaches_the_env(asked):
    """``--eval.task_id`` narrows the selection the env resolves, so only that task comes back."""
    ev = libero_cfg.spatial.override(task_id=3).instantiate()

    list(ev.tasks())

    assert asked == [{'suite': 'libero_spatial', 'task_id': 3}]


def test_a_suite_list_is_one_selection(asked):
    """``all`` sweeps four suites in one run, so the env resolves them in one answer."""
    ev = libero_cfg.all.instantiate()

    list(ev.tasks())

    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    assert asked == [{'suite': suites}]
