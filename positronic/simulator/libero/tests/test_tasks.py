"""Which tasks a LIBERO eval runs: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed, since a real task list needs LIBERO's 3.10 interpreter;
``test_e2e.py`` runs the same command against the real benchmark.
"""

from typing import Any

import pytest

from positronic import keys
from positronic.cfg.eval.sim import libero as libero_cfg
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.libero import keys as libero_keys
from positronic.simulator.libero.adapter import LiberoAdapter


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every suite a spec names is answered with two tasks."""
    specs: list[Any] = []

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        specs.append(spec)
        suites = [spec['suite']] if isinstance(spec['suite'], str) else spec['suite']
        return [
            {eval_keys.TASK: f'{suite}_{i}', libero_keys.SUITE: suite, libero_keys.TASK_ID: i}
            for suite in suites
            for i in (0, 1)
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    adapter = LiberoAdapter({keys.EXTERIOR_IMAGE: 'agentview_image'})

    params = adapter.task_params([{'suite': 'libero_goal', 'task_id': 4, 'name': 'KITCHEN_SCENE_open_the_drawer'}])

    assert params[0][eval_keys.TASK] == 'KITCHEN_SCENE_open_the_drawer', 'the episode records this id'

    token = adapter.reset_token({
        **params[0],
        libero_keys.CAMERA_RESOLUTION: 128,
        libero_keys.CONTROL_MODE: 'ee',
        libero_keys.SETTLE_STEPS: 0,
    })
    assert (token['suite'], token['task_id']) == ('libero_goal', 4)


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, and the config's render and control settings join each task
    the env names."""
    ev = libero_cfg.spatial.override(
        seed=7, trial_count=2, camera_resolution=64, control_mode='joint', settle_steps=3
    ).instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{'suite': 'libero_spatial'}], 'an unbound task_id is absent, so the env narrows nothing'
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [(scene[libero_keys.TASK_ID], scene[eval_keys.SEED]) for scene in scenes] == [(0, 7), (0, 8), (1, 7), (1, 8)]
    assert scenes[0] == {
        eval_keys.TASK: 'libero_spatial_0',
        libero_keys.SUITE: 'libero_spatial',
        libero_keys.TASK_ID: 0,
        libero_keys.CAMERA_RESOLUTION: 64,
        libero_keys.CONTROL_MODE: 'joint',
        libero_keys.SETTLE_STEPS: 3,
        eval_keys.SEED: 7,
    }


def test_a_pinned_task_id_reaches_the_env(asked):
    ev = libero_cfg.spatial.override(task_id=3).instantiate()

    list(ev.tasks())

    assert asked == [{'suite': 'libero_spatial', 'task_id': 3}]


def test_a_suite_list_is_one_selection(asked):
    ev = libero_cfg.all.instantiate()

    list(ev.tasks())

    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    assert asked == [{'suite': suites}]
