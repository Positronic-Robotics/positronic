"""Which tasks a RoboLab eval runs and what it records: the adapter maps the env's records, the config sweeps them.

``RemoteEnvControlSystem.tasks`` is stubbed, since a real task list needs RoboLab's Isaac Lab interpreter;
``e2e.py`` runs the same command against the real benchmark.
"""

from typing import Any

import numpy as np
import pytest
from eval_vocabulary.episode import (
    GT_STATE_GRIPPER_CONTACT,
    GT_STATE_OBJECTS,
    GT_STATE_POSE,
    GT_STATE_PROGRESS,
    GT_STATE_SIGNAL,
    SUBTASK_SIGNAL,
)

from positronic import keys
from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.dataset.serializers import expand_suffixed
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabAdapter


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every spec is answered with two tasks of different budgets."""
    specs: list[Any] = []

    def tasks(self, selection: Any) -> list[dict[str, Any]]:
        specs.append(selection)
        return [
            {eval_keys.TASK: 'BananaInBowlTask', robolab_keys.EPISODE_LENGTH: 50.0},
            {eval_keys.TASK: 'CleanUpToysTask', robolab_keys.EPISODE_LENGTH: 300.0},
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    params = adapter.task_params([{'name': 'AnimalsInBinTask', 'episode_length_s': 90.0}])

    assert params == [{eval_keys.TASK: 'AnimalsInBinTask', robolab_keys.EPISODE_LENGTH: 90.0}]
    token = adapter.reset_token({**params[0], robolab_keys.INSTRUCTION_TYPE: 'vague'})
    assert token == {'task': 'AnimalsInBinTask', 'instruction_type': 'vague'}


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, and the config's instruction phrasing joins each task the
    env names."""
    ev = robolab_cfg.visual.override(trial_count=2, instruction_type='specific').instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{'task': 'visual'}]
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[eval_keys.TASK] for scene in scenes] == ['BananaInBowlTask'] * 2 + ['CleanUpToysTask'] * 2
    assert [trial.meta[eval_keys.TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]
    assert scenes[0] == {
        eval_keys.TASK: 'BananaInBowlTask',
        robolab_keys.EPISODE_LENGTH: 50.0,
        robolab_keys.INSTRUCTION_TYPE: 'specific',
    }


def test_a_spec_that_names_no_task_narrows_nothing(asked):
    list(robolab_cfg.benchmark.instantiate().tasks())

    assert asked == [{}]


def test_each_task_runs_under_the_budget_the_benchmark_gives_it(asked):
    """One deadline for the whole sweep would cut a long task short."""
    trials = list(robolab_cfg.benchmark.instantiate().tasks())

    assert [trial.timeout_sec for trial in trials] == [60.0, 310.0]


def test_a_pinned_timeout_overrides_every_task(asked):
    """``--eval.timeout`` is one deadline for the whole sweep, whatever the benchmark budgets."""
    trials = list(robolab_cfg.benchmark.override(timeout=12.0).instantiate().tasks())

    assert [trial.timeout_sec for trial in trials] == [12.0, 12.0]


def test_the_spec_reaches_the_env_unmapped(asked):
    list(robolab_cfg.procedural.instantiate().tasks())
    list(robolab_cfg.rubiks_cube_and_banana.instantiate().tasks())

    assert asked == [{'task': 'procedural'}, {'task': 'RubiksCubeAndBananaTask'}]


_GT_STATE = {
    f'.{robolab_keys.GT_STATE_OBJECTS}.banana.{robolab_keys.GT_STATE_POSE}': np.zeros(7, dtype=np.float32),
    f'.{robolab_keys.GT_STATE_OBJECTS}.banana.{robolab_keys.GT_STATE_GRIPPER_CONTACT}': True,
    f'.{robolab_keys.GT_STATE_PROGRESS}.0.banana.0.object_grabbed': True,
}


def test_the_privileged_channel_carries_the_subtask_progress_and_the_scene_state():
    subtask = np.array([0, 0, 1, 0.0], dtype=np.float32)
    raw = {
        robolab_keys.SUBTASK: subtask,
        robolab_keys.GT_STATE: _GT_STATE,
        robolab_keys.WRIST_CAMERA: np.zeros((2, 2, 3)),
    }

    privileged = RobolabAdapter({}).privileged(raw)

    assert privileged.keys() == {robolab_keys.SUBTASK, robolab_keys.GT_STATE}
    assert privileged[robolab_keys.SUBTASK] is subtask
    assert privileged[robolab_keys.GT_STATE] is _GT_STATE


def test_the_eval_records_the_scene_state_that_the_env_sends():
    ev = robolab_cfg.banana_in_bowl.instantiate()
    proxy = ev.privileged[SUBTASK_SIGNAL].owner

    assert ev.privileged.keys() == {SUBTASK_SIGNAL, GT_STATE_SIGNAL}
    assert ev.privileged[SUBTASK_SIGNAL] is proxy.privileged[robolab_keys.SUBTASK]
    assert ev.privileged[GT_STATE_SIGNAL] is proxy.privileged[robolab_keys.GT_STATE]


def test_the_env_writes_the_scene_state_in_the_layout_that_a_labeler_reads():
    assert [robolab_keys.GT_STATE_OBJECTS, robolab_keys.GT_STATE_POSE, robolab_keys.GT_STATE_GRIPPER_CONTACT] == [
        GT_STATE_OBJECTS,
        GT_STATE_POSE,
        GT_STATE_GRIPPER_CONTACT,
    ]
    assert robolab_keys.GT_STATE_PROGRESS == GT_STATE_PROGRESS
    recorded = dict(expand_suffixed(GT_STATE_SIGNAL, _GT_STATE))
    assert f'{GT_STATE_SIGNAL}.{GT_STATE_OBJECTS}.banana.{GT_STATE_POSE}' in recorded
