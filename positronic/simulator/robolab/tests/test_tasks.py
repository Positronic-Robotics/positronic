"""Which tasks a RoboLab eval runs and what it records: the adapter maps the env's records, the config sweeps them.

The env server's task listing is stubbed, since a real task list needs RoboLab's Isaac Lab interpreter;
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
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabAdapter


class _Listing:
    """The env server's task listing: two tasks of different budgets for every spec. The records state the phrasing
    that the spec asks for, or ``phrasing`` when a test sets it."""

    def __init__(self) -> None:
        self.asked: list[Any] = []
        self.phrasing: str | None = None

    def tasks(self, selection: dict[str, Any]) -> list[dict[str, Any]]:
        self.asked.append(selection)
        phrasing = self.phrasing or selection[robolab_keys.SELECT_INSTRUCTION_TYPE]
        return [
            _record('BananaInBowlTask', 50.0, 'Put the banana in the bowl', phrasing),
            _record('CleanUpToysTask', 300.0, 'Clean up the toys', phrasing),
        ]


def _record(name: str, episode_length_s: float, instruction: str, phrasing: str) -> dict[str, Any]:
    return {
        robolab_keys.RECORD_NAME: name,
        robolab_keys.RECORD_EPISODE_LENGTH_S: episode_length_s,
        protocol.TASK_INSTRUCTION: instruction,
        robolab_keys.RECORD_INSTRUCTION_TYPE: phrasing,
    }


@pytest.fixture
def listing(monkeypatch) -> _Listing:
    """The listing that the proxy reads through the real adapter."""
    fake = _Listing()
    monkeypatch.setattr(RemoteEnvControlSystem, '_connect', lambda self: fake)
    return fake


@pytest.fixture
def asked(listing) -> list[Any]:
    """The specs the eval sends the env server."""
    return listing.asked


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    params = adapter.task_params([_record('AnimalsInBinTask', 90.0, 'Put the animals in the bin', 'vague')])

    assert params == [
        {eval_keys.TASK: 'AnimalsInBinTask', robolab_keys.EPISODE_LENGTH: 90.0, robolab_keys.INSTRUCTION_TYPE: 'vague'}
    ]
    token = adapter.reset_token(params[0])
    assert token == {robolab_keys.TOKEN_TASK: 'AnimalsInBinTask', robolab_keys.TOKEN_INSTRUCTION_TYPE: 'vague'}


def test_the_reset_token_carries_an_episode_length_only_when_one_replaces_the_budget():
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})
    params = {eval_keys.TASK: 'AnimalsInBinTask', robolab_keys.INSTRUCTION_TYPE: 'default'}

    budget = adapter.reset_token({**params, robolab_keys.EPISODE_LENGTH: 90.0})
    override = adapter.reset_token({
        **params,
        robolab_keys.EPISODE_LENGTH: 90.0,
        robolab_keys.EPISODE_LENGTH_OVERRIDE: 240.0,
    })

    assert robolab_keys.TOKEN_EPISODE_LENGTH_S not in budget
    assert override[robolab_keys.TOKEN_EPISODE_LENGTH_S] == 240.0


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, and the env names each task in the phrasing the config asks
    for."""
    ev = robolab_cfg.visual.override(trial_count=2, instruction_type='specific').instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{'task': 'visual', robolab_keys.SELECT_INSTRUCTION_TYPE: 'specific'}]
    assert [trial.instruction for trial in trials] == ['Put the banana in the bowl'] * 2 + ['Clean up the toys'] * 2
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[eval_keys.TASK] for scene in scenes] == ['BananaInBowlTask'] * 2 + ['CleanUpToysTask'] * 2
    assert [trial.meta[eval_keys.TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]
    assert scenes[0] == {
        eval_keys.TASK: 'BananaInBowlTask',
        robolab_keys.EPISODE_LENGTH: 50.0,
        robolab_keys.INSTRUCTION_TYPE: 'specific',
    }


def test_the_reset_builds_the_phrasing_that_the_listed_instruction_states(asked):
    trials = list(robolab_cfg.visual.override(instruction_type='vague').instantiate().tasks())
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    tokens = [adapter.reset_token(trial.prepare_args[eval_keys.SCENE]) for trial in trials]
    assert [token[robolab_keys.TOKEN_INSTRUCTION_TYPE] for token in tokens] == ['vague', 'vague']


def test_the_config_passes_no_phrasing_apart_from_the_listing(listing):
    """The listed instruction is in the phrasing that the record names, so the reset follows the record even where
    the config asked for another."""
    listing.phrasing = 'default'
    trials = list(robolab_cfg.visual.override(instruction_type='vague').instantiate().tasks())
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    tokens = [adapter.reset_token(trial.prepare_args[eval_keys.SCENE]) for trial in trials]
    assert [token[robolab_keys.TOKEN_INSTRUCTION_TYPE] for token in tokens] == ['default', 'default']


def test_a_spec_that_names_no_task_narrows_nothing(asked):
    list(robolab_cfg.benchmark.instantiate().tasks())

    assert asked == [{robolab_keys.SELECT_INSTRUCTION_TYPE: 'default'}]


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

    phrasing = {robolab_keys.SELECT_INSTRUCTION_TYPE: 'default'}
    assert asked == [{'task': 'procedural', **phrasing}, {'task': 'RubiksCubeAndBananaTask', **phrasing}]


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


def test_a_horizon_reaches_the_reset_token_and_the_deadline_follows_it(asked):
    trials = list(robolab_cfg.benchmark.override(horizon_sec=240.0).instantiate().tasks())
    adapter = RobolabAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[robolab_keys.EPISODE_LENGTH] for scene in scenes] == [50.0, 300.0]
    assert [scene[robolab_keys.EPISODE_LENGTH_OVERRIDE] for scene in scenes] == [240.0, 240.0]
    assert [adapter.reset_token(scene)[robolab_keys.TOKEN_EPISODE_LENGTH_S] for scene in scenes] == [240.0, 240.0]
    assert [trial.timeout_sec for trial in trials] == [250.0, 250.0]


def test_a_pinned_timeout_stays_the_deadline_under_a_horizon(asked):
    trials = list(robolab_cfg.benchmark.override(horizon_sec=240.0, timeout=12.0).instantiate().tasks())

    assert [trial.timeout_sec for trial in trials] == [12.0, 12.0]


@pytest.mark.parametrize('arg', ['horizon_sec', 'timeout'])
@pytest.mark.parametrize('value', [0.0, -5.0, float('inf'), float('nan')])
def test_a_horizon_and_a_timeout_must_be_finite_and_positive(arg, value):
    with pytest.raises(ValueError, match=arg):
        robolab_cfg.benchmark.override(**{arg: value}).instantiate()


@pytest.mark.parametrize(
    'frame, recorded',
    [
        ({protocol.FRAME_DONE: False, protocol.FRAME_SUCCESS: False}, None),
        ({protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: True}, {eval_keys.SUCCESS: True}),
        ({protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False}, {eval_keys.SUCCESS: False}),
        (
            {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False, protocol.FRAME_FAULT: 'not finite'},
            {eval_keys.SUCCESS: False, robolab_keys.FAULT: 'not finite'},
        ),
    ],
    ids=['running', 'success', 'time-out', 'fault'],
)
def test_an_episode_records_how_it_ended(frame, recorded):
    assert RobolabAdapter({}).terminal(frame) == recorded
