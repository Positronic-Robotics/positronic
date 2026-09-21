"""Which tasks an ABC eval runs, and how each arm's signals map onto the canonical contract.

``RemoteEnvControlSystem.tasks`` is stubbed, since a real task list needs ABC's own interpreter.
"""

from typing import Any

import numpy as np
import pytest

import pimm
from positronic import geom, keys
from positronic.cfg.eval.sim import amazon_abc as abc_cfg
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import keys as eval_keys
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import AbcAdapter
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem

_CAMERAS = {keys.EXTERIOR_IMAGE: 'top', keys.WRIST_LEFT_IMAGE: 'left', keys.WRIST_RIGHT_IMAGE: 'right'}


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """The specs the eval sends the proxy; every spec is answered with two of ABC's tasks."""
    specs: list[Any] = []

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        specs.append(spec)
        return [{eval_keys.TASK: 'put_plastic_bottles_in_bin'}, {eval_keys.TASK: 'turn_mug_right_side_up'}]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_adapter_names_a_task_the_way_the_reset_token_does():
    adapter = AbcAdapter(_CAMERAS)

    params = adapter.task_params([{mapping.TASK_NAME: 'put_plastic_bottles_in_bin', mapping.TASK_PROMPT: 'p'}])

    assert params == [{eval_keys.TASK: 'put_plastic_bottles_in_bin'}]
    token = adapter.reset_token({
        **params[0],
        abc_keys.CAMERA_HEIGHT: 168,
        abc_keys.CAMERA_WIDTH: 224,
        eval_keys.SEED: 7,
    })
    assert token == {
        mapping.TOKEN_TASK: 'put_plastic_bottles_in_bin',
        mapping.TOKEN_SEED: 7,
        mapping.TOKEN_CAMERA_HEIGHT: 168,
        mapping.TOKEN_CAMERA_WIDTH: 224,
    }


def test_the_env_answers_which_tasks_the_sweep_runs(asked):
    """The sweep is asked for when the run starts, and the render size the config owns joins each task."""
    ev = abc_cfg.put_bottles.override(trial_count=2, seed=3).instantiate()
    assert asked == []

    trials = ev.tasks()

    assert asked == [{mapping.SELECT_TASKS: 'put_plastic_bottles_in_bin'}]
    assert len(trials) == 4  # two tasks the stub answers with, two trials each
    scene = trials[0].prepare_args[eval_keys.SCENE]
    assert scene[eval_keys.TASK] == 'put_plastic_bottles_in_bin'
    assert (scene[abc_keys.CAMERA_HEIGHT], scene[abc_keys.CAMERA_WIDTH]) == (168, 224)
    assert [t.meta[eval_keys.SEED] for t in trials] == [3, 4, 3, 4]


def test_an_unbound_task_lets_the_env_offer_its_whole_catalogue(asked):
    abc_cfg._abc_eval.instantiate().tasks()

    assert asked == [{}]


def _frame(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {mapping.OBS_SIM_STATE: np.zeros(4)}
    for index, arm in enumerate(mapping.ARMS):
        raw.update({
            protocol.arm_channel(mapping.OBS_JOINT_POS, arm): np.full(mapping.ARM_JOINTS, float(index)),
            protocol.arm_channel(mapping.OBS_JOINT_VEL, arm): np.zeros(mapping.ARM_JOINTS),
            protocol.arm_channel(mapping.OBS_EEF_POS, arm): np.array([0.1, 0.2 * index, 0.3]),
            protocol.arm_channel(mapping.OBS_EEF_QUAT, arm): np.array([1.0, 0.0, 0.0, 0.0]),
            protocol.arm_channel(mapping.OBS_GRIP, arm): np.float32(0.25 * index),
        })
    for camera in _CAMERAS.values():
        raw[camera] = np.zeros((3, 4, 6), dtype=np.uint8)
    return {**raw, **overrides}


def test_every_arm_reports_its_own_state_and_grip():
    obs = AbcAdapter(_CAMERAS).observations(_frame())

    assert set(obs) == {
        *(keys.arm_channel(keys.ROBOT_STATE, arm) for arm in mapping.ARMS),
        *(keys.arm_channel(keys.GRIP, arm) for arm in mapping.ARMS),
        *_CAMERAS,
    }
    right = obs[keys.arm_channel(keys.ROBOT_STATE, 'right')]
    assert np.array_equal(right.q, np.ones(mapping.ARM_JOINTS, dtype=np.float32))
    assert right.ee_pose.translation == pytest.approx([0.1, 0.2, 0.3])
    assert obs[keys.arm_channel(keys.GRIP, 'right')] == pytest.approx(0.25)


def test_a_camera_arrives_as_height_width_channels():
    obs = AbcAdapter(_CAMERAS).observations(_frame())

    assert obs[keys.WRIST_LEFT_IMAGE].array.shape == (4, 6, 3)


def test_the_grip_conventions_are_inverses():
    """ABC drives i2rt's aperture, where 1 is open; positronic's grip is closure, where 1 is closed."""
    assert mapping.invert_grip(0.0) == 1.0
    assert mapping.invert_grip(1.0) == 0.0
    assert mapping.invert_grip(mapping.invert_grip(0.3)) == pytest.approx(0.3)


def test_the_terminal_carries_the_envs_own_verdict():
    adapter = AbcAdapter(_CAMERAS)

    running = {protocol.FRAME_DONE: False, protocol.FRAME_SUCCESS: False}
    assert adapter.terminal(running) is None
    assert adapter.terminal({protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: True}) == {eval_keys.SUCCESS: True}
    assert adapter.terminal({protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False}) == {eval_keys.SUCCESS: False}


def test_the_embodiment_takes_a_command_and_a_grip_per_arm():
    ev = abc_cfg.put_bottles.instantiate()

    assert set(ev.embodiment.commands) == {
        *(keys.arm_channel(keys.ROBOT_COMMAND, arm) for arm in mapping.ARMS),
        *(keys.arm_channel(keys.TARGET_GRIP, arm) for arm in mapping.ARMS),
    }
    assert ev.embodiment.simulated


def test_an_arm_holds_its_last_command_and_the_other_arm_is_untouched():
    adapter = AbcAdapter(_CAMERAS)
    left = keys.arm_channel(keys.ROBOT_COMMAND, 'left')
    right = keys.arm_channel(keys.ROBOT_COMMAND, 'right')
    pose = geom.Transform3D(np.array([0.4, 0.1, 0.3]))

    action = adapter.action({
        left: pimm.Message(roboarm_command.CartesianPosition(pose), ts=0, updated=True),
        right: None,
    })
    assert action[left][protocol.COMMAND_TYPE] == protocol.CARTESIAN
    assert action[right][protocol.COMMAND_TYPE] == protocol.HOLD

    held = adapter.action({left: None, right: None})
    assert held[left][protocol.COMMAND_TYPE] == protocol.CARTESIAN
