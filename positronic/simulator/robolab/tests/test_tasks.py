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

import pimm
from positronic import geom, keys
from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabDroidAdapter, RobolabYamAdapter
from positronic.simulator.robolab.scene_key import SceneKey, scene_key


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
    adapter = RobolabDroidAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    params = adapter.task_params([_record('AnimalsInBinTask', 90.0, 'Put the animals in the bin', 'vague')])

    assert params == [
        {eval_keys.TASK: 'AnimalsInBinTask', robolab_keys.EPISODE_LENGTH: 90.0, robolab_keys.INSTRUCTION_TYPE: 'vague'}
    ]
    token = adapter.reset_token({**params[0], robolab_keys.CAMERAS: robolab_keys.WRIST_LEFT})
    assert token == {
        robolab_keys.TOKEN_TASK: 'AnimalsInBinTask',
        robolab_keys.TOKEN_INSTRUCTION_TYPE: 'vague',
        robolab_keys.TOKEN_CAMERAS: robolab_keys.WRIST_LEFT,
    }


def test_the_reset_token_carries_an_episode_length_only_when_one_replaces_the_budget():
    adapter = RobolabDroidAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})
    params = {
        eval_keys.TASK: 'AnimalsInBinTask',
        robolab_keys.INSTRUCTION_TYPE: 'default',
        robolab_keys.CAMERAS: robolab_keys.WRIST_LEFT,
    }

    budget = adapter.reset_token({**params, robolab_keys.EPISODE_LENGTH: 90.0})
    override = adapter.reset_token({
        **params,
        robolab_keys.EPISODE_LENGTH: 90.0,
        robolab_keys.EPISODE_LENGTH_OVERRIDE: 240.0,
    })

    assert robolab_keys.TOKEN_EPISODE_LENGTH_S not in budget
    assert override[robolab_keys.TOKEN_EPISODE_LENGTH_S] == 240.0


def test_the_reset_token_carries_a_scene_state_only_when_the_params_give_one():
    adapter = RobolabYamAdapter({keys.EXTERIOR_IMAGE: robolab_keys.TOP_CAMERA})
    params = {
        eval_keys.TASK: 'BananaInBowlTask',
        robolab_keys.INSTRUCTION_TYPE: 'default',
        robolab_keys.CAMERAS: robolab_keys.TOP_AND_WRISTS,
    }
    banana = {robolab_keys.STATE_ROOT_POSE: [0.5, -0.1, 0.02, 1.0, 0.0, 0.0, 0.0]}
    state = {robolab_keys.STATE_RIGID_OBJECTS: {'banana': banana}}

    assert robolab_keys.TOKEN_STATE not in adapter.reset_token(params)
    assert adapter.reset_token({**params, robolab_keys.STATE: state})[robolab_keys.TOKEN_STATE] == state


@pytest.mark.parametrize('cameras', list(robolab_keys.CAMERA_SETS))
def test_each_reset_token_names_the_camera_set_of_its_eval(listing, cameras):
    trials = list(robolab_cfg.benchmark.override(cameras=cameras).instantiate().tasks())
    adapter = RobolabDroidAdapter({})

    tokens = [adapter.reset_token(trial.prepare_args[eval_keys.SCENE]) for trial in trials]
    assert [token[robolab_keys.TOKEN_CAMERAS] for token in tokens] == [cameras, cameras]


def test_two_camera_sets_of_one_task_have_two_scene_keys():
    adapter = RobolabDroidAdapter({})
    params = {eval_keys.TASK: 'BananaInBowlTask', robolab_keys.INSTRUCTION_TYPE: 'default'}

    def key(cameras: str) -> SceneKey:
        return scene_key(adapter.reset_token({**params, robolab_keys.CAMERAS: cameras}), 4)

    assert key(robolab_keys.WRIST_LEFT_RIGHT) != key(robolab_keys.EXTERIOR_PAIR_WRIST)
    assert key(robolab_keys.WRIST_LEFT_RIGHT) == key(robolab_keys.WRIST_LEFT_RIGHT)


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
        robolab_keys.CAMERAS: robolab_keys.WRIST_LEFT_RIGHT,
    }


def test_the_reset_builds_the_phrasing_that_the_listed_instruction_states(asked):
    trials = list(robolab_cfg.visual.override(instruction_type='vague').instantiate().tasks())
    adapter = RobolabDroidAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

    tokens = [adapter.reset_token(trial.prepare_args[eval_keys.SCENE]) for trial in trials]
    assert [token[robolab_keys.TOKEN_INSTRUCTION_TYPE] for token in tokens] == ['vague', 'vague']


def test_the_config_passes_no_phrasing_apart_from_the_listing(listing):
    """The listed instruction is in the phrasing that the record names, so the reset follows the record even where
    the config asked for another."""
    listing.phrasing = 'default'
    trials = list(robolab_cfg.visual.override(instruction_type='vague').instantiate().tasks())
    adapter = RobolabDroidAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

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

    privileged = RobolabDroidAdapter({}).privileged(raw)

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
    adapter = RobolabDroidAdapter({keys.EXTERIOR_IMAGE: 'over_shoulder_left_camera'})

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
            {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False, robolab_keys.FRAME_FAULT: 'not finite'},
            {eval_keys.SUCCESS: False, robolab_keys.FAULT: 'not finite'},
        ),
    ],
    ids=['running', 'success', 'failure', 'fault'],
)
def test_the_terminal_payload_carries_the_verdict_and_a_fault(frame, recorded):
    assert RobolabDroidAdapter({}).terminal(frame) == recorded


def test_a_yam_eval_drives_both_arms_and_reads_the_three_yam_cameras(listing):
    ev = robolab_cfg.yam_banana_in_bowl.instantiate()

    assert ev.embodiment.descriptor == 'remote.robolab.yam_bimanual'
    arms = {keys.arm_channel(channel, arm) for arm in keys.BIMANUAL_ARMS for channel in (keys.ROBOT_STATE, keys.GRIP)}
    cameras = {keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE}
    assert set(ev.embodiment.observations) == arms | cameras
    commands = {keys.arm_channel(c, arm) for arm in keys.BIMANUAL_ARMS for c in (keys.ROBOT_COMMAND, keys.TARGET_GRIP)}
    assert set(ev.embodiment.commands) == commands


def test_an_eval_that_names_only_the_yam_renders_the_three_yam_cameras(listing):
    ev = robolab_cfg.banana_in_bowl.override(robot=robolab_keys.YAM).instantiate()

    cameras = {keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE}
    assert cameras <= set(ev.embodiment.observations)


def test_a_yam_eval_with_a_droid_camera_set_fails():
    with pytest.raises(ValueError, match='camera sets'):
        robolab_cfg.yam_banana_in_bowl.override(cameras=robolab_keys.WRIST_LEFT).instantiate()


_YAM_EEF = {
    keys.LEFT_ARM: geom.Transform3D(np.array([0.5, 0.25, 0.4]), geom.Rotation.from_euler([0.1, 0.2, 0.3])),
    keys.RIGHT_ARM: geom.Transform3D(np.array([0.55, -0.2, 0.35]), geom.Rotation.from_euler([-0.3, 0.1, -0.2])),
}


def _yam_raw_obs() -> dict[str, Any]:
    raw: dict[str, Any] = {robolab_keys.TOP_CAMERA: np.full((4, 6, 3), 7, dtype=np.uint8)}
    for index, (arm, eef) in enumerate(_YAM_EEF.items()):
        raw[protocol.arm_channel(protocol.OBS_EEF_POS, arm)] = eef.translation
        raw[protocol.arm_channel(protocol.OBS_EEF_QUAT, arm)] = eef.rotation.as_quat
        raw[protocol.arm_channel(protocol.OBS_JOINT_POS, arm)] = np.full(6, 0.1 * (index + 1))
        raw[protocol.arm_channel(protocol.OBS_JOINT_VEL, arm)] = np.full(6, -0.1 * (index + 1))
        raw[protocol.arm_channel(protocol.OBS_GRIP, arm)] = 0.25 * (index + 1)
    return raw


def test_the_yam_adapter_reports_each_arm_where_the_env_measures_it():
    obs = RobolabYamAdapter({keys.EXTERIOR_IMAGE: robolab_keys.TOP_CAMERA}).observations(_yam_raw_obs())

    for index, (arm, eef) in enumerate(_YAM_EEF.items()):
        state = obs[keys.arm_channel(keys.ROBOT_STATE, arm)]
        np.testing.assert_allclose(state.ee_pose.as_matrix, eef.as_matrix, atol=1e-6)
        np.testing.assert_allclose(state.q, np.full(6, 0.1 * (index + 1)), atol=1e-6)
        np.testing.assert_allclose(state.dq, np.full(6, -0.1 * (index + 1)), atol=1e-6)
        assert obs[keys.arm_channel(keys.GRIP, arm)] == 0.25 * (index + 1)
    assert (obs[keys.EXTERIOR_IMAGE].array == 7).all()


def test_the_yam_adapter_sends_each_arm_its_own_command_and_grip():
    adapter = RobolabYamAdapter({})
    adapter.observations(_yam_raw_obs())
    targets = {arm: np.full(6, 0.3 * (index + 1)) for index, arm in enumerate(keys.BIMANUAL_ARMS)}
    commands: dict[str, pimm.Message | None] = {}
    for index, (arm, target) in enumerate(targets.items()):
        commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = pimm.Message(roboarm_command.JointPosition(target))
        commands[keys.arm_channel(keys.TARGET_GRIP, arm)] = pimm.Message(float(index))

    action = adapter.action(commands)

    for index, (arm, target) in enumerate(targets.items()):
        wired = action[protocol.arm_channel(protocol.ROBOT_COMMAND, arm)]
        assert wired[protocol.COMMAND_TYPE] == protocol.JOINT_POS
        np.testing.assert_array_equal(wired[protocol.COMMAND_JOINT_POS], target)
        assert action[protocol.arm_channel(protocol.TARGET_GRIP, arm)] == float(index)


def test_the_exterior_pair_set_binds_both_exteriors_and_the_wrist():
    wire = robolab_cfg._droid_camera_dict(robolab_keys.camera_set(robolab_keys.DROID, robolab_keys.EXTERIOR_PAIR_WRIST))

    assert wire == {
        keys.EXTERIOR_IMAGE: robolab_keys.EXTERIOR_CAMERA,
        keys.EXTERIOR_IMAGE_2: robolab_keys.EXTERIOR_2_CAMERA,
        keys.WRIST_IMAGE: robolab_keys.WRIST_CAMERA,
    }
