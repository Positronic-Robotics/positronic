"""The frames the ABC env server answers, checked against a stand-in for ``abc_sim`` in positronic's interpreter."""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from positronic.eval import keys as eval_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import CAMERAS, AbcAdapter
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.tests.mujoco_chain import hinge_chain_xml

_ABC_DIR = Path(__file__).parents[1]
_SERVER_DIR = _ABC_DIR.parent / 'env_server'
_APERTURE_PATH = _ABC_DIR.parents[1] / 'drivers' / 'gripper' / 'aperture.py'
_TIMESTEP, _DECIMATION = 0.002, 17
_PROMPT = 'put the plastic bottles in the bin'
_TOKEN = {
    mapping.TOKEN_TASK: 'put_plastic_bottles_in_bin',
    mapping.TOKEN_CAMERA_HEIGHT: 4,
    mapping.TOKEN_CAMERA_WIDTH: 6,
}


_SCENE = (
    f'<mujoco><compiler angle="radian"/><option timestep="{_TIMESTEP}"/><worldbody>'
    f'{hinge_chain_xml("left", 0.3, mapping.ARM_JOINTS)}{hinge_chain_xml("right", -0.3, mapping.ARM_JOINTS)}'
    '<body name="gate" pos="0.1 0.05 1.6" quat="0.9 0.1 0.3 0.2"><camera name="top" quat="0 1 0 0" fovy="70"/></body>'
    '</worldbody></mujoco>'
)


class _FakeScene:
    """The part of ABC's gym env that the server drives, with the verdict a test sets."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        self._control_decimation = _DECIMATION
        self.robot_names = list(mapping.ARMS)
        self.terminate_on_success = False
        self.max_episode_steps: int | None = None
        self.task: str | None = None
        self.truncated = False
        # None stands for a task that ships no evaluator, which ABC reports as no metrics at all.
        self.task_eval: dict[str, Any] | None = {mapping.ABC_EVAL_EVER_SUCCESS: False}
        self.prompt = _PROMPT
        # Whether a step of a scored task omits its metrics, as a broken ABC would.
        self.drops_step_metrics = False
        self.actions: list[np.ndarray] = []
        # How many builds raise as ABC's unseeded build-time draw raises when it cannot lay out a scene.
        self.failed_layouts = 0
        self.builds = 0
        # The gripper width ABC reports per arm, in i2rt's aperture (1 open).
        self.apertures = dict.fromkeys(mapping.ARMS, 0.0)
        self.config = types.SimpleNamespace(
            robots={arm: types.SimpleNamespace(init_q=[0.5] * (mapping.ARM_JOINTS + 1)) for arm in mapping.ARMS}
        )

    def forget_arm_state(self) -> None:
        pass

    def reset(self, seed: int | None = None, randomize: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._obs(), {}

    def get_obs(self) -> dict[str, Any]:
        return self._obs()

    def evaluate_task(self) -> Any:
        if self.task_eval is None:
            return None
        return types.SimpleNamespace(to_info=lambda squeeze: {'num_bottles_in_bin': 0})

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        # ABC's own step: it ends a trial at its first success when the server asks it to.
        if self.task_eval is None or self.drops_step_metrics:
            return self._obs(), 0.0, False, self.truncated, {}
        terminated = self.terminate_on_success and self.task_eval[mapping.ABC_EVAL_EVER_SUCCESS]
        return self._obs(), 0.0, terminated, self.truncated, {mapping.ABC_INFO_TASK_EVAL: self.task_eval}

    def close(self) -> None:
        pass

    def _obs(self) -> dict[str, Any]:
        images = {camera: np.zeros((3, 4, 6), dtype=np.uint8) for camera in CAMERAS.values()}
        return {
            mapping.ABC_OBS_STATE: np.concatenate([
                np.append(np.zeros(mapping.ARM_JOINTS), self.apertures[arm]) for arm in self.robot_names
            ]),
            mapping.ABC_OBS_IMAGES: images,
            mapping.ABC_OBS_PROMPT: self.prompt,
        }


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene() -> _FakeScene:
    return _FakeScene()


@pytest.fixture
def env(monkeypatch, scene: _FakeScene) -> Any:
    """An ``AbcEnv`` loaded as the server's interpreter loads it, with ``abc_sim`` answering ``scene``."""
    abc_sim = types.ModuleType('abc_sim')
    task = types.SimpleNamespace(name=_TOKEN[mapping.TOKEN_TASK], env_task='bottles', prompt=_PROMPT)

    def make_env(
        task: str, terminate_on_success: bool = False, max_episode_steps: int | None = None, **_: Any
    ) -> _FakeScene:
        scene.builds += 1
        if scene.failed_layouts > 0:
            scene.failed_layouts -= 1
            raise RuntimeError('put_relative could not place non-overlapping objects')
        scene.task = task
        scene.terminate_on_success = terminate_on_success
        scene.max_episode_steps = max_episode_steps
        return scene

    abc_sim.__dict__.update(get_task_spec=lambda name: task, list_task_specs=lambda: [task], make_env=make_env)
    monkeypatch.setitem(sys.modules, 'abc_sim', abc_sim)
    for name in ('protocol', 'arm_action', 'mujoco_state', 'server'):
        monkeypatch.setitem(sys.modules, name, _load(name, _SERVER_DIR / f'{name}.py'))
    monkeypatch.setitem(sys.modules, 'mapping', _load('mapping', _ABC_DIR / 'mapping.py'))
    monkeypatch.setitem(sys.modules, 'aperture', _load('aperture', _APERTURE_PATH))
    return _load('abc_env', _ABC_DIR / 'env.py').AbcEnv()


def _hold_both() -> dict[str, Any]:
    action: dict[str, Any] = {}
    for arm in mapping.ARMS:
        action[protocol.arm_channel(protocol.ROBOT_COMMAND, arm)] = {protocol.COMMAND_TYPE: protocol.HOLD}
        action[protocol.arm_channel(protocol.TARGET_GRIP, arm)] = 0.0
    return action


def test_a_reset_answers_one_slot_beside_the_scene_meta_and_the_control_period(env):
    frame = env.reset(_TOKEN)

    assert set(frame) == {protocol.SLOTS, protocol.FRAME_META, protocol.FRAME_ROBOT_META, protocol.FRAME_CONTROL_DT}
    (slot,) = frame[protocol.SLOTS]
    assert protocol.arm_channel(protocol.OBS_JOINT_POS, 'right') in slot[protocol.FRAME_OBS]
    assert frame[protocol.FRAME_META] == {mapping.META_TASK: _PROMPT}
    assert frame[protocol.FRAME_CONTROL_DT] == pytest.approx(_TIMESTEP * _DECIMATION)


def test_a_reset_places_the_top_camera_at_the_molmospaces_pose_above_the_arm_mounts(env, scene):
    env.reset(_TOKEN)

    camera = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top')
    np.testing.assert_allclose(scene.data.cam_xpos[camera], [0.0, 0.0, 0.86], atol=1e-9)
    rotation = scene.data.cam_xmat[camera].reshape(3, 3)
    # The camera looks along its -z: forward along the arms' +x and down, with the image's top away from the arms.
    np.testing.assert_allclose(-rotation[:, 2], [0.4603, 0.0, -0.8878], atol=1e-3)
    np.testing.assert_allclose(rotation[:, 1], [0.8878, 0.0, 0.4603], atol=1e-3)
    assert scene.model.cam_fovy[camera] == 58.0


def test_the_arms_start_from_the_median_first_state_of_the_molmoact2_yam_dataset(env, scene):
    env.reset(_TOKEN)

    assert {arm: robot.init_q for arm, robot in scene.config.robots.items()} == {
        'left': [-0.002, 0.013, 0.014, -0.064, 0.028, -0.055, 0.992],
        'right': [0.039, 0.013, 0.008, -0.070, -0.057, 0.018, 0.993],
    }


_EVER_SUCCESS, _EVER_FAILED = mapping.ABC_EVAL_EVER_SUCCESS, mapping.ABC_EVAL_EVER_FAILED


# Each kind of ABC evaluator, with the metrics it reports at the pin:
# - most tasks latch `ever_success` at their first success;
# - ball_tray_balancing reports `ever_failed`, and `ever_success` as `not ever_failed`;
# - count_into_opaque_box reports `ever_failed` as always false, and `ever_success` as the count at that step.
@pytest.mark.parametrize(
    ('task_eval', 'truncated', 'done', 'success'),
    [
        ({_EVER_SUCCESS: False}, False, False, False),
        ({_EVER_SUCCESS: True}, False, True, True),
        ({_EVER_SUCCESS: False}, True, True, False),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, False, False, True),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, True, True, True),
        ({_EVER_SUCCESS: False, _EVER_FAILED: True}, False, True, False),
        ({_EVER_SUCCESS: False, _EVER_FAILED: False}, False, False, False),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, False, False, True),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, True, True, True),
        ({_EVER_SUCCESS: False, _EVER_FAILED: False}, True, True, False),
    ],
    ids=[
        'running',
        'succeeded',
        'out_of_time',
        'ball_still_held',
        'ball_held_to_the_limit',
        'ball_dropped',
        'count_wrong_while_running',
        'count_right_while_running',
        'count_right_at_the_limit',
        'count_wrong_at_the_limit',
    ],
)
def test_a_step_ends_the_trial_and_gives_its_verdict_as_abcs_own_eval_does(
    env, scene, task_eval: dict[str, bool], truncated: bool, done: bool, success: bool
):
    env.reset(_TOKEN)
    scene.task_eval, scene.truncated = task_eval, truncated

    frame = env.step({0: _hold_both()})

    assert set(frame) == {protocol.SLOTS, protocol.FRAME_CONTROL_DT}
    (slot,) = frame[protocol.SLOTS]
    assert set(slot) == {protocol.FRAME_OBS, protocol.FRAME_DONE, protocol.FRAME_SUCCESS, protocol.FRAME_META}
    assert (slot[protocol.FRAME_DONE], slot[protocol.FRAME_SUCCESS]) == (done, success)


def test_the_env_is_built_for_the_task_so_abc_attaches_that_tasks_evaluator(env, scene):
    env.reset(_TOKEN)

    # Several tasks share one scene (`env_task`): ABC picks the evaluator and the prompt by the task name.
    assert scene.task == _TOKEN[mapping.TOKEN_TASK]


def test_a_build_whose_unseeded_layout_fails_is_built_again(env, scene):
    scene.failed_layouts = 1

    frame = env.reset(_TOKEN)

    # The seeded reset draws the trial's scene, so the scene the failed build drew was never the trial's.
    assert (scene.builds, len(frame[protocol.SLOTS])) == (2, 1)


def test_a_build_that_keeps_failing_raises_after_a_bounded_number_of_tries(env, scene):
    scene.failed_layouts = 100

    with pytest.raises(RuntimeError, match='could not place'):
        env.reset(_TOKEN)
    assert scene.builds < 100


def test_abc_truncates_a_trial_at_the_horizon_of_its_own_eval(env, scene):
    env.reset(_TOKEN)

    # 236 chunks of 15 actions, as ABC's sim eval runs; a maintenance task succeeds only when ABC truncates there.
    assert scene.max_episode_steps == 3540


@pytest.mark.parametrize(('truncated', 'done'), [(False, False), (True, True)], ids=['running', 'at_the_horizon'])
def test_a_task_without_an_evaluator_runs_to_abcs_horizon_and_never_succeeds(env, scene, truncated: bool, done: bool):
    scene.task_eval = None
    env.reset(_TOKEN)
    scene.truncated = truncated

    (slot,) = env.step({0: _hold_both()})[protocol.SLOTS]

    assert (slot[protocol.FRAME_DONE], slot[protocol.FRAME_SUCCESS]) == (done, False)


def test_a_step_of_a_scored_task_that_reports_no_metrics_raises(env, scene):
    env.reset(_TOKEN)
    scene.drops_step_metrics = True

    # Run to its horizon, it would record a failure that hides the broken evaluator.
    with pytest.raises(RuntimeError, match='no task metrics'):
        env.step({0: _hold_both()})


def test_a_count_task_is_decided_by_its_count_at_the_limit_and_not_by_an_earlier_right_count(env, scene):
    env.reset(_TOKEN)
    adapter = AbcAdapter(CAMERAS)
    scene.task_eval = {_EVER_SUCCESS: True, _EVER_FAILED: False}
    assert adapter.terminal(protocol.slot_frame(env.step({0: _hold_both()}), 0)) is None

    scene.task_eval, scene.truncated = {_EVER_SUCCESS: False, _EVER_FAILED: False}, True
    result = protocol.slot_frame(env.step({0: _hold_both()}), 0)

    assert adapter.terminal(result) == {eval_keys.SUCCESS: False}


def test_a_step_of_a_scored_task_whose_metrics_carry_no_verdict_raises(env, scene):
    env.reset(_TOKEN)
    scene.task_eval = {'num_bottles_in_bin': 0}

    # ABC's own eval reads the verdict by its key; a default would record a failure that hides the broken evaluator.
    with pytest.raises(KeyError, match=_EVER_SUCCESS):
        env.step({0: _hold_both()})


def test_a_step_reports_the_prompt_the_task_moved_to(env, scene):
    env.reset(_TOKEN)
    scene.prompt = 'find the red mug'

    result = protocol.slot_frame(env.step({0: _hold_both()}), 0)

    assert result[protocol.FRAME_META] == {mapping.META_TASK: 'find the red mug'}


def test_the_client_reads_the_envs_own_verdict_from_its_slot(env, scene):
    env.reset(_TOKEN)
    scene.task_eval = {_EVER_SUCCESS: True}

    result = protocol.slot_frame(env.step({0: _hold_both()}), 0)

    assert AbcAdapter(CAMERAS).terminal(result) == {eval_keys.SUCCESS: True}


def test_a_pose_out_of_reach_holds_that_arm_and_moves_the_other(env, scene):
    env.reset(_TOKEN)
    action = _hold_both()
    far = np.concatenate([[10.0, 0.0, 0.0], np.eye(3).reshape(-1)])
    action[protocol.arm_channel(protocol.ROBOT_COMMAND, 'left')] = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN,
        protocol.COMMAND_POSE: far,
    }
    target = np.full(mapping.ARM_JOINTS, 0.2)
    action[protocol.arm_channel(protocol.ROBOT_COMMAND, 'right')] = {
        protocol.COMMAND_TYPE: protocol.JOINT_POS,
        protocol.COMMAND_JOINT_POS: target,
    }

    env.step({0: action})

    left, right = np.split(scene.actions[-1], len(mapping.ARMS))
    np.testing.assert_allclose(left[: mapping.ARM_JOINTS], np.zeros(mapping.ARM_JOINTS))
    np.testing.assert_allclose(right[: mapping.ARM_JOINTS], target)


def test_a_pose_in_reach_moves_the_arm_to_it(env, scene):
    env.reset(_TOKEN)
    site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, mapping.CONTROL_SITE.format(arm='left'))
    scene.data.qpos[: mapping.ARM_JOINTS] = 0.3
    mujoco.mj_kinematics(scene.model, scene.data)
    pose = np.concatenate([scene.data.site_xpos[site], scene.data.site_xmat[site]])
    scene.data.qpos[: mapping.ARM_JOINTS] = 0.25
    action = _hold_both()
    action[protocol.arm_channel(protocol.ROBOT_COMMAND, 'left')] = {
        protocol.COMMAND_TYPE: protocol.CARTESIAN,
        protocol.COMMAND_POSE: pose,
    }

    env.step({0: action})

    left, _right = np.split(scene.actions[-1], len(mapping.ARMS))
    np.testing.assert_allclose(left[: mapping.ARM_JOINTS], np.full(mapping.ARM_JOINTS, 0.3), atol=1e-2)


def _left_command(command: dict[str, Any], grip: float = 0.0) -> dict[str, Any]:
    action = _hold_both()
    action[protocol.arm_channel(protocol.ROBOT_COMMAND, 'left')] = command
    action[protocol.arm_channel(protocol.TARGET_GRIP, 'left')] = grip
    return action


def _joint_position(joints: np.ndarray) -> dict:
    return {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: joints}


def _joint_delta(delta: np.ndarray) -> dict:
    return {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: delta}


@pytest.mark.parametrize(
    'command',
    [
        _joint_position(np.full(mapping.ARM_JOINTS, np.nan)),
        _joint_position(np.full(mapping.ARM_JOINTS, np.inf)),
        _joint_position(np.zeros(mapping.ARM_JOINTS - 1)),
        _joint_delta(np.full(mapping.ARM_JOINTS, np.nan)),
        _joint_delta(np.zeros(mapping.ARM_JOINTS - 1)),
    ],
    ids=['nan', 'infinite', 'too_few_joints', 'delta_nan', 'delta_too_few_joints'],
)
def test_a_joint_command_that_is_no_finite_pose_of_the_arm_holds_that_arm(env, scene, command: dict):
    env.reset(_TOKEN)
    scene.data.qpos[: mapping.ARM_JOINTS] = 0.25
    action = _left_command(command)
    target = np.full(mapping.ARM_JOINTS, 0.2)
    action[protocol.arm_channel(protocol.ROBOT_COMMAND, 'right')] = {
        protocol.COMMAND_TYPE: protocol.JOINT_POS,
        protocol.COMMAND_JOINT_POS: target,
    }

    env.step({0: action})

    left, right = np.split(scene.actions[-1], len(mapping.ARMS))
    np.testing.assert_allclose(left[: mapping.ARM_JOINTS], np.full(mapping.ARM_JOINTS, 0.25))
    np.testing.assert_allclose(right[: mapping.ARM_JOINTS], target)


def test_a_finite_joint_target_at_the_joint_limit_is_stepped(env, scene):
    env.reset(_TOKEN)
    limit = np.full(mapping.ARM_JOINTS, 3.0)

    env.step({0: _left_command({protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: limit})})

    left, _right = np.split(scene.actions[-1], len(mapping.ARMS))
    np.testing.assert_allclose(left[: mapping.ARM_JOINTS], limit)


def test_a_grip_target_that_is_not_finite_holds_the_grippers_measured_width(env, scene):
    scene.apertures = {'left': 0.6, 'right': 0.1}
    env.reset(_TOKEN)

    env.step({0: _left_command({protocol.COMMAND_TYPE: protocol.HOLD}, grip=np.nan)})

    left, right = np.split(scene.actions[-1], len(mapping.ARMS))
    # i2rt's aperture: the grip target 0.0 of the right arm is a fully open gripper.
    assert (left[-1], right[-1]) == (pytest.approx(0.6), pytest.approx(1.0))
