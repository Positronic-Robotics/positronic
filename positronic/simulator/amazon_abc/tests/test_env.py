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

_ABC_DIR = Path(__file__).parents[1]
_SERVER_DIR = _ABC_DIR.parent / 'env_server'
_TIMESTEP, _DECIMATION = 0.002, 17
_PROMPT = 'put the plastic bottles in the bin'
_TOKEN = {
    mapping.TOKEN_TASK: 'put_plastic_bottles_in_bin',
    mapping.TOKEN_CAMERA_HEIGHT: 4,
    mapping.TOKEN_CAMERA_WIDTH: 6,
}


_AXES = ('1 0 0', '0 0 1', '0 1 0')


def _arm_xml(arm: str, y: float) -> str:
    """A chain of six hinges with the control site at its tip, mounted at ``y``."""
    links = ''.join(
        f'<body name="{arm}_link{i}" pos="0 0 0.1"><joint name="{arm}_joint{i}" axis="{_AXES[i % 3]}" '
        'range="-3 3"/><geom size="0.02" mass="0.1"/>'
        for i in range(1, mapping.ARM_JOINTS + 1)
    )
    tip = f'<site name="{arm}_grasp_site" pos="0 0 0.1"/>'
    return f'<body name="{arm}_base" pos="0 {y} 0">{links}{tip}{"</body>" * mapping.ARM_JOINTS}</body>'


_SCENE = (
    f'<mujoco><compiler angle="radian"/><option timestep="{_TIMESTEP}"/><worldbody>'
    f'{_arm_xml("left", 0.3)}{_arm_xml("right", -0.3)}</worldbody></mujoco>'
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
        self.truncated = False
        self.task_eval: dict[str, Any] = {mapping.ABC_EVAL_EVER_SUCCESS: False}
        self.prompt = _PROMPT
        self.actions: list[np.ndarray] = []

    def forget_arm_state(self) -> None:
        pass

    def reset(self, seed: int | None = None, randomize: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._obs(), {}

    def evaluate_task(self) -> Any:
        return types.SimpleNamespace(to_info=lambda squeeze: {'num_bottles_in_bin': 0})

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        # ABC's own step: it ends a trial at its first success when the server asks it to.
        terminated = self.terminate_on_success and self.task_eval[mapping.ABC_EVAL_EVER_SUCCESS]
        return self._obs(), 0.0, terminated, self.truncated, {mapping.ABC_INFO_TASK_EVAL: self.task_eval}

    def close(self) -> None:
        pass

    def _obs(self) -> dict[str, Any]:
        images = {camera: np.zeros((3, 4, 6), dtype=np.uint8) for camera in CAMERAS.values()}
        return {
            mapping.ABC_OBS_STATE: np.zeros(len(mapping.ARMS) * (mapping.ARM_JOINTS + 1)),
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

    def make_env(terminate_on_success: bool = False, max_episode_steps: int | None = None, **_: Any) -> _FakeScene:
        scene.terminate_on_success = terminate_on_success
        scene.max_episode_steps = max_episode_steps
        return scene

    abc_sim.__dict__.update(get_task_spec=lambda name: task, list_task_specs=lambda: [task], make_env=make_env)
    monkeypatch.setitem(sys.modules, 'abc_sim', abc_sim)
    for name, directory in (('protocol', _SERVER_DIR), ('arm_action', _SERVER_DIR), ('server', _SERVER_DIR)):
        monkeypatch.setitem(sys.modules, name, _load(name, directory / f'{name}.py'))
    monkeypatch.setitem(sys.modules, 'mapping', _load('mapping', _ABC_DIR / 'mapping.py'))
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
    assert protocol.arm_channel(mapping.OBS_JOINT_POS, 'right') in slot[protocol.FRAME_OBS]
    assert frame[protocol.FRAME_META] == {mapping.META_TASK: _PROMPT}
    assert frame[protocol.FRAME_CONTROL_DT] == pytest.approx(_TIMESTEP * _DECIMATION)


_EVER_SUCCESS, _EVER_FAILED = mapping.ABC_EVAL_EVER_SUCCESS, mapping.ABC_EVAL_EVER_FAILED


@pytest.mark.parametrize(
    ('task_eval', 'truncated', 'done', 'success'),
    [
        ({_EVER_SUCCESS: False}, False, False, False),
        ({_EVER_SUCCESS: True}, False, True, True),
        ({_EVER_SUCCESS: False}, True, True, False),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, False, False, True),
        ({_EVER_SUCCESS: True, _EVER_FAILED: False}, True, True, True),
        ({_EVER_SUCCESS: False, _EVER_FAILED: True}, False, True, False),
    ],
    ids=['running', 'succeeded', 'out_of_time', 'still_held', 'held_to_the_limit', 'dropped'],
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


def test_abc_truncates_a_trial_at_the_horizon_of_its_own_eval(env, scene):
    env.reset(_TOKEN)

    # 236 chunks of 15 actions, as ABC's sim eval runs; a maintenance task succeeds only when ABC truncates there.
    assert scene.max_episode_steps == 3540


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
