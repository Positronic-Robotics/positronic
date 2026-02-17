import numpy as np
import pytest

import positronic.drivers.roboarm.command as cmd_module
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.geom import Rotation
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, RelativeTargetPositionAction
from positronic.policy.base import DecodedEncodedPolicy, Policy
from positronic.policy.observation import SimpleObservationEncoder


def test_observation_encode_images_and_state_shapes():
    # Image matches target size; ensures no resampling artifacts in assertions
    h, w = 6, 8
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    enc = SimpleObservationEncoder(
        state={'observation.state': ['a', 'b']}, images={'observation.images.left': ('left.image', (w, h))}
    )
    obs = enc.encode({'left.image': img, 'a': [1, 2], 'b': 3.0})

    assert 'observation.images.left' in obs and 'observation.state' in obs
    left = obs['observation.images.left']
    state = obs['observation.state']

    assert left.shape == (h, w, 3)
    assert left.dtype == np.uint8
    assert np.all(left == 255)

    assert state.shape == (3,)
    np.testing.assert_allclose(state, np.array([1, 2, 3], dtype=np.float32))


def test_observation_encode_missing_or_bad_images_raise():
    enc = SimpleObservationEncoder(
        state={'observation.state': []}, images={'observation.images.left': ('left.image', (8, 6))}
    )
    with pytest.raises(KeyError):  # Missing key
        enc.encode({})

    with pytest.raises(ValueError):  # Wrong shape
        enc.encode({'left.image': np.zeros((8, 8), dtype=np.uint8)})


def test_observation_encode_missing_state_inputs_raise():
    enc = SimpleObservationEncoder(state={'observation.state': ['missing']}, images={})
    with pytest.raises(KeyError):
        enc.encode({})


def test_observation_encode_task_field_parameter():
    """Test that task_field parameter controls which field task string is stored in."""
    # Default: task_field='task'
    enc_task = SimpleObservationEncoder(state={'observation.state': ['a']}, images={})
    obs_task = enc_task.encode({'a': 1.0, 'task': 'test_task'})
    assert obs_task['task'] == 'test_task' and 'prompt' not in obs_task

    # OpenPI: task_field='prompt'
    enc_prompt = SimpleObservationEncoder(state={'observation.state': ['a']}, images={}, task_field='prompt')
    obs_prompt = enc_prompt.encode({'a': 1.0, 'task': 'test_task'})
    assert obs_prompt['prompt'] == 'test_task' and 'task' not in obs_prompt

    # Disabled: task_field=None
    enc_none = SimpleObservationEncoder(state={'observation.state': ['a']}, images={}, task_field=None)
    obs_none = enc_none.encode({'a': 1.0, 'task': 'test_task'})
    assert 'task' not in obs_none and 'prompt' not in obs_none


def test_absolute_position_action_encode_decode_quat():
    # Identity rotation, known translation/grip
    ts = [1000, 2000]
    q = [Rotation.identity for _ in ts]
    t = [np.array([0.1, -0.2, 0.3], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    pose = [np.concatenate([t[i], q[i].as_quat]).astype(np.float32) for i in range(len(ts))]

    ep = EpisodeContainer({'robot_commands.pose': DummySignal(ts, pose), 'target_grip': DummySignal(ts, g)})

    act = AbsolutePositionAction('robot_commands.pose', 'target_grip', Rotation.Representation.QUAT, action_fps=30.0)
    sig = act.encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    decoded = act.decode({'action': vec}, inputs={})
    command = cmd_module.from_wire(decoded['robot_command'])
    target_grip = decoded['target_grip']
    assert isinstance(command, cmd_module.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, t[0], atol=1e-6)
    np.testing.assert_allclose(command.pose.rotation.as_quat, q[0].as_quat, atol=1e-6)
    assert np.isclose(target_grip, g[0])


def test_relative_target_position_action_encode_decode_quat():
    ts = [1000]
    q_cur = [Rotation.identity]
    # 90 deg about X: w=cos(45)=sqrt(2)/2, x=sin(45)=sqrt(2)/2
    q_tgt = [Rotation.from_quat([np.sqrt(0.5), np.sqrt(0.5), 0, 0])]
    t_cur = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    t_tgt = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
    g_tgt = [0.7]

    cur_pose = [np.concatenate([t_cur[0], q_cur[0].as_quat]).astype(np.float32)]
    tgt_pose = [np.concatenate([t_tgt[0], q_tgt[0].as_quat]).astype(np.float32)]

    ep = EpisodeContainer({
        'robot_state.ee_pose': DummySignal(ts, cur_pose),
        'robot_commands.pose': DummySignal(ts, tgt_pose),
        'target_grip': DummySignal(ts, g_tgt),
    })

    act = RelativeTargetPositionAction(Rotation.Representation.QUAT, action_fps=30.0)
    vec = list(act.encode_episode(ep))[0][0]
    # First 4 should match target quaternion since current is identity
    np.testing.assert_allclose(vec[:4], q_tgt[0].as_quat, atol=1e-6)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec[4:7], t_tgt[0] - t_cur[0], atol=1e-6)
    assert np.isclose(vec[7], g_tgt[0])

    decoded = act.decode({'action': vec}, inputs={'robot_state.ee_pose': np.concatenate([t_cur[0], q_cur[0].as_quat])})
    command = cmd_module.from_wire(decoded['robot_command'])
    target_grip = decoded['target_grip']
    assert isinstance(command, cmd_module.CartesianPosition)
    # Decode applies diff to current translation
    np.testing.assert_allclose(command.pose.translation, t_cur[0] + vec[4:7], atol=1e-6)
    assert np.isclose(target_grip, g_tgt[0])


def test_absolute_joints_action_encode_decode():
    # Known joint positions and grip
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    ep = EpisodeContainer({'robot_commands.joints': DummySignal(ts, joints), 'target_grip': DummySignal(ts, g)})

    act = AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7, action_fps=30.0)
    sig = act.encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 7 joints + 1 grip
    assert vec.dtype == np.float32

    decoded = act.decode(vec, {})
    command = cmd_module.from_wire(decoded['robot_command'])
    target_grip = decoded['target_grip']
    assert isinstance(command, cmd_module.JointPosition)
    np.testing.assert_allclose(command.positions, joints[0], atol=1e-6)
    assert np.isclose(target_grip, g[0])


class _ChunkPolicy(Policy):
    def __init__(self, actions: list[dict]):
        self._actions = actions

    def select_action(self, obs):
        return list(self._actions)

    def reset(self):
        pass

    @property
    def meta(self):
        return {}

    def close(self):
        pass


def test_action_horizon_sec_truncates_chunk():
    actions = [{'v': i} for i in range(10)]
    # action_horizon_sec=0.1s at action_fps=30 -> 3 actions
    policy = DecodedEncodedPolicy(_ChunkPolicy(actions), action_horizon_sec=0.1, action_fps=30.0)
    result = policy.select_action({})
    assert [r['v'] for r in result] == [0, 1, 2]


def test_action_horizon_sec_none_returns_full_chunk():
    actions = [{'v': i} for i in range(5)]
    policy = DecodedEncodedPolicy(_ChunkPolicy(actions), action_fps=30.0)
    result = policy.select_action({})
    assert len(result) == 5


def test_action_horizon_sec_larger_than_chunk():
    actions = [{'v': i} for i in range(3)]
    # action_horizon_sec=10s at action_fps=10 -> 100 actions max, but only 3 available
    policy = DecodedEncodedPolicy(_ChunkPolicy(actions), action_horizon_sec=10.0, action_fps=10.0)
    result = policy.select_action({})
    assert len(result) == 3


def test_timestamps_embedded_in_actions():
    actions = [{'v': i} for i in range(4)]
    policy = DecodedEncodedPolicy(_ChunkPolicy(actions), action_fps=10.0)
    result = policy.select_action({})
    assert len(result) == 4
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * 0.1)


def test_action_horizon_sec_seconds_truncates():
    actions = [{'v': i} for i in range(100)]
    # 0.1s at 30fps -> 3 actions
    policy = DecodedEncodedPolicy(_ChunkPolicy(actions), action_horizon_sec=0.1, action_fps=30.0)
    result = policy.select_action({})
    assert len(result) == 3
    dt = 1.0 / 30.0
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * dt)


def test_single_action_has_zero_timestamp():
    class _SinglePolicy(Policy):
        def select_action(self, obs):
            return {'v': 42}

        def reset(self):
            pass

        @property
        def meta(self):
            return {}

        def close(self):
            pass

    policy = DecodedEncodedPolicy(_SinglePolicy(), action_fps=15.0)
    result = policy.select_action({})
    assert isinstance(result, dict)
    assert result['timestamp'] == 0.0
    assert result['v'] == 42
