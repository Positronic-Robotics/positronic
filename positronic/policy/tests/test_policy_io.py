import numpy as np
import pytest

import positronic.drivers.roboarm.command as cmd_module
from positronic.cfg.codecs import compose
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.geom import Rotation
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, RelativePositionAction
from positronic.policy.base import Policy
from positronic.policy.codec import ActionTiming, BinarizeGripInference, BinarizeGripTraining, Codec
from positronic.policy.observation import ObservationCodec


def test_observation_encode_images_and_state_shapes():
    # Image matches target size; ensures no resampling artifacts in assertions
    h, w = 6, 8
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    enc = ObservationCodec(
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
    enc = ObservationCodec(state={'observation.state': []}, images={'observation.images.left': ('left.image', (8, 6))})
    with pytest.raises(KeyError):  # Missing key
        enc.encode({})

    with pytest.raises(ValueError):  # Wrong shape
        enc.encode({'left.image': np.zeros((8, 8), dtype=np.uint8)})


def test_observation_encode_missing_state_inputs_raise():
    enc = ObservationCodec(state={'observation.state': ['missing']}, images={})
    with pytest.raises(KeyError):
        enc.encode({})


def test_observation_encode_task_field_parameter():
    """Test that task_field parameter controls which field task string is stored in."""
    # Default: task_field='task'
    enc_task = ObservationCodec(state={'observation.state': ['a']}, images={})
    obs_task = enc_task.encode({'a': 1.0, 'task': 'test_task'})
    assert obs_task['task'] == 'test_task' and 'prompt' not in obs_task

    # OpenPI: task_field='prompt'
    enc_prompt = ObservationCodec(state={'observation.state': ['a']}, images={}, task_field='prompt')
    obs_prompt = enc_prompt.encode({'a': 1.0, 'task': 'test_task'})
    assert obs_prompt['prompt'] == 'test_task' and 'task' not in obs_prompt

    # Disabled: task_field=None
    enc_none = ObservationCodec(state={'observation.state': ['a']}, images={}, task_field=None)
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

    act = AbsolutePositionAction('robot_commands.pose', 'target_grip', Rotation.Representation.QUAT)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({'action': vec}, context={})
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

    act = RelativePositionAction(Rotation.Representation.QUAT)
    vec = list(act._encode_episode(ep))[0][0]
    # First 4 should match target quaternion since current is identity
    np.testing.assert_allclose(vec[:4], q_tgt[0].as_quat, atol=1e-6)
    assert vec.dtype == np.float32
    np.testing.assert_allclose(vec[4:7], t_tgt[0] - t_cur[0], atol=1e-6)
    assert np.isclose(vec[7], g_tgt[0])

    decoded = act._decode_single(
        {'action': vec}, context={'robot_state.ee_pose': np.concatenate([t_cur[0], q_cur[0].as_quat])}
    )
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

    act = AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 7 joints + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({'action': vec}, context={})
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


class _SinglePolicy(Policy):
    def select_action(self, obs):
        return {'v': 42}


class _PassthroughCodec(Codec):
    def __init__(self, tag):
        self._tag = tag

    def encode(self, data):
        data[f'encoded_by_{self._tag}'] = True
        return data


class _MetaPolicy(Policy):
    def select_action(self, obs):
        return {}

    @property
    def meta(self):
        return {'base_key': 'base_value'}


def test_action_horizon_sec_truncates_chunk():
    actions = [{'v': i} for i in range(10)]
    # action_horizon_sec=0.1s at action_fps=30 -> 3 actions
    codec = ActionTiming(fps=30.0, horizon_sec=0.1)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.select_action({})
    assert [r['v'] for r in result] == [0, 1, 2]


def test_action_horizon_sec_none_returns_full_chunk():
    actions = [{'v': i} for i in range(5)]
    codec = ActionTiming(fps=30.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.select_action({})
    assert len(result) == 5


def test_action_horizon_sec_larger_than_chunk():
    actions = [{'v': i} for i in range(3)]
    # action_horizon_sec=10s at action_fps=10 -> 100 actions max, but only 3 available
    codec = ActionTiming(fps=10.0, horizon_sec=10.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.select_action({})
    assert len(result) == 3


def test_timestamps_embedded_in_actions():
    actions = [{'v': i} for i in range(4)]
    codec = ActionTiming(fps=10.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.select_action({})
    assert len(result) == 4
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * 0.1)


def test_action_horizon_sec_seconds_truncates():
    actions = [{'v': i} for i in range(100)]
    # 0.1s at 30fps -> 3 actions
    codec = ActionTiming(fps=30.0, horizon_sec=0.1)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.select_action({})
    assert len(result) == 3
    dt = 1.0 / 30.0
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * dt)


def test_single_action_has_zero_timestamp():
    codec = ActionTiming(fps=15.0)
    policy = codec.wrap(_SinglePolicy())
    result = policy.select_action({})
    assert isinstance(result, dict)
    assert result['timestamp'] == 0.0
    assert result['v'] == 42


def test_codec_composition():
    """Test that codecs compose correctly via |."""
    left = _PassthroughCodec('left')
    right = _PassthroughCodec('right')
    composed = left | right

    result = composed.encode({})
    assert result['encoded_by_left'] is True
    assert result['encoded_by_right'] is True


def test_codec_wrap_meta_merges():
    """Test that wrapped policy meta merges base and codec meta."""
    codec = ActionTiming(fps=15.0, horizon_sec=1.0)
    policy = codec.wrap(_MetaPolicy())
    meta = policy.meta
    assert meta['base_key'] == 'base_value'
    assert meta['action_fps'] == 15.0
    assert meta['action_horizon_sec'] == 1.0


def test_timestamps_survive_action_decoder_composition():
    """Timestamps from ActionTiming must survive through composed action decoders."""
    action_codec = AbsolutePositionAction('robot_commands.pose', 'target_grip', Rotation.Representation.QUAT)
    timing = ActionTiming(fps=15.0, horizon_sec=1.0)
    composed = timing | action_codec

    # Build a raw action vector: 4 quat + 3 trans + 1 grip = 8
    raw_action = np.zeros(8, dtype=np.float32)
    raw_action[:4] = Rotation.identity.as_quat
    raw_action[4:7] = [0.1, 0.2, 0.3]
    raw_action[7] = 0.5

    raw_chunk = [{'action': raw_action} for _ in range(5)]
    decoded = composed.decode(raw_chunk)

    assert len(decoded) == 5
    for i, action in enumerate(decoded):
        assert 'robot_command' in action
        assert 'target_grip' in action
        assert 'timestamp' in action, f'Action {i} missing timestamp — stripped by action decoder'
        assert action['timestamp'] == pytest.approx(i / 15.0)


def test_composed_training_encoder_uses_parallel():
    """``timing | (obs & action)`` training encoder produces only derived keys, no originals."""
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    grip = [0.5, 0.6]
    img = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in ts]

    ep = EpisodeContainer({
        'robot_state.q': DummySignal(ts, joints),
        'grip': DummySignal(ts, grip),
        'robot_commands.joints': DummySignal(ts, joints),
        'target_grip': DummySignal(ts, grip),
        'image.wrist': DummySignal(ts, img),
        'image.exterior': DummySignal(ts, img),
        'task': 'test',
    })

    obs = ObservationCodec(
        state={'observation.state': {'robot_state.q': 7, 'grip': 1}},
        images={'observation.images.left': ('image.wrist', (4, 4))},
    )
    action = AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7)
    timing = ActionTiming(fps=15.0)
    composed = timing | (obs & action)

    encoder = composed.training_encoder
    result = encoder(ep)

    # Observation codec's derived keys
    assert 'observation.state' in result
    assert 'observation.images.left' in result

    # Action codec's derived key — must be accessible (reads target_grip from base episode)
    assert 'action' in result
    vec = list(result['action'])[0][0]
    assert vec.shape == (8,)
    np.testing.assert_allclose(vec[:7], joints[0], atol=1e-6)
    np.testing.assert_allclose(vec[7], grip[0], atol=1e-6)

    # Original episode keys should NOT appear (no Identity pass-through)
    assert 'target_grip' not in result
    assert 'robot_commands.joints' not in result

    # Meta should merge from all codecs
    assert encoder.meta.get('action_fps') == 15.0
    assert 'lerobot_features' in encoder.meta


def test_binarize_grip_inference():
    binarize = BinarizeGripInference()
    assert binarize._decode_single({'target_grip': 0.3}, None) == {'target_grip': 0.0}
    assert binarize._decode_single({'target_grip': 0.7}, None) == {'target_grip': 1.0}
    assert binarize._decode_single({'target_grip': 0.5}, None) == {'target_grip': 0.0}

    binarize_low = BinarizeGripInference(threshold=0.3)
    assert binarize_low._decode_single({'target_grip': 0.4}, None) == {'target_grip': 1.0}


def test_binarize_grip_training():
    ts = [1000, 2000]
    ep = EpisodeContainer({'grip': DummySignal(ts, [0.3, 0.8]), 'target_grip': DummySignal(ts, [0.7, 0.2])})

    binarize = BinarizeGripTraining(('grip', 'target_grip'))
    result = binarize.training_encoder(ep)
    grip_vals = [v for v, _ in result['grip']]
    tgt_vals = [v for v, _ in result['target_grip']]
    np.testing.assert_array_equal(grip_vals, [0.0, 1.0])
    np.testing.assert_array_equal(tgt_vals, [1.0, 0.0])


def test_binarize_grip_training_respects_threshold():
    ts = [1000]
    ep = EpisodeContainer({'grip': DummySignal(ts, [0.4]), 'target_grip': DummySignal(ts, [0.4])})

    keys = ('grip', 'target_grip')
    default = BinarizeGripTraining(keys)
    result = default.training_encoder(ep)
    assert list(result['grip'])[0][0] == pytest.approx(0.0)

    low = BinarizeGripTraining(keys, threshold=0.3)
    result = low.training_encoder(ep)
    assert list(result['grip'])[0][0] == pytest.approx(1.0)


def test_binarize_grip_training_composed_with_action_codec():
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({'robot_commands.joints': DummySignal(ts, joints), 'target_grip': DummySignal(ts, [0.7])})

    binarize = BinarizeGripTraining(('grip', 'target_grip'))
    action = AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7)
    composed = binarize | action

    result = composed.training_encoder(ep)
    vec = list(result['action'])[0][0]
    assert vec[-1] == pytest.approx(1.0)


def test_parallel_codec_encode_merges_outputs():
    """``obs & action`` encode produces only obs keys (action returns {})."""
    obs = ObservationCodec(state={'observation.state': {'a': 1}}, images={}, task_field=None)
    action = AbsolutePositionAction('x', 'y')
    composed = obs & action
    result = composed.encode({'a': 1.0})
    assert 'observation.state' in result
    # Action codec returns {} from encode — no passthrough leakage
    assert set(result.keys()) == {'observation.state'}


def test_parallel_codec_decode_merges_outputs():
    """``obs & action`` decode produces only action-decoded keys (obs returns {})."""
    obs = ObservationCodec(state={'observation.state': {'a': 1}}, images={})
    action = AbsolutePositionAction('x', 'y')
    composed = obs & action

    raw_action = np.zeros(8, dtype=np.float32)
    raw_action[:4] = Rotation.identity.as_quat  # valid quaternion
    raw_action[4:7] = [0.1, 0.2, 0.3]
    raw_action[7] = 0.5
    result = composed.decode({'action': raw_action})
    # Obs returns {} from decode, action returns decoded keys
    assert 'robot_command' in result
    assert 'target_grip' in result
    assert 'action' not in result


def test_sequential_into_parallel_training():
    """``binarize | (obs & action)`` — binarize modifies grip seen by both."""
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({
        'robot_state.q': DummySignal(ts, joints),
        'grip': DummySignal(ts, [0.7]),
        'robot_commands.joints': DummySignal(ts, joints),
        'target_grip': DummySignal(ts, [0.3]),
        'image.wrist': DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
        'image.exterior': DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
    })

    obs = ObservationCodec(
        state={'observation.state': {'robot_state.q': 7, 'grip': 1}},
        images={'observation.images.left': ('image.wrist', (4, 4))},
        task_field=None,
    )
    action = AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7)
    binarize = BinarizeGripTraining(('grip', 'target_grip'))
    composed = binarize | (obs & action)

    result = composed.training_encoder(ep)

    # Binarize runs first — grip (0.7 > 0.5 → 1.0), target_grip (0.3 ≤ 0.5 → 0.0)
    # Action encoder reads binarized target_grip
    vec = list(result['action'])[0][0]
    assert vec[-1] == pytest.approx(0.0)

    # Obs encoder reads binarized grip in observation.state
    state = list(result['observation.state'])[0][0]
    assert state[-1] == pytest.approx(1.0)


def test_compose_training_encoder_produces_only_derived_keys():
    """Composed codec training encoder must not leak original episode keys into the output."""
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    grip = [0.5, 0.6]
    img = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in ts]

    ep = EpisodeContainer({
        'robot_state.q': DummySignal(ts, joints),
        'grip': DummySignal(ts, grip),
        'robot_commands.joints': DummySignal(ts, joints),
        'target_grip': DummySignal(ts, grip),
        'image.wrist': DummySignal(ts, img),
        'image.exterior': DummySignal(ts, img),
        'task': 'test',
    })

    codec = compose(
        obs=ObservationCodec(
            state={'observation.state': {'robot_state.q': 7, 'grip': 1}},
            images={'observation.images.left': ('image.wrist', (4, 4))},
        ),
        action=AbsoluteJointsAction('robot_commands.joints', 'target_grip', num_joints=7),
    )

    result = codec.training_encoder(ep)

    # Derived keys present
    assert 'observation.state' in result
    assert 'action' in result

    # Original episode keys must NOT leak through — this fails if compose uses | instead of &
    assert 'target_grip' not in result
    assert 'robot_commands.joints' not in result
    assert 'robot_state.q' not in result
    assert 'grip' not in result


def test_operator_precedence():
    """``a | b & c`` binds as ``a | (b & c)`` — & has higher precedence than |."""
    a = _PassthroughCodec('a')
    b = _PassthroughCodec('b')
    c = _PassthroughCodec('c')

    composed = a | b & c
    result = composed.encode({})
    # a encodes first (sequential |), then b & c both see a's output (parallel &)
    assert result['encoded_by_a'] is True
    assert result['encoded_by_b'] is True
    assert result['encoded_by_c'] is True
