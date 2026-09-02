import numpy as np
import pytest

import positronic.drivers.roboarm.command as cmd_module
from positronic import keys as obs_keys
from positronic.cfg.codecs import compose
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.geom import Rotation
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction
from positronic.policy.base import Policy, Session
from positronic.policy.codec import (
    ActionHorizon,
    ActionTimestamp,
    ActionTiming,
    BinarizeGripInference,
    BinarizeGripTraining,
    Codec,
    FlipGrip,
)
from positronic.policy.observation import ObservationCodec
from positronic.policy.spec import from_spec


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


def test_observation_encode_task():
    enc = ObservationCodec(state={'observation.state': ['a']}, images={})
    obs = enc.encode({'a': 1.0, obs_keys.TASK: 'test_task'})
    assert obs[obs_keys.TASK] == 'test_task'

    obs_no_task = enc.encode({'a': 1.0})
    assert obs_keys.TASK not in obs_no_task


def test_observation_codec_spec_preserves_lowercase_task():
    enc = ObservationCodec(state={}, images={}, lowercase_task=True)
    rebuilt = from_spec(enc.to_spec())
    assert isinstance(rebuilt, ObservationCodec)
    obs = rebuilt.encode({obs_keys.TASK: 'MixedCase'})
    assert obs[obs_keys.TASK] == 'mixedcase'


def test_absolute_position_action_encode_decode_quat():
    # Identity rotation, known translation/grip
    ts = [1000, 2000]
    q = [Rotation.identity for _ in ts]
    t = [np.array([0.1, -0.2, 0.3], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    pose = [np.concatenate([t[i], q[i].as_quat]).astype(np.float32) for i in range(len(ts))]

    ep = EpisodeContainer({obs_keys.TARGET_EE_POSE: DummySignal(ts, pose), 'target_grip': DummySignal(ts, g)})

    act = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, 'target_grip', Rotation.Representation.QUAT)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({'action': vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded['target_grip']
    assert isinstance(command, cmd_module.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, t[0], atol=1e-6)
    np.testing.assert_allclose(command.pose.rotation.as_quat, q[0].as_quat, atol=1e-6)
    assert np.isclose(target_grip, g[0])


def test_absolute_joints_action_encode_decode():
    # Known joint positions and grip
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    ep = EpisodeContainer({obs_keys.TARGET_JOINTS: DummySignal(ts, joints), 'target_grip': DummySignal(ts, g)})

    act = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, 'target_grip', num_joints=7)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 7 joints + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({'action': vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded['target_grip']
    assert isinstance(command, cmd_module.JointPosition)
    np.testing.assert_allclose(command.positions, joints[0], atol=1e-6)
    assert np.isclose(target_grip, g[0])


class _FixedSession(Session):
    def __init__(self, result):
        self._result = result

    def __call__(self, obs, time_ns):
        return self._result


class _ChunkPolicy(Policy):
    def __init__(self, actions: list[dict]):
        self._actions = actions

    def new_session(self, context=None, rt=None):
        return _FixedSession(list(self._actions))


class _SinglePolicy(Policy):
    def new_session(self, context=None, rt=None):
        return _FixedSession({'v': 42})


class _PassthroughCodec(Codec):
    def __init__(self, tag):
        self._tag = tag

    def encode(self, data):
        data[f'encoded_by_{self._tag}'] = True
        return data


class _MetaSession(_FixedSession):
    @property
    def meta(self):
        return {'base_key': 'base_value'}


class _MetaPolicy(Policy):
    def new_session(self, context=None, rt=None):
        return _MetaSession({})


_T0_OBS = {obs_keys.OBS_TIME_NS: 0}


def test_action_horizon_sec_truncates_chunk():
    actions = [{'v': i} for i in range(10)]
    # action_horizon_sec=0.1s at action_fps=30 -> 3 actions
    codec = ActionTiming(fps=30.0, horizon_sec=0.1)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert [r['v'] for r in result if 'v' in r] == [0, 1, 2]
    assert result[-1] == {'timestamp': pytest.approx(0.1)}  # horizon sentinel


def test_action_horizon_sec_none_returns_full_chunk():
    actions = [{'v': i} for i in range(5)]
    codec = ActionTiming(fps=30.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert len(result) == 6  # 5 actions + timestamp sentinel


def test_action_horizon_sec_larger_than_chunk():
    actions = [{'v': i} for i in range(3)]
    # action_horizon_sec=10s at action_fps=10 -> 100 actions max, but only 3 available
    codec = ActionTiming(fps=10.0, horizon_sec=10.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert len(result) == 4  # 3 actions + timestamp sentinel (nothing truncated)


def test_timestamps_embedded_in_actions():
    actions = [{'v': i} for i in range(4)]
    codec = ActionTiming(fps=10.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert len(result) == 5  # 4 actions + timestamp sentinel
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * 0.1)


def test_action_horizon_sec_seconds_truncates():
    actions = [{'v': i} for i in range(100)]
    # 0.1s at 30fps -> 3 actions
    codec = ActionTiming(fps=30.0, horizon_sec=0.1)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert len(result) == 4  # 3 actions + horizon sentinel
    dt = 1.0 / 30.0
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * dt)


def test_action_timestamp_stamps_chunk():
    actions = [{'v': i} for i in range(4)]
    codec = ActionTimestamp(fps=10.0)
    result = codec.decode(actions)
    assert len(result) == 5  # 4 actions + timestamp sentinel
    for i, action in enumerate(result):
        assert action['timestamp'] == pytest.approx(i * 0.1)


def test_action_timestamp_single_action():
    codec = ActionTimestamp(fps=15.0)
    result = codec.decode({'v': 42})
    assert result['timestamp'] == 0.0


def test_action_timestamp_meta():
    codec = ActionTimestamp(fps=15.0)
    assert codec.meta == {'action_fps': 15.0}


def test_action_horizon_truncates():
    actions = [{'v': i, 'timestamp': i * 0.1} for i in range(10)]
    codec = ActionHorizon(0.3)
    result = codec.decode(actions)
    assert [r['v'] for r in result if 'v' in r] == [0, 1, 2]
    assert result[-1] == {'timestamp': pytest.approx(0.3)}  # horizon sentinel


def test_action_horizon_passes_single_action():
    action = {'v': 1, 'timestamp': 0.0}
    codec = ActionHorizon(0.5)
    result = codec.decode(action)
    assert result == {'v': 1, 'timestamp': 0.0}


def test_action_horizon_meta():
    codec = ActionHorizon(1.0)
    assert codec.meta == {'action_horizon_sec': 1.0}


def test_action_timestamp_and_horizon_compose():
    actions = [{'v': i} for i in range(10)]
    codec = ActionHorizon(0.3) | ActionTimestamp(fps=10.0)
    policy = codec.wrap(_ChunkPolicy(actions))
    result = policy.new_session()(_T0_OBS, 0)
    assert len(result) == 4  # 3 actions + horizon sentinel
    assert [r['v'] for r in result if 'v' in r] == [0, 1, 2]
    assert result[-1] == {'timestamp': pytest.approx(0.3)}  # horizon sentinel


def test_single_action_has_zero_timestamp():
    codec = ActionTiming(fps=15.0)
    policy = codec.wrap(_SinglePolicy())
    result = policy.new_session()(_T0_OBS, 0)
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
    """A wrapped session reports the base meta and the codec meta."""
    codec = ActionTiming(fps=15.0, horizon_sec=1.0)
    policy = codec.wrap(_MetaPolicy())
    meta = policy.new_session().meta
    assert meta['base_key'] == 'base_value'
    assert meta['action_fps'] == 15.0
    assert meta['action_horizon_sec'] == 1.0


def test_timestamps_survive_action_decoder_composition():
    """Timestamps from ActionTiming must survive through composed action decoders."""
    action_codec = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, 'target_grip', Rotation.Representation.QUAT)
    timing = ActionTiming(fps=15.0, horizon_sec=1.0)
    composed = timing | action_codec

    # Build a raw action vector: 4 quat + 3 trans + 1 grip = 8
    raw_action = np.zeros(8, dtype=np.float32)
    raw_action[:4] = Rotation.identity.as_quat
    raw_action[4:7] = [0.1, 0.2, 0.3]
    raw_action[7] = 0.5

    raw_chunk = [{'action': raw_action} for _ in range(5)]
    decoded = composed.decode(raw_chunk)

    assert len(decoded) == 6  # 5 actions + timestamp sentinel
    for i, action in enumerate(decoded[:5]):
        assert obs_keys.ROBOT_COMMAND in action
        assert 'target_grip' in action
        assert 'timestamp' in action, f'Action {i} missing timestamp — stripped by action decoder'
        assert action['timestamp'] == pytest.approx(i / 15.0)
    assert decoded[-1] == {'timestamp': pytest.approx(5 / 15.0)}  # timestamp sentinel


def test_composed_training_encoder_uses_parallel():
    """``timing | (obs & action)`` training encoder produces only derived keys, no originals."""
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    grip = [0.5, 0.6]
    img = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in ts]

    ep = EpisodeContainer({
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, grip),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        'target_grip': DummySignal(ts, grip),
        obs_keys.WRIST_IMAGE: DummySignal(ts, img),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, img),
        obs_keys.TASK: 'test',
    })

    obs = ObservationCodec(
        state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
        images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
    )
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, 'target_grip', num_joints=7)
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
    assert obs_keys.TARGET_JOINTS not in result

    # Meta should merge from all codecs
    assert encoder.meta.get('action_fps') == 15.0
    assert 'lerobot_features' in encoder.meta


def test_binarize_grip_inference():
    binarize = BinarizeGripInference()
    assert binarize._decode_single({'target_grip': 0.3}) == {'target_grip': 0.0}
    assert binarize._decode_single({'target_grip': 0.7}) == {'target_grip': 1.0}
    assert binarize._decode_single({'target_grip': 0.5}) == {'target_grip': 0.0}

    binarize_low = BinarizeGripInference(threshold=0.3)
    assert binarize_low._decode_single({'target_grip': 0.4}) == {'target_grip': 1.0}


def test_binarize_grip_training():
    ts = [1000, 2000]
    ep = EpisodeContainer({obs_keys.GRIP: DummySignal(ts, [0.3, 0.8]), 'target_grip': DummySignal(ts, [0.7, 0.2])})

    binarize = BinarizeGripTraining((obs_keys.GRIP, 'target_grip'))
    result = binarize.training_encoder(ep)
    grip_vals = [v for v, _ in result[obs_keys.GRIP]]
    tgt_vals = [v for v, _ in result['target_grip']]
    np.testing.assert_array_equal(grip_vals, [0.0, 1.0])
    np.testing.assert_array_equal(tgt_vals, [1.0, 0.0])


def test_binarize_grip_training_respects_threshold():
    ts = [1000]
    ep = EpisodeContainer({obs_keys.GRIP: DummySignal(ts, [0.4]), 'target_grip': DummySignal(ts, [0.4])})

    keys = (obs_keys.GRIP, 'target_grip')
    default = BinarizeGripTraining(keys)
    result = default.training_encoder(ep)
    assert list(result[obs_keys.GRIP])[0][0] == pytest.approx(0.0)

    low = BinarizeGripTraining(keys, threshold=0.3)
    result = low.training_encoder(ep)
    assert list(result[obs_keys.GRIP])[0][0] == pytest.approx(1.0)


def test_binarize_grip_training_composed_with_action_codec():
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({obs_keys.TARGET_JOINTS: DummySignal(ts, joints), 'target_grip': DummySignal(ts, [0.7])})

    binarize = BinarizeGripTraining((obs_keys.GRIP, 'target_grip'))
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, 'target_grip', num_joints=7)
    composed = binarize | action

    result = composed.training_encoder(ep)
    vec = list(result['action'])[0][0]
    assert vec[-1] == pytest.approx(1.0)


def test_flip_grip():
    flip = FlipGrip()

    obs = {obs_keys.GRIP: 0.2, 'other': 1.0}
    assert flip.encode(obs) == {obs_keys.GRIP: pytest.approx(0.8), 'other': 1.0}
    assert obs[obs_keys.GRIP] == 0.2  # the codec copies rather than mutates: the raw dict is the recording tap's input
    assert flip.encode({'other': 1.0}) == {'other': 1.0}

    assert flip._decode_single({'target_grip': 0.9}) == {'target_grip': pytest.approx(0.1)}
    assert flip._decode_single({'pose': 1.0}) == {'pose': 1.0}
    assert flip.decode([{'target_grip': 1.0}, {'target_grip': 0.25}]) == [
        {'target_grip': pytest.approx(0.0)},
        {'target_grip': pytest.approx(0.75)},
    ]


def test_flip_grip_composed_with_obs_and_action():
    obs = ObservationCodec(state={'observation.state': {obs_keys.GRIP: 1}}, images={})
    action = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, 'target_grip', Rotation.Representation.QUAT)
    composed = FlipGrip() | (obs & action)

    encoded = composed.encode({obs_keys.GRIP: 0.2})
    np.testing.assert_allclose(encoded['observation.state'], [0.8])

    vec = np.concatenate([[0.1, -0.2, 0.3], Rotation.identity.as_quat, [0.9]]).astype(np.float32)
    decoded = composed.decode({'action': vec})
    assert decoded['target_grip'] == pytest.approx(0.1)


def test_parallel_codec_encode_merges_outputs():
    """``obs & action`` encode produces only obs keys (action returns {})."""
    obs = ObservationCodec(state={'observation.state': {'a': 1}}, images={})
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
    assert obs_keys.ROBOT_COMMAND in result
    assert 'target_grip' in result
    assert 'action' not in result


def test_sequential_into_parallel_training():
    """``binarize | (obs & action)`` — binarize modifies grip seen by both."""
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, [0.7]),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        'target_grip': DummySignal(ts, [0.3]),
        obs_keys.WRIST_IMAGE: DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
    })

    obs = ObservationCodec(
        state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
        images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
    )
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, 'target_grip', num_joints=7)
    binarize = BinarizeGripTraining((obs_keys.GRIP, 'target_grip'))
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
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, grip),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        'target_grip': DummySignal(ts, grip),
        obs_keys.WRIST_IMAGE: DummySignal(ts, img),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, img),
        obs_keys.TASK: 'test',
    })

    codec = compose(
        obs=ObservationCodec(
            state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
            images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
        ),
        action=AbsoluteJointsAction(obs_keys.TARGET_JOINTS, 'target_grip', num_joints=7),
    )

    result = codec.training_encoder(ep)

    # Derived keys present
    assert 'observation.state' in result
    assert 'action' in result

    # Original episode keys must NOT leak through — this fails if compose uses | instead of &
    assert 'target_grip' not in result
    assert obs_keys.TARGET_JOINTS not in result
    assert obs_keys.JOINTS not in result
    assert obs_keys.GRIP not in result


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
