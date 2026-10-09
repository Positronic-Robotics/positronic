import numpy as np
import pytest

from positronic import keys as obs_keys
from positronic.cfg.codecs import compose
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.geom import Rotation, Transform3D
from positronic.policy import spec
from positronic.policy.codecs import (
    ACTION,
    LEROBOT_FEATURES,
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    Metadata,
)
from positronic.policy.codecs.action import AbsoluteJointsAction, AbsolutePositionAction
from positronic.policy.codecs.observation import ObservationCodec


class _TaggingCodec(Codec):
    def __init__(self, tag):
        self._tag = tag

    def encode(self, data):
        data[f'encoded_by_{self._tag}'] = True
        return data


def test_codec_composition():
    left = _TaggingCodec('left')
    right = _TaggingCodec('right')
    composed = left | right

    result = composed.encode({})
    assert result['encoded_by_left'] is True
    assert result['encoded_by_right'] is True


def test_composed_training_encoder_uses_parallel():
    """``metadata | (obs & action)`` training encoder produces only derived keys, no originals."""
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    grip = [0.5, 0.6]
    img = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in ts]

    ep = EpisodeContainer({
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, grip),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        obs_keys.TARGET_GRIP: DummySignal(ts, grip),
        obs_keys.WRIST_IMAGE: DummySignal(ts, img),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, img),
        obs_keys.TASK: 'test',
    })

    obs = ObservationCodec(
        state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
        images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
    )
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7)
    metadata = Metadata({'action_fps': 15.0})
    composed = metadata | (obs & action)

    encoder = composed.training_encoder
    result = encoder(ep)

    assert 'observation.state' in result
    assert 'observation.images.left' in result

    # Action codec's derived key — must be accessible (reads target_grip from base episode)
    assert ACTION in result
    vec = list(result[ACTION])[0][0]
    assert vec.shape == (8,)
    np.testing.assert_allclose(vec[:7], joints[0], atol=1e-6)
    np.testing.assert_allclose(vec[7], grip[0], atol=1e-6)

    # Original episode keys should NOT appear (no Identity pass-through)
    assert obs_keys.TARGET_GRIP not in result
    assert obs_keys.TARGET_JOINTS not in result

    assert encoder.meta.get('action_fps') == 15.0
    assert LEROBOT_FEATURES in encoder.meta


def test_parallel_codec_encode_merges_outputs():
    """``obs & action`` encode produces only obs keys (action returns {})."""
    obs = ObservationCodec(state={'observation.state': {'a': 1}}, images={})
    action = AbsolutePositionAction('x', 'y')
    composed = obs & action
    result = composed.encode({'a': 1.0})
    assert 'observation.state' in result
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
    result = composed.decode({ACTION: raw_action})
    assert obs_keys.ROBOT_COMMAND in result
    assert obs_keys.TARGET_GRIP in result
    assert ACTION not in result


def test_sequential_into_parallel_training():
    """``binarize | (obs & action)`` — binarize modifies grip seen by both."""
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, [0.7]),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        obs_keys.TARGET_GRIP: DummySignal(ts, [0.3]),
        obs_keys.WRIST_IMAGE: DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, [np.zeros((4, 4, 3), dtype=np.uint8)]),
    })

    obs = ObservationCodec(
        state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
        images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
    )
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7)
    binarize = BinarizeGripTraining((obs_keys.GRIP, obs_keys.TARGET_GRIP))
    composed = binarize | (obs & action)

    result = composed.training_encoder(ep)

    # Binarize runs first — grip (0.7 > 0.5 → 1.0), target_grip (0.3 ≤ 0.5 → 0.0)
    vec = list(result[ACTION])[0][0]
    assert vec[-1] == pytest.approx(0.0)

    state = list(result['observation.state'])[0][0]
    assert state[-1] == pytest.approx(1.0)


def test_compose_training_encoder_produces_only_derived_keys():
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    grip = [0.5, 0.6]
    img = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in ts]

    ep = EpisodeContainer({
        obs_keys.JOINTS: DummySignal(ts, joints),
        obs_keys.GRIP: DummySignal(ts, grip),
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        obs_keys.TARGET_GRIP: DummySignal(ts, grip),
        obs_keys.WRIST_IMAGE: DummySignal(ts, img),
        obs_keys.EXTERIOR_IMAGE: DummySignal(ts, img),
        obs_keys.TASK: 'test',
    })

    codec = compose(
        obs=ObservationCodec(
            state={'observation.state': {obs_keys.JOINTS: 7, obs_keys.GRIP: 1}},
            images={'observation.images.left': (obs_keys.WRIST_IMAGE, (4, 4))},
        ),
        action=AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7),
    )

    result = codec.training_encoder(ep)

    assert 'observation.state' in result
    assert ACTION in result

    # Original episode keys must NOT leak through — this fails if compose uses | instead of &
    assert obs_keys.TARGET_GRIP not in result
    assert obs_keys.TARGET_JOINTS not in result
    assert obs_keys.JOINTS not in result
    assert obs_keys.GRIP not in result


def test_operator_precedence():
    """``a | b & c`` binds as ``a | (b & c)`` — & has higher precedence than |."""
    a = _TaggingCodec('a')
    b = _TaggingCodec('b')
    c = _TaggingCodec('c')

    composed = a | b & c
    result = composed.encode({})
    # a encodes first (sequential |), then b & c both see a's output (parallel &)
    assert result['encoded_by_a'] is True
    assert result['encoded_by_b'] is True
    assert result['encoded_by_c'] is True


class TestCodecComposition:
    """Codec composition preserves metadata and frame declarations."""

    def test_codec_and_stays_codec_only(self):
        """& only works between codecs, not layers."""
        c1 = Metadata({'action_fps': 10.0})
        c2 = Metadata({'action_fps': 5.0})
        composed = c1 & c2
        assert isinstance(composed, Codec)

    def test_agreeing_declarations_merge(self):
        assert (Metadata({'action_fps': 10.0}) | Metadata({'action_fps': 10.0})).meta['action_fps'] == 10.0

    def test_disagreeing_declarations_have_no_merged_answer(self):
        composed = Metadata({'action_fps': 10.0}) & Metadata({'action_fps': 5.0})
        with pytest.raises(ValueError, match='action_fps'):
            _ = composed.meta

    def test_two_frame_codecs_refuse_to_advertise_one_frame(self):
        """Poses come out at the product of both transforms, which neither codec's declaration names."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        b = Transform3D(np.array([0.01, 0.0, 0.02]), Rotation.from_euler([0.0, 0.0, -0.4]))
        with pytest.raises(ValueError, match=roboarm_keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(b)).meta

    def test_the_same_frame_twice_is_still_two_moves(self):
        """The second move starts where the first left off, so the shared value names neither end of the pair."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        with pytest.raises(ValueError, match=roboarm_keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(a)).meta

    def test_parallel_frame_codecs_keep_the_frame_they_share(self):
        """Both halves encode the same input, so one move happens and the shared declaration describes it."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        np.testing.assert_allclose(
            (ChangeEEFrame(a) & ChangeEEFrame(a)).meta[roboarm_keys.EE_FRAME], a.as_vector(Rotation.Representation.QUAT)
        )


@pytest.mark.parametrize(
    'definition',
    [
        ObservationCodec(state={'state': {'grip': 1}}, images={}) & AbsolutePositionAction('pose', 'grip'),
        FlipGrip() | (BinarizeGripInference() & AbsoluteJointsAction('joints', 'grip')),
    ],
)
def test_codec_specs_round_trip(definition):
    assert spec.from_spec(definition.to_spec()).to_spec() == definition.to_spec()
