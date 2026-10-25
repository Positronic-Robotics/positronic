import json

import numpy as np
import pytest
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import keys as obs_keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.geom import Rotation
from positronic.policy.codecs import ACTION, BinarizeGripInference, BinarizeGripTraining, FlipGrip
from positronic.policy.codecs.action import AbsoluteJointsAction, AbsolutePositionAction
from positronic.policy.codecs.observation import StateObservationCodec
from positronic.policy.spec import from_spec


def test_binarize_grip_inference():
    binarize = BinarizeGripInference()
    assert binarize._decode_single({obs_keys.TARGET_GRIP: 0.3}) == {obs_keys.TARGET_GRIP: 0.0}
    assert binarize._decode_single({obs_keys.TARGET_GRIP: 0.7}) == {obs_keys.TARGET_GRIP: 1.0}
    assert binarize._decode_single({obs_keys.TARGET_GRIP: 0.5}) == {obs_keys.TARGET_GRIP: 0.0}

    binarize_low = BinarizeGripInference(threshold=0.3)
    assert binarize_low._decode_single({obs_keys.TARGET_GRIP: 0.4}) == {obs_keys.TARGET_GRIP: 1.0}


def test_binarize_grip_training():
    ts = [1000, 2000]
    ep = EpisodeContainer({
        obs_keys.GRIP: DummySignal(ts, [0.3, 0.8]),
        obs_keys.TARGET_GRIP: DummySignal(ts, [0.7, 0.2]),
    })

    binarize = BinarizeGripTraining((obs_keys.GRIP, obs_keys.TARGET_GRIP))
    result = binarize.training_encoder(ep)
    grip_vals = [v for v, _ in result[obs_keys.GRIP]]
    tgt_vals = [v for v, _ in result[obs_keys.TARGET_GRIP]]
    np.testing.assert_array_equal(grip_vals, [0.0, 1.0])
    np.testing.assert_array_equal(tgt_vals, [1.0, 0.0])


def test_binarize_grip_training_respects_threshold():
    ts = [1000]
    ep = EpisodeContainer({obs_keys.GRIP: DummySignal(ts, [0.4]), obs_keys.TARGET_GRIP: DummySignal(ts, [0.4])})

    keys = (obs_keys.GRIP, obs_keys.TARGET_GRIP)
    default = BinarizeGripTraining(keys)
    result = default.training_encoder(ep)
    assert list(result[obs_keys.GRIP])[0][0] == pytest.approx(0.0)

    low = BinarizeGripTraining(keys, threshold=0.3)
    result = low.training_encoder(ep)
    assert list(result[obs_keys.GRIP])[0][0] == pytest.approx(1.0)


def test_binarize_grip_training_composed_with_action_codec():
    ts = [1000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)]

    ep = EpisodeContainer({
        obs_keys.TARGET_JOINTS: DummySignal(ts, joints),
        obs_keys.TARGET_GRIP: DummySignal(ts, [0.7]),
    })

    binarize = BinarizeGripTraining((obs_keys.GRIP, obs_keys.TARGET_GRIP))
    action = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7)
    composed = binarize | action

    result = composed.training_encoder(ep)
    vec = list(result[ACTION])[0][0]
    assert vec[-1] == pytest.approx(1.0)


def test_flip_grip():
    flip = FlipGrip()

    obs = {obs_keys.GRIP: 0.2, 'other': 1.0}
    assert flip.encode(obs) == {obs_keys.GRIP: pytest.approx(0.8), 'other': 1.0}
    assert obs[obs_keys.GRIP] == 0.2  # the codec copies rather than mutates: the raw dict is the recording tap's input
    assert flip.encode({'other': 1.0}) == {'other': 1.0}

    assert flip._decode_single({obs_keys.TARGET_GRIP: 0.9}) == {obs_keys.TARGET_GRIP: pytest.approx(0.1)}
    assert flip._decode_single({'pose': 1.0}) == {'pose': 1.0}
    assert flip.decode([{obs_keys.TARGET_GRIP: 1.0}, {obs_keys.TARGET_GRIP: 0.25}]) == [
        {obs_keys.TARGET_GRIP: pytest.approx(0.0)},
        {obs_keys.TARGET_GRIP: pytest.approx(0.75)},
    ]


_LEFT, _RIGHT = (obs_keys.arm_channel(obs_keys.GRIP, arm) for arm in obs_keys.BIMANUAL_ARMS)
_TARGETS = [obs_keys.arm_channel(obs_keys.TARGET_GRIP, arm) for arm in obs_keys.BIMANUAL_ARMS]


def test_flip_grip_with_arms_flips_the_grip_of_each_arm():
    flip = FlipGrip(arms=obs_keys.BIMANUAL_ARMS)
    obs = {_LEFT: 0.2, _RIGHT: 1.0, obs_keys.GRIP: 0.3, obs_keys.TASK: 'fold'}
    assert flip.encode(obs) == {
        _LEFT: pytest.approx(0.8),
        _RIGHT: pytest.approx(0.0),
        obs_keys.GRIP: 0.3,
        obs_keys.TASK: 'fold',
    }
    assert obs[_LEFT] == 0.2

    decoded = flip._decode_single(dict(zip(_TARGETS, (0.9, 0.25), strict=True)))
    assert decoded == {_TARGETS[0]: pytest.approx(0.1), _TARGETS[1]: pytest.approx(0.75)}


def test_flip_grip_without_arms_leaves_the_channels_of_each_arm():
    flip = FlipGrip()
    assert flip.encode({_LEFT: 0.2, _RIGHT: 1.0}) == {_LEFT: 0.2, _RIGHT: 1.0}
    assert flip._decode_single({_TARGETS[0]: 0.9}) == {_TARGETS[0]: 0.9}


def test_flip_grip_without_arms_publishes_version_1_with_no_arguments():
    assert FlipGrip().to_spec() == {NAME: 'flip_grip', VERSION: 1}


def test_flip_grip_with_arms_publishes_version_2_with_the_arms():
    spec = FlipGrip(arms=obs_keys.BIMANUAL_ARMS).to_spec()
    assert spec == {NAME: 'flip_grip', VERSION: 2, ARGS: {'arms': ['left', 'right']}}


@pytest.mark.parametrize('flip', [FlipGrip(), FlipGrip(arms=obs_keys.BIMANUAL_ARMS)], ids=['v1', 'v2'])
def test_a_client_builds_each_flip_grip_version_from_its_spec(flip):
    rebuilt = from_spec(json.loads(json.dumps(flip.to_spec())))
    assert isinstance(rebuilt, FlipGrip)
    assert rebuilt.to_spec() == flip.to_spec()
    obs = {obs_keys.GRIP: 0.2, _LEFT: 0.2, _RIGHT: 0.6}
    assert rebuilt.encode(obs) == flip.encode(obs)


def test_flip_grip_version_1_refuses_arguments():
    with pytest.raises(TypeError):
        from_spec({NAME: 'flip_grip', VERSION: 1, ARGS: {'arms': ['left', 'right']}})


def test_an_unknown_flip_grip_version_is_refused():
    with pytest.raises(ValueError, match='Unsupported component .flip_grip. version 3; supported versions: \\[1, 2\\]'):
        from_spec({NAME: 'flip_grip', VERSION: 3})


def test_flip_grip_composed_with_obs_and_action():
    obs = StateObservationCodec(state={'observation.state': {obs_keys.GRIP: 1}})
    action = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, obs_keys.TARGET_GRIP, Rotation.Representation.QUAT)
    composed = FlipGrip() | (obs & action)

    encoded = composed.encode({obs_keys.GRIP: 0.2})
    np.testing.assert_allclose(encoded['observation.state'], [0.8])

    vec = np.concatenate([[0.1, -0.2, 0.3], Rotation.identity.as_quat, [0.9]]).astype(np.float32)
    decoded = composed.decode({ACTION: vec})
    assert decoded[obs_keys.TARGET_GRIP] == pytest.approx(0.1)
