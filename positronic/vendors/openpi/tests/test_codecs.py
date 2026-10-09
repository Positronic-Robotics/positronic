import json

import numpy as np
import pytest

from positronic import keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.policy.codec import LEROBOT_FEATURES, Codec
from positronic.policy.spec import from_spec
from positronic.vendors import openpi
from positronic.vendors.openpi import codecs


@pytest.fixture
def raw_observation() -> dict:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
        keys.EE_POSE: np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        keys.JOINTS: np.zeros(7, dtype=np.float32),
        keys.GRIP: 0.0,
        keys.WRIST_IMAGE: frame,
        keys.EXTERIOR_IMAGE: frame,
        keys.TASK: 'pick up the cube',
    }


@pytest.mark.parametrize('name', ['ee_obs', 'ee_joints_obs', 'joints_obs', 'droid_obs', 'libero_obs'])
def test_warmup_observation_carries_every_field_a_codec_encodes(name, raw_observation):
    encoded = getattr(codecs, name).instantiate().encode(raw_observation)

    assert set(encoded) <= set(openpi.warm_observation())


def test_warmup_state_is_the_width_the_transform_that_does_not_pad_hands_over(raw_observation):
    """``LiberoInputs`` passes the state straight to the model, so this is the one width that has to be right."""
    encoded = codecs.libero_obs.instantiate().encode(raw_observation)

    assert openpi.warm_observation()[openpi.STATE].shape == encoded[openpi.STATE].shape


def test_training_state_aligns_recorded_samples():
    codec = codecs.observation(state_features={keys.JOINTS: 2, keys.GRIP: 1})
    episode = EpisodeContainer({
        keys.JOINTS: DummySignal([100, 300], [[1, 2], [3, 4]]),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0]),
    })
    state = codec.training_encoder(episode)[openpi.TRAINING_STATE]
    assert list(state.timestamps(RECORDED_TIME)) == [100, 200, 300]
    np.testing.assert_array_equal(state.values(), [[1, 2, 0], [1, 2, 1], [3, 4, 1]])


@pytest.mark.parametrize(
    'preset, features',
    [
        (codecs.ee_obs, {keys.EE_POSE: 7, keys.GRIP: 1}),
        (codecs.ee_joints_obs, {keys.EE_POSE: 7, keys.GRIP: 1, keys.JOINTS: 7}),
        (codecs.joints_obs, {keys.JOINTS: 7, keys.GRIP: 1}),
    ],
    ids=['ee', 'ee_joints', 'joints'],
)
@pytest.mark.parametrize('task', [None, '', 'pick up the cube'])
def test_observation_spec_round_trip_preserves_training_and_inference_layouts(preset, features, task):
    inputs = {
        keys.EE_POSE: np.array([0.1, -0.2, 0.3, 1, 0, 0, 0]),
        keys.JOINTS: np.arange(7, dtype=np.float64),
        keys.GRIP: 0.25,
        'hand': np.full((2, 8, 3), [10, 20, 30], dtype=np.uint8),
        'front': np.full((6, 4, 3), [40, 50, 60], dtype=np.uint8),
    }
    if task is not None:
        inputs[keys.TASK] = task
    definition = preset.override(wrist_camera='hand', exterior_camera='front', image_size=(8, 6)).instantiate()
    codec = from_spec(json.loads(json.dumps(definition.to_spec())))
    assert isinstance(codec, Codec)

    encoded = codec.encode(inputs)
    expected_state = np.concatenate([np.asarray(inputs[name]).reshape(-1) for name in features]).astype(np.float32)
    wrist = np.pad(inputs['hand'], ((2, 2), (0, 0), (0, 0)))
    exterior = np.pad(inputs['front'], ((0, 0), (2, 2), (0, 0)))
    assert set(encoded) == {openpi.STATE, openpi.WRIST_IMAGE, openpi.IMAGE} | (
        {openpi.PROMPT} if task is not None else set()
    )
    np.testing.assert_array_equal(encoded[openpi.STATE], expected_state, strict=True)
    np.testing.assert_array_equal(encoded[openpi.WRIST_IMAGE], wrist, strict=True)
    np.testing.assert_array_equal(encoded[openpi.IMAGE], exterior, strict=True)
    if task is not None:
        assert encoded[openpi.PROMPT] == task
    assert codec.meta == {Codec.IMAGE_SIZES: (8, 6)}

    episode_data = {name: DummySignal([100], [value]) for name, value in inputs.items() if name != keys.TASK}
    if task is not None:
        episode_data[keys.TASK] = task
    encoder = codec.training_encoder
    training = encoder(EpisodeContainer(episode_data))
    assert set(training) == {openpi.TRAINING_STATE, openpi.TRAINING_WRIST_IMAGE, openpi.TRAINING_IMAGE, keys.TASK}
    np.testing.assert_array_equal(training[openpi.TRAINING_STATE].values()[0], expected_state, strict=True)
    np.testing.assert_array_equal(training[openpi.TRAINING_WRIST_IMAGE].values()[0], wrist, strict=True)
    np.testing.assert_array_equal(training[openpi.TRAINING_IMAGE].values()[0], exterior, strict=True)
    assert training[keys.TASK] == (task or '')
    image_feature = {'shape': (6, 8, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'}
    assert encoder.meta == {
        LEROBOT_FEATURES: {
            openpi.TRAINING_STATE: {'shape': (sum(features.values()),), 'dtype': 'float32', 'names': list(features)},
            openpi.TRAINING_WRIST_IMAGE: image_feature,
            openpi.TRAINING_IMAGE: image_feature,
        }
    }
