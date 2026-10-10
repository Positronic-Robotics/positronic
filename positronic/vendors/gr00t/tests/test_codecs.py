import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm import models
from positronic.policy import spec
from positronic.policy.codecs import ACTION, Codec
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings


@pytest.mark.parametrize('config', [model_settings.load_settings, model_settings.three_camera_settings])
def test_pose_conversion_preserves_droid_convention_and_tool_frame_metadata(config):
    settings = config()
    settings[model_settings.IMAGE_MAPPINGS] = {}
    codec = spec.from_spec(recipe.inference(settings))
    assert isinstance(codec, Codec)
    rng = np.random.default_rng(3)
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    for _ in range(20):
        pose = geom.Transform3D(rng.normal(size=3), geom.Rotation.from_quat(rng.normal(size=4)))
        inputs = {
            keys.EE_POSE: pose.as_vector(geom.Rotation.Representation.QUAT),
            keys.JOINTS: np.zeros(7),
            keys.GRIP: 0.2,
            keys.TASK: 'pick',
        }
        tool_pose = pose * models.DROID_EE_FRAME
        expected = np.concatenate([
            tool_pose.translation,
            (tool_pose.rotation.as_rotation_matrix @ correction)[:2].reshape(6),
        ]).astype(np.float32)
        encoded = codec.encode(inputs)[gr00t.STATE][gr00t.EE_POSE][0, 0]
        np.testing.assert_allclose(encoded, expected, atol=2e-7)

    expected_frame = models.DROID_EE_FRAME.as_vector(geom.Rotation.Representation.QUAT)
    np.testing.assert_array_equal(codec.meta[roboarm_keys.EE_FRAME], expected_frame)
    np.testing.assert_array_equal(recipes.droid(settings=settings).meta[roboarm_keys.EE_FRAME], expected_frame)


@pytest.mark.parametrize('config', [model_settings.load_settings, model_settings.three_camera_settings])
@pytest.mark.parametrize('horizon', [0, 1, 40])
def test_droid_decodes_native_chunk_and_binarizes_grip(config, horizon):
    codec = spec.from_spec(recipe.inference(config()))
    assert isinstance(codec, Codec)
    targets = np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7) / 100
    output = (
        {
            gr00t.JOINT_POSITION: targets[np.newaxis],
            gr00t.GRIP: np.array([0.5 if i % 2 else 0.51 for i in range(horizon)]).reshape(1, horizon, 1),
        },
        {},
    )
    decoded = codec.decode(output)
    assert len(decoded) == horizon
    for i, item in enumerate(decoded):
        np.testing.assert_array_equal(item[keys.ROBOT_COMMAND].positions, targets[i])
        assert item[keys.ROBOT_COMMAND].positions.dtype == np.float32
        assert item[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert item[keys.TARGET_GRIP] == (0.0 if i % 2 else 1.0)
        assert 'timestamp' not in item


def test_training_cadence_metadata():
    training = recipes.droid(settings=model_settings.load_settings(overrides={ACTION_FPS: 20}))
    assert training.meta[ACTION_FPS] == 20


def test_training_actions_align_recorded_samples():
    training = recipes.droid(
        settings=model_settings.load_settings(
            overrides={model_settings.IMAGE_MAPPINGS: {}, model_settings.EE_FRAME: [0, 0, 0, 1, 0, 0, 0]}
        )
    )
    episode = EpisodeContainer({
        keys.EE_POSE: DummySignal([100], [[0, 0, 0, 1, 0, 0, 0]]),
        keys.JOINTS: DummySignal([100, 300], [np.zeros(7), np.ones(7)]),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0]),
    })
    encoded = training(episode)
    action = encoded[ACTION]
    assert list(action.timestamps(RECORDED_TIME)) == [100, 200, 300]
    values = np.asarray(action.values())
    np.testing.assert_array_equal(values[:, :9], np.repeat(encoded[gr00t.EE_POSE].values(), 3, axis=0))
    np.testing.assert_array_equal(values[:, 9], [0, 1, 1])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.zeros(7), np.ones(7)])
