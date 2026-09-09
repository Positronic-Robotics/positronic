import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import models
from positronic.vendors import gr00t
from positronic.vendors.gr00t.codecs import droid, droid_three_cameras


@pytest.fixture
def observation():
    pose = geom.Transform3D([0.3, -0.2, 0.5], geom.Rotation.from_euler([0.4, -0.3, 0.7]))
    return {
        keys.EE_POSE: pose.as_vector(geom.Rotation.Representation.QUAT),
        keys.GRIP: 0.25,
        keys.JOINTS: np.arange(7, dtype=np.float64) / 10,
        keys.WRIST_IMAGE: np.random.default_rng(1).integers(0, 256, (377, 611, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.random.default_rng(2).integers(0, 256, (240, 320, 3), dtype=np.uint8),
        keys.EXTERIOR_IMAGE_2: np.full((180, 320, 3), 73, dtype=np.uint8),
        keys.TASK: 'Put the cup on the plate',
    }


@pytest.mark.parametrize('config', [droid, droid_three_cameras])
def test_training_and_inference_encode_the_same_absolute_state_and_images(config, observation):
    codec = config()
    episode = EpisodeContainer({
        name: value if name == keys.TASK else DummySignal([0, 1], [value, value]) for name, value in observation.items()
    })
    training = codec.training_encoder(episode)
    encoded = codec.encode(observation)
    for name, value in encoded[gr00t.STATE].items():
        assert np.asarray(training[name][0][0]).dtype == np.float32
        np.testing.assert_allclose(training[name][0][0], value[0, 0], atol=1e-6)
    for name, frames in encoded[gr00t.VIDEO].items():
        assert frames.shape == (1, 1, 180, 320, 3)
        np.testing.assert_array_equal(training[name][0][0], frames[0, 0])
    expected_action = np.concatenate([encoded[gr00t.STATE][name][0, 0] for name in gr00t.STATE_DIMS])
    np.testing.assert_allclose(training['action'][0][0], expected_action)


def test_three_camera_configuration_uses_a_distinct_second_external_image(observation):
    encoded = droid_three_cameras().encode(observation)
    assert len(encoded[gr00t.VIDEO]) == 3
    np.testing.assert_array_equal(
        encoded[gr00t.VIDEO][gr00t.EXTERIOR_IMAGE_2][0, 0], observation[keys.EXTERIOR_IMAGE_2]
    )
    del observation[keys.EXTERIOR_IMAGE_2]
    with pytest.raises(KeyError):
        droid_three_cameras().encode(observation)


def test_droid_frame_and_pixels_match_upstream_robot_client(observation):
    reference = os.environ.get('GR00T_REFERENCE_ROOT')
    if reference is None:
        pytest.skip('Set GR00T_REFERENCE_ROOT to the GR00T checkout for cross-repository parity')
    loaded = {}
    for name, path in {'frame': 'gr00t/data/state_action/droid_frame.py', 'image': 'examples/DROID/utils.py'}.items():
        spec = importlib.util.spec_from_file_location(name, Path(reference) / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    raw_pose = geom.Transform3D.from_vector(observation[keys.EE_POSE], geom.Rotation.Representation.QUAT)
    tool_pose = raw_pose * models.DROID_EE_FRAME
    upstream_pose = np.concatenate([
        tool_pose.translation,
        Rotation.from_matrix(tool_pose.rotation.as_rotation_matrix).as_euler('XYZ'),
    ])
    encoded = droid().encode(observation)
    np.testing.assert_allclose(
        encoded[gr00t.STATE][gr00t.EE_POSE][0, 0], loaded['frame'].compute_eef_9d(upstream_pose), atol=1e-6
    )
    for name, source in {gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE: keys.WRIST_IMAGE}.items():
        expected = loaded['image'].resize_with_pad(observation[source], 180, 320)
        np.testing.assert_array_equal(encoded[gr00t.VIDEO][name][0, 0], expected)
