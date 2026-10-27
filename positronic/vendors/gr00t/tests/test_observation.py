import importlib.util
import json
import os
from functools import reduce
from operator import or_
from pathlib import Path

import numpy as np
import pytest
from positronic_model_server import serialization
from scipy.spatial.transform import Rotation

from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.time import Time
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm import models
from positronic.policy import spec
from positronic.policy.codecs import ACTION, GR00T_MODALITY, Codec
from positronic.policy.codecs.metadata import MODEL_SETTINGS
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import keys as serving_keys
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings


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


def _episode(observation, task_key=keys.TASK, pose_key=keys.EE_POSE):
    return EpisodeContainer({
        **{
            name: value if name == task_key else DummySignal([0, 1], [value, value])
            for name, value in observation.items()
        },
        keys.TARGET_GRIP: DummySignal([0, 1], [1.0, 1.0]),
        **models.ee_frames_entry({pose_key: models.EEFrame(site=models.GRIPPER_SITE)}),
    })


@pytest.mark.parametrize('config', [model_settings.droid, model_settings.droid_three_cameras])
def test_training_and_inference_encode_the_same_absolute_state_and_images(config, observation):
    settings = config()
    if gr00t.EXTERIOR_IMAGE_2 in settings[model_settings.IMAGE_MAPPINGS]:
        settings[model_settings.IMAGE_MAPPINGS][gr00t.EXTERIOR_IMAGE_2] = 'alternate_view'
        observation['alternate_view'] = observation.pop(keys.EXTERIOR_IMAGE_2)
    codec = spec.from_spec(recipe.inference(settings))
    assert isinstance(codec, Codec)
    training = recipes.droid(settings=settings)(_episode(observation))
    encoded = codec.encode(observation)
    assert set(encoded) == {gr00t.STATE, gr00t.VIDEO, gr00t.LANGUAGE}
    assert encoded[gr00t.LANGUAGE] == {gr00t.TASK: [[observation[keys.TASK]]]}
    for name, value in encoded[gr00t.STATE].items():
        assert value.shape == (1, 1, gr00t.STATE_DIMS[name])
        assert value.dtype == np.float32
        assert np.asarray(training[name][0][0]).dtype == np.float32
        np.testing.assert_allclose(training[name][0][0], value[0, 0], atol=1e-6)
    for name, frames in encoded[gr00t.VIDEO].items():
        assert frames.shape == (1, 1, 180, 320, 3)
        np.testing.assert_array_equal(training[name][0][0], frames[0, 0])
    expected_action = {**{name: value[0, 0] for name, value in encoded[gr00t.STATE].items()}, gr00t.GRIP: [1.0]}
    np.testing.assert_allclose(training[ACTION][0][0], np.concatenate([expected_action[n] for n in gr00t.STATE_DIMS]))


@pytest.mark.parametrize('image_mappings', [{}, {gr00t.EE_POSE: keys.WRIST_IMAGE}])
def test_observation_layout_preserves_empty_camera_groups_and_names_shared_with_state(image_mappings, observation):
    settings = model_settings.droid()
    settings[model_settings.IMAGE_MAPPINGS] = image_mappings
    codec = spec.from_spec(recipe.inference(settings))
    assert isinstance(codec, Codec)
    encoded = codec.encode(observation)
    assert set(encoded[gr00t.VIDEO]) == set(image_mappings)
    assert encoded[gr00t.STATE][gr00t.EE_POSE].shape == (1, 1, 9)
    for name in image_mappings:
        assert encoded[gr00t.VIDEO][name].shape == (1, 1, 180, 320, 3)


def test_observation_layout_requires_a_live_prompt(observation):
    del observation[keys.TASK]
    with pytest.raises(KeyError, match=keys.TASK):
        spec.from_spec(recipe.inference(model_settings.droid())).encode(observation)


@pytest.mark.parametrize('task', [None, 'Pick up the cup'])
def test_training_episode_materializes_without_requiring_a_recorded_task(observation, task):
    observation.pop(keys.TASK)
    if task is not None:
        observation[keys.TASK] = task
    training = recipes.droid()(_episode(observation))
    frame = training.time[[Time(**{RECORDED_TIME: 0})]]
    assert frame[keys.TASK] == (task or '')


def test_three_camera_configuration_uses_a_distinct_second_external_image(observation):
    encoded = spec.from_spec(recipe.inference(model_settings.droid_three_cameras())).encode(observation)
    assert len(encoded[gr00t.VIDEO]) == 3
    np.testing.assert_array_equal(
        encoded[gr00t.VIDEO][gr00t.EXTERIOR_IMAGE_2][0, 0], observation[keys.EXTERIOR_IMAGE_2]
    )
    del observation[keys.EXTERIOR_IMAGE_2]
    with pytest.raises(KeyError):
        spec.from_spec(recipe.inference(model_settings.droid_three_cameras())).encode(observation)


def test_action_metadata_matches_values_when_state_dimensions_are_reordered(monkeypatch, observation):
    monkeypatch.setattr(gr00t, 'STATE_DIMS', dict(reversed(list(gr00t.STATE_DIMS.items()))))
    encoder = recipes.droid()
    encoded = encoder(_episode(observation))
    action = encoded[ACTION][0][0]
    for name, bounds in encoder.meta[GR00T_MODALITY][ACTION].items():
        expected = [1.0] if name == gr00t.GRIP else encoded[name][0][0]
        np.testing.assert_allclose(action[bounds[gr00t.START] : bounds[gr00t.END]], expected)


def test_standalone_observation_names_match_the_client():
    for name in ('EE_POSE', 'JOINTS', 'GRIP', 'TASK', 'EXTERIOR_IMAGE', 'EXTERIOR_IMAGE_2', 'WRIST_IMAGE'):
        assert getattr(serving_keys, name) == getattr(keys, name)


def test_standalone_ee_frames_match_the_client():
    assert model_settings.EE_FRAMES == roboarm_keys.EE_FRAMES
    assert {model_settings.SITE, model_settings.OFFSET} == set(models.EEFrame.model_fields)
    assert model_settings.GRIPPER_SITE == models.GRIPPER_SITE


def test_python_settings_drive_independent_training_and_inference_recipes(observation):
    settings = {
        **model_settings.droid(),
        model_settings.IMAGE_SIZE: [160, 90],
        model_settings.IMAGE_MAPPINGS: {'custom_camera': keys.EXTERIOR_IMAGE_2},
        model_settings.ROTATION_OFFSET: [1, 0, 0, 0],
        ACTION_FPS: 20,
    }
    for name, source in settings[model_settings.OBSERVATION_KEYS].items():
        renamed = f'custom.{name}'
        observation[renamed] = observation.pop(source)
        settings[model_settings.OBSERVATION_KEYS][name] = renamed
    pose_key = settings[model_settings.OBSERVATION_KEYS][gr00t.EE_POSE]
    settings[model_settings.EE_FRAMES] = {
        pose_key: {model_settings.SITE: model_settings.GRIPPER_SITE, model_settings.OFFSET: [0, 0, 0, 1, 0, 0, 0]}
    }
    training = recipes.droid(settings=settings)
    description = json.loads(json.dumps(recipe.inference(settings)))
    codec = spec.from_spec(description)
    assert isinstance(codec, Codec)
    prepared = training(_episode(observation, settings[model_settings.OBSERVATION_KEYS][gr00t.TASK], pose_key))
    encoded = codec.encode(observation)
    assert prepared[keys.TASK] == encoded[gr00t.LANGUAGE][gr00t.TASK][0][0]
    assert encoded[gr00t.VIDEO]['custom_camera'].shape == (1, 1, 90, 160, 3)
    np.testing.assert_array_equal(prepared['custom_camera'][0][0], encoded[gr00t.VIDEO]['custom_camera'][0, 0])
    np.testing.assert_array_equal(prepared[gr00t.EE_POSE][0][0], encoded[gr00t.STATE][gr00t.EE_POSE][0, 0])
    assert training.meta[MODEL_SETTINGS] == codec.meta[MODEL_SETTINGS] == settings


@pytest.mark.parametrize('config', [model_settings.droid, model_settings.droid_three_cameras])
@pytest.mark.parametrize('image_size', [[320, 180], [160, 90]])
@pytest.mark.parametrize('compress', [False, True])
def test_complete_client_description_prepares_images_before_serialization(config, observation, image_size, compress):
    settings = config()
    settings[model_settings.IMAGE_SIZE] = image_size
    description = recipe.client_stack(
        settings,
        fps=10,
        horizon_sec=1,
        image_mappings=settings[model_settings.IMAGE_MAPPINGS],
        compress_images=compress,
    )
    assert any(part.get(spec.NAME) == 'encode_images' for part in description[spec.SEQ]) is compress
    pipeline = spec.from_spec(description)
    codec = reduce(or_, (part for part in pipeline._components if isinstance(part, Codec)))
    encoded = codec.encode(observation)
    received = serialization.deserialise(serialization.serialise(encoded))
    direct = spec.from_spec(recipe.inference(settings)).encode(observation)
    assert pipeline.meta()[ACTION_FPS] == 10
    assert settings[model_settings.ACTION_FPS] == 15
    for name in direct[gr00t.VIDEO]:
        assert received[gr00t.VIDEO][name].shape == (1, 1, image_size[1], image_size[0], 3)
        if not compress:
            np.testing.assert_array_equal(received[gr00t.VIDEO][name], direct[gr00t.VIDEO][name])
        else:
            assert isinstance(encoded[gr00t.VIDEO][name], dict)


def test_droid_frame_and_pixels_match_upstream_robot_client(observation):
    reference = os.environ.get('GR00T_REFERENCE_ROOT')
    if reference is None:
        pytest.skip('Set GR00T_REFERENCE_ROOT to the GR00T checkout for cross-repository parity')
    loaded = {}
    for name, path in {'frame': 'gr00t/data/state_action/droid_frame.py', 'image': 'examples/DROID/utils.py'}.items():
        module_spec = importlib.util.spec_from_file_location(name, Path(reference) / path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        loaded[name] = module
    raw_pose = geom.Transform3D.from_vector(observation[keys.EE_POSE], geom.Rotation.Representation.QUAT)
    tool_pose = raw_pose * models.DROID_EE_FRAME
    upstream_pose = np.concatenate([
        tool_pose.translation,
        Rotation.from_matrix(tool_pose.rotation.as_rotation_matrix).as_euler('XYZ'),
    ])
    encoded = spec.from_spec(recipe.inference(model_settings.droid())).encode(observation)
    np.testing.assert_allclose(
        encoded[gr00t.STATE][gr00t.EE_POSE][0, 0], loaded['frame'].compute_eef_9d(upstream_pose), atol=1e-6
    )
    for name, source in {gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE: keys.WRIST_IMAGE}.items():
        expected = loaded['image'].resize_with_pad(observation[source], 180, 320)
        np.testing.assert_array_equal(encoded[gr00t.VIDEO][name][0, 0], expected)
