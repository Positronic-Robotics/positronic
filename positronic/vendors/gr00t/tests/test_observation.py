import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.time import Time
from positronic.drivers.roboarm import models
from positronic.policy import spec
from positronic.policy.codecs import ACTION, GR00T_MODALITY, Codec, RestrictImageSize
from positronic.policy.codecs.metadata import MODEL_SETTINGS
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes, server
from positronic.vendors.gr00t import serving as gr00t
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


@pytest.mark.parametrize('config', [model_settings.load_settings, model_settings.three_camera_settings])
def test_training_and_inference_encode_the_same_absolute_state_and_images(config, observation, tmp_path):
    settings = config()
    if gr00t.EXTERIOR_IMAGE_2 in settings[model_settings.IMAGE_MAPPINGS]:
        settings[model_settings.IMAGE_MAPPINGS][gr00t.EXTERIOR_IMAGE_2] = 'alternate_view'
        observation['alternate_view'] = observation.pop(keys.EXTERIOR_IMAGE_2)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    settings = config(path=path)
    codec = spec.from_spec(recipe.inference(settings))
    assert isinstance(codec, Codec)
    episode = EpisodeContainer({
        name: value if name == keys.TASK else DummySignal([0, 1], [value, value]) for name, value in observation.items()
    })
    training = recipes.droid(settings=settings)(episode)
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
    expected_action = np.concatenate([encoded[gr00t.STATE][name][0, 0] for name in gr00t.STATE_DIMS])
    np.testing.assert_allclose(training[ACTION][0][0], expected_action)


@pytest.mark.parametrize('image_mappings', [{}, {gr00t.EE_POSE: keys.WRIST_IMAGE}])
def test_observation_layout_preserves_empty_camera_groups_and_names_shared_with_state(image_mappings, observation):
    codec = spec.from_spec(
        recipe.inference(model_settings.load_settings(overrides={model_settings.IMAGE_MAPPINGS: image_mappings}))
    )
    assert isinstance(codec, Codec)
    encoded = codec.encode(observation)
    assert set(encoded[gr00t.VIDEO]) == set(image_mappings)
    assert encoded[gr00t.STATE][gr00t.EE_POSE].shape == (1, 1, 9)
    for name in image_mappings:
        assert encoded[gr00t.VIDEO][name].shape == (1, 1, 180, 320, 3)
    assert codec.meta[Codec.IMAGE_SIZES] == ((320, 180) if image_mappings else {})


def test_observation_layout_requires_a_live_prompt(observation):
    del observation[keys.TASK]
    with pytest.raises(KeyError, match=keys.TASK):
        spec.from_spec(recipe.inference(model_settings.load_settings())).encode(observation)


@pytest.mark.parametrize('task', [None, 'Pick up the cup'])
def test_training_episode_materializes_without_requiring_a_recorded_task(observation, task):
    observation.pop(keys.TASK)
    fields = {name: DummySignal([0, 1], [value, value]) for name, value in observation.items()}
    if task is not None:
        fields[keys.TASK] = task
    training = recipes.droid()(EpisodeContainer(fields))
    frame = training.time[[Time(**{RECORDED_TIME: 0})]]
    assert frame[keys.TASK] == (task or '')


def test_three_camera_configuration_uses_a_distinct_second_external_image(observation):
    encoded = spec.from_spec(recipe.inference(model_settings.three_camera_settings())).encode(observation)
    assert len(encoded[gr00t.VIDEO]) == 3
    np.testing.assert_array_equal(
        encoded[gr00t.VIDEO][gr00t.EXTERIOR_IMAGE_2][0, 0], observation[keys.EXTERIOR_IMAGE_2]
    )
    del observation[keys.EXTERIOR_IMAGE_2]
    with pytest.raises(KeyError):
        spec.from_spec(recipe.inference(model_settings.three_camera_settings())).encode(observation)


def test_action_metadata_matches_values_when_state_dimensions_are_reordered(monkeypatch, observation):
    monkeypatch.setattr(gr00t, 'STATE_DIMS', dict(reversed(list(gr00t.STATE_DIMS.items()))))
    episode = EpisodeContainer({
        name: value if name == keys.TASK else DummySignal([0, 1], [value, value]) for name, value in observation.items()
    })
    encoder = recipes.droid()
    encoded = encoder(episode)
    action = encoded[ACTION][0][0]
    for name, bounds in encoder.meta[GR00T_MODALITY][ACTION].items():
        np.testing.assert_allclose(action[bounds['start'] : bounds['end']], encoded[name][0][0])


def test_saved_settings_drive_independent_training_and_inference_recipes(tmp_path, observation):
    settings = model_settings.load_settings(
        overrides={
            model_settings.IMAGE_SIZE: [160, 90],
            model_settings.IMAGE_MAPPINGS: {'custom_camera': keys.EXTERIOR_IMAGE_2},
            model_settings.EE_FRAME: [0, 0, 0, 1, 0, 0, 0],
            model_settings.ROTATION_OFFSET: [1, 0, 0, 0],
            ACTION_FPS: 20,
        }
    )
    for name, source in settings[model_settings.OBSERVATION_KEYS].items():
        renamed = f'custom.{name}'
        observation[renamed] = observation.pop(source)
        settings[model_settings.OBSERVATION_KEYS][name] = renamed
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    training = recipes.droid(settings=model_settings.load_settings(path))
    description = json.loads(json.dumps(recipe.inference(model_settings.load_settings(path))))
    codec = spec.from_spec(description)
    assert isinstance(codec, Codec)
    episode = EpisodeContainer({
        name: value if name == settings[model_settings.OBSERVATION_KEYS][gr00t.TASK] else DummySignal([0], [value])
        for name, value in observation.items()
    })
    prepared = training(episode)
    encoded = codec.encode(observation)
    assert prepared[keys.TASK] == encoded[gr00t.LANGUAGE][gr00t.TASK][0][0]
    assert encoded[gr00t.VIDEO]['custom_camera'].shape == (1, 1, 90, 160, 3)
    np.testing.assert_array_equal(prepared['custom_camera'][0][0], encoded[gr00t.VIDEO]['custom_camera'][0, 0])
    np.testing.assert_array_equal(prepared[gr00t.EE_POSE][0][0], encoded[gr00t.STATE][gr00t.EE_POSE][0, 0])
    assert training.meta[MODEL_SETTINGS] == codec.meta[MODEL_SETTINGS] == settings


@pytest.mark.parametrize('config', [server.droid, server.droid_three_cameras])
def test_images_are_bounded_before_remote_without_changing_model_pixels(config, observation):
    pipeline = config()
    local, codec = pipeline.local, pipeline.codec
    assert codec is not None
    resize = next(layer for layer in local._components if isinstance(layer, RestrictImageSize))
    wire_observation = resize.encode(observation)
    settings = codec.meta[MODEL_SETTINGS]
    for source in settings[model_settings.IMAGE_MAPPINGS].values():
        assert wire_observation[source].shape[0] <= settings[model_settings.IMAGE_SIZE][1]
        assert wire_observation[source].shape[1] <= settings[model_settings.IMAGE_SIZE][0]
    direct = codec.encode(observation)
    remote_encoded = codec.encode(wire_observation)
    for name in direct[gr00t.VIDEO]:
        np.testing.assert_array_equal(remote_encoded[gr00t.VIDEO][name], direct[gr00t.VIDEO][name])


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
    encoded = spec.from_spec(recipe.inference(model_settings.load_settings())).encode(observation)
    np.testing.assert_allclose(
        encoded[gr00t.STATE][gr00t.EE_POSE][0, 0], loaded['frame'].compute_eef_9d(upstream_pose), atol=1e-6
    )
    for name, source in {gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE: keys.WRIST_IMAGE}.items():
        expected = loaded['image'].resize_with_pad(observation[source], 180, 320)
        np.testing.assert_array_equal(encoded[gr00t.VIDEO][name][0, 0], expected)
