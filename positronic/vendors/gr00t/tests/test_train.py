import json
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from positronic import keys
from positronic.cfg.ds import transform
from positronic.dataset import Dataset
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.policy.codecs import ACTION, GR00T_MODALITY, GR00T_MODALITY_PATH, LEROBOT_FEATURES, Codec
from positronic.policy.codecs.metadata import MODEL_SETTINGS, MODEL_SETTINGS_PATH
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes, train
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings


@pytest.fixture
def raw_dataset():
    fields = {
        keys.EE_POSE: [[0, 0, 0, 1, 0, 0, 0]] * 2,
        keys.JOINTS: [np.zeros(7), np.ones(7)],
        keys.GRIP: [0.25, 0.75],
        keys.WRIST_IMAGE: [np.zeros((180, 320, 3), dtype=np.uint8)] * 2,
        keys.EXTERIOR_IMAGE: [np.ones((180, 320, 3), dtype=np.uint8)] * 2,
        keys.EXTERIOR_IMAGE_2: [np.full((180, 320, 3), 73, dtype=np.uint8)] * 2,
    }
    episode = EpisodeContainer(
        {**{key: DummySignal([0, 100_000_000], values) for key, values in fields.items()}, keys.TASK: 'pick'},
        meta={'uid': 'recording'},
    )
    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = episode
    dataset.meta = {'source': 'recordings'}
    return dataset


@pytest.fixture
def prepared_dataset(raw_dataset):
    return transform(
        base=raw_dataset,
        transforms=[
            recipes.droid.override(
                settings=model_settings.load_settings(
                    overrides={
                        ACTION_FPS: 20,
                        model_settings.IMAGE_MAPPINGS: model_settings.three_camera_settings()[
                            model_settings.IMAGE_MAPPINGS
                        ],
                    }
                )
            )
        ],
    )


def test_recipe_preserves_dataset_features_labels_and_recording_metadata(prepared_dataset):
    dataset = prepared_dataset
    assert len(dataset) == 1
    assert dataset.meta['source'] == 'recordings'
    assert dataset.meta[ACTION_FPS] == 20
    features = dataset.meta[LEROBOT_FEATURES]
    assert features[ACTION] == {'shape': (17,), 'dtype': 'float32', 'names': ['actions']}
    for name, size in gr00t.STATE_DIMS.items():
        assert features[name] == {'shape': (size,), 'dtype': 'float32'}
    for name in (gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE):
        assert features[name] == {'shape': (180, 320, 3), 'dtype': 'video', 'names': ['height', 'width', 'channel']}
    assert dataset.meta[GR00T_MODALITY][ACTION] == {
        gr00t.EE_POSE: {gr00t.START: 0, gr00t.END: 9},
        gr00t.GRIP: {gr00t.START: 9, gr00t.END: 10},
        gr00t.JOINT_POSITION: {gr00t.START: 10, gr00t.END: 17},
    }
    episode = dataset[0]
    assert episode.meta == {'uid': 'recording'}
    assert episode[keys.TASK] == 'pick'
    action = episode[ACTION]
    assert list(action.timestamps(RECORDED_TIME)) == [0, 100_000_000]
    values = np.asarray(action.values())
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values[:, 9], [0.25, 0.75])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.ones(7)])


def test_training_builds_and_materializes_without_inference_or_codec_training(monkeypatch, raw_dataset):
    def unexpected(*args):
        raise AssertionError('Training must not construct an inference pipeline or read Codec.training_encoder')

    monkeypatch.setattr(recipe, 'inference', unexpected)
    monkeypatch.setattr(Codec, 'training_encoder', property(unexpected))
    dataset = transform(base=raw_dataset, transforms=[recipes.droid_three_cameras])
    assert np.asarray(dataset[0][ACTION].values()).shape == (2, 17)


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('has_settings', [False, True])
def test_finetuning_forwards_dataset_cameras_and_resume_to_gr00t(
    tmp_path, monkeypatch, resume, has_settings, prepared_dataset
):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE]
    (dataset / GR00T_MODALITY_PATH).write_text(json.dumps(prepared_dataset.meta[GR00T_MODALITY]))
    if has_settings:
        (dataset / MODEL_SETTINGS_PATH).write_text(json.dumps(prepared_dataset.meta[MODEL_SETTINGS]))
    output = tmp_path / 'output'
    output.mkdir()
    if not resume:
        (output / MODEL_SETTINGS_PATH).parent.mkdir()
        (output / MODEL_SETTINGS_PATH).write_text(json.dumps({'stale': True}))
    sync = Mock(return_value=output)
    run = Mock(side_effect=lambda *args, **kwargs: (output / 'checkpoint-2').mkdir())
    monkeypatch.setattr(train.pos3, 'download', lambda _: dataset)
    monkeypatch.setattr(train.pos3, 'sync', sync)
    monkeypatch.setattr(train.utils, 'save_run_metadata', Mock())
    monkeypatch.setattr(train.subprocess, 'run', run)
    train.main(input_path=str(dataset), output_path=str(output), exp_name='droid', resume=resume, num_train_steps=2)
    arguments = run.call_args.args[0]
    assert arguments[arguments.index('--base-model-path') + 1] == gr00t.BASE_MODEL
    offset = arguments.index('--video-keys') + 1
    assert arguments[offset : offset + 3] == cameras
    assert ('--resume-from-checkpoint' in arguments) == resume
    assert sync.call_args.kwargs['delete_remote'] == (not resume)
    assert run.call_args.kwargs['check'] is True
    assert (output / MODEL_SETTINGS_PATH).exists() == has_settings
    assert (output / 'checkpoint-2' / MODEL_SETTINGS_PATH).exists() == has_settings
    if has_settings:
        expected_settings = prepared_dataset.meta[MODEL_SETTINGS]
        assert json.loads((output / MODEL_SETTINGS_PATH).read_text()) == expected_settings
        assert json.loads((output / 'checkpoint-2' / MODEL_SETTINGS_PATH).read_text()) == expected_settings


def test_resume_rejects_different_dataset_settings(tmp_path, monkeypatch, prepared_dataset):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / GR00T_MODALITY_PATH).write_text(json.dumps(prepared_dataset.meta[GR00T_MODALITY]))
    (dataset / MODEL_SETTINGS_PATH).write_text(json.dumps(prepared_dataset.meta[MODEL_SETTINGS]))
    output = tmp_path / 'output'
    (output / 'meta').mkdir(parents=True)
    (output / MODEL_SETTINGS_PATH).write_text(json.dumps(model_settings.load_settings()))
    monkeypatch.setattr(train.pos3, 'download', lambda _: dataset)
    monkeypatch.setattr(train.pos3, 'sync', lambda *args, **kwargs: output)
    with pytest.raises(ValueError, match='resumed training run'):
        train.main(input_path=str(dataset), output_path=str(output), exp_name='droid', resume=True)
