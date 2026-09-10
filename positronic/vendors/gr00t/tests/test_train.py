import json
from unittest.mock import Mock

import pytest

from positronic.vendors import gr00t
from positronic.vendors.gr00t import train


@pytest.mark.parametrize('resume', [False, True])
def test_finetuning_forwards_dataset_cameras_and_resume_to_gr00t(tmp_path, monkeypatch, resume):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE]
    (dataset / 'meta' / 'modality.json').write_text(json.dumps({gr00t.VIDEO: dict.fromkeys(cameras, {})}))
    output = tmp_path / 'output'
    output.mkdir()
    sync = Mock(return_value=output)
    run = Mock()
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
