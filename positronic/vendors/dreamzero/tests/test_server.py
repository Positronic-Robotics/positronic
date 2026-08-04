from pathlib import Path

import pytest

pytest.importorskip('huggingface_hub')

from positronic.vendors.dreamzero import server  # noqa: E402
from positronic.vendors.dreamzero.server import DreamZeroSource, _checkpoint_id, _experiment_name  # noqa: E402

RUN_DIR = 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/'


class _FakeSubprocess:
    def __init__(self, **kwargs):
        self.roboarena_port = kwargs['roboarena_port']

    def start(self, on_progress=None):
        pass

    def stop(self):
        pass


@pytest.fixture
def latest(monkeypatch):
    """Pin what the run directory's newest checkpoint is, and record what ``load`` downloads."""
    downloaded = []
    monkeypatch.setattr(server, 'DreamZeroSubprocess', _FakeSubprocess)
    monkeypatch.setattr(server, '_download_checkpoint', lambda path: downloaded.append(path) or Path('/nonexistent'))

    def _set(step: str) -> list[str]:
        monkeypatch.setattr(server, 'get_latest_checkpoint', lambda _path, _prefix: f'checkpoint-{step}')
        return downloaded

    return _set


def test_run_directory_is_served_at_its_latest_step(latest):
    latest('100000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    assert source.get_models() == ['100000']
    assert source.meta('100000') == {
        'type': 'dreamzero',
        'backbone': 'wan2.2',
        'num_gpus': 1,
        'experiment_name': 'w22f1_100k_200626',
    }


def test_a_newer_checkpoint_does_not_displace_the_id_being_loaded(latest):
    downloaded = latest('105000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    policy = source.load('100000')

    assert downloaded == [RUN_DIR + 'checkpoint-100000']
    assert policy.meta == {'checkpoint_path': RUN_DIR + 'checkpoint-100000'}


def test_a_huggingface_repo_is_itself_the_checkpoint():
    assert _checkpoint_id('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'
    assert _experiment_name('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'


def test_a_zero_padded_step_keeps_its_number_as_the_id():
    assert _checkpoint_id('s3://bucket/exp/checkpoint-005000') == '5000'
