from pathlib import Path

import pytest

pytest.importorskip('huggingface_hub')

from positronic.vendors.dreamzero import server  # noqa: E402
from positronic.vendors.dreamzero.server import DreamZeroSource, _checkpoint_id, _experiment_name  # noqa: E402

RUN_DIR = 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/'


class _FakeSubprocess:
    def __init__(self, model_path, roboarena_port, **kwargs):
        self.model_path = model_path
        self.roboarena_port = roboarena_port

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
        'checkpoint_path': RUN_DIR + 'checkpoint-100000',
        'experiment_name': 'w22f1_100k_200626',
    }


def test_a_newer_checkpoint_does_not_displace_the_id_being_loaded(latest):
    downloaded = latest('105000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    source.load('100000')

    assert downloaded == [RUN_DIR + 'checkpoint-100000']
    assert source.meta('100000')['checkpoint_path'] == RUN_DIR + 'checkpoint-100000'


def test_a_run_directory_named_like_a_step_still_serves_its_checkpoint(latest):
    downloaded = latest('100000')
    source = DreamZeroSource(model_path='s3://bucket/100000/', backbone='wan2.2')

    source.load(source.get_models()[0])

    assert downloaded == ['s3://bucket/100000/checkpoint-100000']


def test_a_huggingface_repo_is_itself_the_checkpoint():
    assert _checkpoint_id('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'
    assert _experiment_name('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'


def test_a_zero_padded_step_is_reached_by_the_name_its_directory_carries(latest):
    latest('005000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')
    pinned = source.get_models()[0]

    downloaded = latest('010000')
    source.load(pinned)

    assert downloaded == [RUN_DIR + 'checkpoint-005000']


def test_meta_does_not_relist_the_bucket(monkeypatch):
    monkeypatch.setattr(server, 'get_latest_checkpoint', lambda _path, _prefix: pytest.fail('meta must not reach S3'))
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    assert source.meta('100000')['experiment_name'] == 'w22f1_100k_200626'
