from pathlib import Path

import pytest

pytest.importorskip('huggingface_hub')

from positronic.vendors.dreamzero import roboarena, server  # noqa: E402
from positronic.vendors.dreamzero.server import (  # noqa: E402
    DreamZeroSource,
    _checkpoint_id,
    _experiment_name,
    _warm_observation,
)

RUN_DIR = 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/'


class _FakeSubprocess:
    def __init__(self, model_path, roboarena_port, **kwargs):
        self.model_path = model_path
        self.roboarena_port = roboarena_port

    def start(self, on_progress=None):
        pass

    def warmup(self, on_progress=None):
        pass

    def stop(self):
        pass


@pytest.fixture
def holding(monkeypatch):
    """Pin the checkpoints the run directory holds, oldest first, and record what ``load`` downloads."""
    downloaded = []
    monkeypatch.setattr(server, 'DreamZeroSubprocess', _FakeSubprocess)
    monkeypatch.setattr(server, '_download_checkpoint', lambda path: downloaded.append(path) or Path('/nonexistent'))

    def _set(*steps: str) -> list[str]:
        names = [f'checkpoint-{s}' for s in steps]
        monkeypatch.setattr(server, 'list_checkpoints', lambda _path, prefix='': list(names))
        return downloaded

    return _set


def test_a_run_directory_advertises_every_checkpoint_it_holds(holding):
    holding('95000', '100000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    assert source.get_models() == ['95000', '100000']
    assert source.resolve(None) == '100000'


def test_run_directory_is_served_at_its_latest_step(holding):
    holding('100000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    assert source.get_models() == ['100000']
    # rules-allow: hardcoded-keys — the wire spelling is what this asserts; reading it from the same
    # constants the code writes with would pass whatever those constants held.
    assert source.meta('100000') == {
        'type': 'dreamzero',
        'backbone': 'wan2.2',
        'num_gpus': 1,
        'checkpoint_path': RUN_DIR + 'checkpoint-100000',
        'experiment_name': 'w22f1_100k_200626',
    }


def test_a_newer_checkpoint_does_not_displace_the_id_being_loaded(holding):
    downloaded = holding('100000', '105000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    source.load('100000')

    assert downloaded == [RUN_DIR + 'checkpoint-100000']
    # rules-allow: hardcoded-keys — as above, the spelling is the assertion.
    assert source.meta('100000')['checkpoint_path'] == RUN_DIR + 'checkpoint-100000'


def test_a_run_directory_named_like_a_step_still_serves_its_checkpoint(holding):
    downloaded = holding('100000')
    source = DreamZeroSource(model_path='s3://bucket/100000/', backbone='wan2.2')

    source.load(source.resolve(None))

    assert downloaded == ['s3://bucket/100000/checkpoint-100000']


def test_a_huggingface_repo_is_addressed_by_its_whole_name():
    """The id a client puts in /api/v1/session/<id>, which for a repo keeps its own slash."""
    source = DreamZeroSource(model_path='GEAR-Dreams/DreamZero-DROID')

    assert source.get_models() == ['GEAR-Dreams/DreamZero-DROID']
    assert source.resolve('GEAR-Dreams/DreamZero-DROID') == 'GEAR-Dreams/DreamZero-DROID'
    assert _experiment_name('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'


def test_a_pinned_checkpoint_directory_is_addressed_by_its_step():
    source = DreamZeroSource(model_path='s3://bucket/exp/checkpoint-40000')

    assert source.get_models() == ['40000']
    assert _checkpoint_id('checkpoint-005000') == '005000'


def test_a_zero_padded_step_is_reached_by_the_name_its_directory_carries(holding):
    holding('005000')
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')
    pinned = source.resolve(None)

    downloaded = holding('005000', '010000')
    source.load(pinned)

    assert downloaded == [RUN_DIR + 'checkpoint-005000']


def test_warmup_observation_follows_the_cameras_the_server_announced():
    announced = {
        roboarena.RESOLUTION: (176, 320),
        roboarena.NEEDS_WRIST_CAMERA: True,
        roboarena.NUM_EXTERIOR_CAMERAS: 2,
        roboarena.NEEDS_STEREO_CAMERA: False,
    }

    obs = _warm_observation(announced, 'session-1')

    assert set(obs) == {
        roboarena.JOINT_POSITION,
        roboarena.GRIPPER_POSITION,
        roboarena.PROMPT,
        roboarena.SESSION_ID,
        roboarena.WRIST_IMAGE,
        roboarena.exterior_image(0),
        roboarena.exterior_image(1),
    }
    # The announcement gives the resolution height-first, the way an image array is shaped.
    assert obs[roboarena.WRIST_IMAGE].shape == (176, 320, 3)


def test_warmup_observation_drops_a_camera_the_server_does_not_want():
    announced = {
        roboarena.RESOLUTION: (176, 320),
        roboarena.NEEDS_WRIST_CAMERA: False,
        roboarena.NUM_EXTERIOR_CAMERAS: 1,
        roboarena.NEEDS_STEREO_CAMERA: False,
    }

    obs = _warm_observation(announced, 'session-1')

    assert roboarena.WRIST_IMAGE not in obs
    assert roboarena.exterior_image(1) not in obs


def test_meta_does_not_relist_the_bucket(monkeypatch):
    monkeypatch.setattr(server, 'list_checkpoints', lambda _path, prefix='': pytest.fail('meta must not reach S3'))
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    # rules-allow: hardcoded-keys — as above, the spelling is the assertion.
    assert source.meta('100000')['experiment_name'] == 'w22f1_100k_200626'
