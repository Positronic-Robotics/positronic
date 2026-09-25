from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from positronic_wire import roboarena as roboarena_wire
from positronic_wire import wire

pytest.importorskip('huggingface_hub')

from positronic.offboard import keys as offboard_keys  # noqa: E402
from positronic.vendors.dreamzero import roboarena, server  # noqa: E402
from positronic.vendors.dreamzero.server import (  # noqa: E402
    _checkpoint_id,
    _experiment_name,
    _warm_observation,
)

RUN_DIR = 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/'


class _FakeSubprocess:
    def __init__(self, model_path, roboarena_port, **kwargs):
        self.model_path = model_path
        self.roboarena_port = roboarena_port

    def start(self):
        pass

    def warmup(self):
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
        monkeypatch.setattr(server, 'get_latest_checkpoint', lambda _path, prefix='': names[-1])
        return downloaded

    return _set


def test_run_directory_is_served_at_its_latest_step(holding):
    downloaded = holding('95000', '100000')

    model = server.dreamzero_model(model_path=RUN_DIR, backbone='wan2.2')

    assert downloaded == [RUN_DIR + 'checkpoint-100000']
    # rules-allow: hardcoded-keys — the wire spelling is what this asserts; reading it from the same
    # constants the code writes with would pass whatever those constants held.
    assert model.meta() == {
        'checkpoint_id': '100000',
        'type': 'dreamzero',
        'backbone': 'wan2.2',
        'num_gpus': 1,
        'checkpoint_path': RUN_DIR + 'checkpoint-100000',
        'experiment_name': 'w22f1_100k_200626',
    }


def test_a_run_directory_named_like_a_step_still_serves_its_checkpoint(holding):
    downloaded = holding('100000')

    server.dreamzero_model(model_path='s3://bucket/100000/', backbone='wan2.2')

    assert downloaded == ['s3://bucket/100000/checkpoint-100000']


def test_a_huggingface_repo_is_addressed_by_its_whole_name(holding):
    """A repo names no step, so its id keeps its own slash."""
    holding()

    model = server.dreamzero_model(model_path='GEAR-Dreams/DreamZero-DROID')

    assert model.meta()[offboard_keys.CHECKPOINT_ID] == 'GEAR-Dreams/DreamZero-DROID'
    assert _experiment_name('GEAR-Dreams/DreamZero-DROID') == 'DreamZero-DROID'


def test_a_pinned_checkpoint_directory_is_addressed_by_its_step(holding):
    holding()

    model = server.dreamzero_model(model_path='s3://bucket/exp/checkpoint-40000')

    assert model.meta()[offboard_keys.CHECKPOINT_ID] == '40000'
    assert _checkpoint_id('checkpoint-005000') == '005000'


def test_a_zero_padded_step_is_reached_by_the_name_its_directory_carries(holding):
    downloaded = holding('005000')

    model = server.dreamzero_model(model_path=RUN_DIR, backbone='wan2.2')

    assert downloaded == [RUN_DIR + 'checkpoint-005000']
    assert model.meta()[offboard_keys.CHECKPOINT_ID] == '005000'


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


def test_a_server_that_announces_no_resolution_cannot_be_warmed():
    announced = {
        roboarena.RESOLUTION: None,
        roboarena.NEEDS_WRIST_CAMERA: True,
        roboarena.NUM_EXTERIOR_CAMERAS: 1,
        roboarena.NEEDS_STEREO_CAMERA: False,
    }

    with pytest.raises(ValueError, match='no image resolution'):
        _warm_observation(announced, 'session-1')


def test_session_owns_video_cache_and_probe_cannot_reset_it(monkeypatch):
    first, second = Mock(), Mock()
    first.infer.return_value = second.infer.return_value = np.zeros((24, 8))
    monkeypatch.setattr(server, 'RoboarenaClient', Mock(side_effect=[first, second]))
    backend = Mock(roboarena_port=1234)
    model = server.DreamZeroModel(backend, {})
    try:
        assert len(model({}, session_id='first')) == 24
        assert first.infer.call_args.args[0][roboarena.SESSION_ID] == 'first'
        model.end_session('probe')
        first.reset.assert_not_called()
        with pytest.raises(RuntimeError, match='another session'):
            model({}, session_id='second')
        model.end_session('first')
        first.reset.assert_called_once_with(session_id='first')
        first.close.assert_called_once()
        assert len(model({}, session_id='second')) == 24
        model.end_session('second')
        second.reset.assert_called_once_with(session_id='second')
    finally:
        model.close()
    backend.stop.assert_called_once()


@pytest.mark.parametrize(
    'failure',
    [roboarena_wire.TextAnswer('CUDA out of memory'), wire.ConnectRefused(wire.Refusal.FINAL, 'CUDA out of memory')],
)
def test_a_reset_that_fails_is_logged_and_the_client_closed(monkeypatch, caplog, failure):
    client = Mock()
    client.infer.return_value = np.zeros((24, 8))
    client.reset.side_effect = failure
    monkeypatch.setattr(server, 'RoboarenaClient', Mock(return_value=client))
    model = server.DreamZeroModel(Mock(roboarena_port=1234), {})
    model({}, session_id='first')

    model.end_session('first')

    assert 'CUDA out of memory' in caplog.text
    client.close.assert_called_once()
