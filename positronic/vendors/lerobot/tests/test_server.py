import threading
from pathlib import Path
from unittest.mock import Mock

import configuronic as cfn
import numpy as np
import pytest
import torch
from positronic_model_server import server_wire, websocket_wire
from positronic_wire import websocket, wire

pytest.importorskip('lerobot', minversion='0.4')

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from positronic import keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import FRANKA_HAND_FRAME, GRIPPER_SITE, parse_ee_frames
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.offboard.spec import PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import ACTION
from positronic.policy.codecs.observation import TASK_FIELD
from positronic.policy.processors import ChunkedSchedule
from positronic.vendors.lerobot import server
from positronic.vendors.lerobot.policy import LerobotModel, warm_observation


@pytest.fixture
def unloaded(monkeypatch):
    """Checkpoints ``41`` and ``42``, and a model built without downloading or warming either."""
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path, prefix='': ['41', '42'])
    monkeypatch.setattr(server.pos3, 'download', lambda path: path)
    monkeypatch.setattr(server, 'LerobotModel', lambda _path, _device, extra_meta: Mock(meta=lambda: extra_meta))
    monkeypatch.setattr(server, 'warm_observation', Mock())
    monkeypatch.setattr(server, 'warmup', Mock())


@pytest.mark.parametrize('configured, expected', [(None, '42'), ('41', '41')])
def test_the_configured_checkpoint_is_served_else_the_latest(unloaded, configured, expected):
    model = server.lerobot_model(checkpoints_dir='s3://bucket/exp', checkpoint=configured, device='cpu')
    assert model.meta()[policy_keys.CHECKPOINT_ID] == expected


def test_a_configured_checkpoint_the_directory_lacks_is_refused(unloaded):
    with pytest.raises(ValueError, match='not found'):
        server.lerobot_model(checkpoints_dir='s3://bucket/exp', checkpoint='43', device='cpu')


def test_warmup_observation_matches_the_features_the_policy_declares():
    state, camera = 'observation.state', 'observation.images.left'
    config = ACTConfig(
        input_features={
            state: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 320)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    obs = warm_observation(config)

    assert obs[state].shape == (8,)
    # Declared channels-first, handed over channels-last the way a session takes it.
    assert obs[camera].shape == (224, 320, 3)
    assert obs[TASK_FIELD] == ''


_STATE, _ENV = 'observation.state', 'observation.environment_state'
_CHUNK = 4


@pytest.fixture(scope='module')
def act_checkpoint(tmp_path_factory) -> Path:
    """A small random ACT checkpoint that plays each chunk of ``_CHUNK`` actions one call at a time."""
    config = ACTConfig(
        input_features={
            _STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            _ENV: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        device='cpu',
    )
    checkpoint = tmp_path_factory.mktemp('act')
    torch.manual_seed(0)
    ACTPolicy(config).save_pretrained(checkpoint)
    for processor in make_pre_post_processors(config):
        processor.save_pretrained(checkpoint)
    return checkpoint


@pytest.fixture
def act_model(act_checkpoint) -> LerobotModel:
    return LerobotModel(str(act_checkpoint), 'cpu')


def _obs(value: float) -> dict:
    return {_STATE: np.full(3, value, np.float32), _ENV: np.zeros(3, np.float32), TASK_FIELD: ''}


def _played(model: LerobotModel, obs: dict, session_id: str, calls: int) -> list[np.ndarray]:
    return [model(obs, session_id=session_id)[0][ACTION] for _ in range(calls)]


def _alone(model: LerobotModel, obs: dict, calls: int) -> list[np.ndarray]:
    """The actions one session plays for ``obs`` while no other session is open."""
    played = _played(model, obs, 'alone', calls)
    model.end_session('alone')
    return played


def test_overlapping_sessions_each_play_their_own_action_chunk(act_model):
    a, b = _obs(1.0), _obs(-1.0)
    expected_a, expected_b = _alone(act_model, a, _CHUNK), _alone(act_model, b, _CHUNK)
    assert not np.allclose(expected_a[0], expected_b[0])

    played_a, played_b = [], []
    for _ in range(_CHUNK):
        played_a += _played(act_model, a, 'a', 1)
        played_b += _played(act_model, b, 'b', 1)

    np.testing.assert_allclose(played_a, expected_a)
    np.testing.assert_allclose(played_b, expected_b)


def test_the_server_answers_overlapping_sessions_each_from_its_own_chunk(act_model):
    a, b = _obs(1.0), _obs(-1.0)
    expected_a, expected_b = _alone(act_model, a, _CHUNK), _alone(act_model, b, _CHUNK)
    policy_server = PolicyServer(lambda: act_model, PolicyDeployment(ChunkedSchedule(fps=15)))
    wires = [websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))]
    ready = threading.Event()
    thread = threading.Thread(target=policy_server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=30)
        served = wires[0].served_address
        assert isinstance(served, server_wire.ServedHostPort)
        address = wire.HostPortAddress('127.0.0.1', served.port, wire.SESSION_PATH, '')
        client = InferenceClient(websocket.WebsocketClientWire(), address)
        session_a, session_b = client.new_session(), client.new_session()
        try:
            played_a, played_b = [], []
            for _ in range(_CHUNK):
                played_a.append(session_a.infer(a)[0][ACTION])
                played_b.append(session_b.infer(b)[0][ACTION])
        finally:
            session_a.close()
            session_b.close()
    finally:
        policy_server.shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive()

    np.testing.assert_allclose(played_a, expected_a)
    np.testing.assert_allclose(played_b, expected_b)


def test_an_ended_session_starts_afresh_and_the_other_sessions_keep_their_chunks(act_model):
    a, b = _obs(1.0), _obs(-1.0)
    expected_a, expected_b = _alone(act_model, a, _CHUNK), _alone(act_model, b, _CHUNK)
    _played(act_model, a, 'a', 2)
    _played(act_model, b, 'b', 2)
    act_model.end_session('probe')
    act_model.end_session('a')
    np.testing.assert_allclose(_played(act_model, b, 'b', 2), expected_b[2:])
    np.testing.assert_allclose(_played(act_model, a, 'a', 1), expected_a[:1])


def test_open_sessions_share_the_loaded_weights(act_model):
    _played(act_model, _obs(1.0), 'a', 1)
    _played(act_model, _obs(-1.0), 'b', 1)
    loaded = [t.data_ptr() for t in [*act_model._policy.parameters(), *act_model._policy.buffers()]]
    for session in act_model._sessions.values():
        assert [t.data_ptr() for t in [*session.policy.parameters(), *session.policy.buffers()]] == loaded


def _pipeline(name: str) -> cfn.Config:
    return server.COMMANDS[name].kwargs['pipeline']


@pytest.mark.parametrize('name', ['serve', 'ee', 'joints'])
def test_a_pose_pipeline_that_states_no_frame_fails_to_build(name):
    with pytest.raises(TypeError, match='ee_frame'):
        _pipeline(name).instantiate()


@pytest.mark.parametrize('name', ['joints_ik', 'joints_ik_sim'])
def test_a_joint_pipeline_builds_with_no_frame(name):
    assert roboarm_keys.EE_FRAMES not in _pipeline(name).instantiate().meta()


def test_the_phail_checkpoint_states_the_franka_hand_frame_it_was_trained_in():
    frame = parse_ee_frames(_pipeline('phail').instantiate().meta()[roboarm_keys.EE_FRAMES])[keys.EE_POSE]
    assert (frame.site, frame.offset) == (GRIPPER_SITE, FRANKA_HAND_FRAME)
