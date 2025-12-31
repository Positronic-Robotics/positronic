from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from positronic.offboard.client import InferenceClient
from positronic.offboard.serialisation import deserialise, serialise
from positronic.vendors.lerobot import server as lerobot_server


class _PassthroughCodec:
    def encode(self, obs):
        return obs

    def decode(self, action):
        return action


class _DummyWebSocket:
    def __init__(self):
        self.client = ('test', 0)
        self.accept = AsyncMock()
        self.send_bytes = AsyncMock()
        self.close = AsyncMock()

    async def receive_bytes(self):
        raise WebSocketDisconnect()


def test_inference_client_connect_and_infer(inference_server, mock_policy):
    """Test standard client connection and inference flow."""
    host, port = inference_server
    client = InferenceClient(host, port)

    session = client.new_session()
    try:
        # 1. Verify Metadata Handshake
        assert session.metadata['model_name'] == 'test_model'

        # 2. Verify Inference
        obs = {'image': 'test'}
        action = session.infer(obs)

        assert action['action_data'] == [1, 2, 3]
        mock_policy.select_action.assert_called_with(obs)
    finally:
        session.close()


def test_inference_client_reset(inference_server, mock_policy):
    """Test that starting a new session calls reset on the policy."""
    host, port = inference_server
    client = InferenceClient(host, port)

    # First session (Reset #1)
    session = client.new_session()
    session.close()

    # Second session (Reset #2)
    session = client.new_session()
    session.close()

    assert mock_policy.reset.call_count == 2


def test_inference_client_selects_model_id(multi_policy_server):
    host, port, policies = multi_policy_server
    client = InferenceClient(host, port)

    default_session = client.new_session()
    try:
        assert default_session.metadata['model_name'] == 'alpha'
        action = default_session.infer({'obs': 'default'})
        assert action['action_data'] == ['alpha']
    finally:
        default_session.close()

    beta_session = client.new_session('beta')
    try:
        assert beta_session.metadata['model_name'] == 'beta'
        action = beta_session.infer({'obs': 'beta'})
        assert action['action_data'] == ['beta']
    finally:
        beta_session.close()

    policies['alpha'].select_action.assert_called_with({'obs': 'default'})
    policies['beta'].select_action.assert_called_with({'obs': 'beta'})


def test_wire_serialisation_accepts_mappingproxy():
    backing = {'a': 1, 'b': {'c': 2}}
    frozen = MappingProxyType(backing)
    payload = {'obs': frozen}

    round_trip = deserialise(serialise(payload))

    # mappingproxy is normalized to a plain dict for the wire.
    assert round_trip == {'obs': {'a': 1, 'b': {'c': 2}}}


@pytest.mark.asyncio
async def test_lerobot_server_uses_configured_checkpoint(monkeypatch):
    monkeypatch.setattr(lerobot_server, 'list_checkpoints', lambda _path: ['42'])

    server = lerobot_server.InferenceServer(
        policy_factory=lambda _checkpoint: MagicMock(),
        observation_encoder=_PassthroughCodec(),
        action_decoder=_PassthroughCodec(),
        checkpoints_dir='s3://bucket/exp',
        checkpoint='42',
    )

    requested = {}

    async def fake_get_policy(checkpoint_id: str):
        requested['checkpoint_id'] = checkpoint_id
        policy = MagicMock()
        policy.meta = {'model_name': 'test'}
        return policy

    server.policy_manager.get_policy = fake_get_policy
    server.policy_manager.release_session = AsyncMock()

    websocket = _DummyWebSocket()
    await server.websocket_endpoint(websocket)

    assert requested['checkpoint_id'] == '42'
    server.policy_manager.release_session.assert_awaited_once()
