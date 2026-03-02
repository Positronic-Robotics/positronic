from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from positronic.utils.serialization import deserialise

pytest.importorskip('torch')

from positronic.vendors.lerobot_0_3_3 import server as lerobot_server  # noqa: E402


class _PassthroughEncoder:
    def encode(self, obs):
        return obs


class _PassthroughDecoder:
    def decode(self, action, obs=None):
        return action


class _PassthroughCodec:
    def __init__(self):
        self.observation = _PassthroughEncoder()
        self.action = _PassthroughDecoder()


class _DummyWebSocket:
    def __init__(self):
        self.client = ('test', 0)
        self.events = []
        self.accept = AsyncMock()
        self._send_bytes = AsyncMock()
        self._close = AsyncMock()

    async def receive_bytes(self):
        raise WebSocketDisconnect()

    async def send_bytes(self, payload):
        self.events.append('send_bytes')
        await self._send_bytes(payload)

    async def close(self, **kwargs):
        self.events.append('close')
        await self._close(**kwargs)


@pytest.mark.asyncio
async def test_lerobot_server_uses_configured_checkpoint(monkeypatch):
    monkeypatch.setattr(lerobot_server, 'list_checkpoints', lambda _path: ['42'])

    server = lerobot_server.InferenceServer(
        policy_factory=lambda _checkpoint: MagicMock(),
        codec=_PassthroughCodec(),
        checkpoints_dir='s3://bucket/exp',
        checkpoint='42',
    )

    requested = {}

    async def fake_get_policy(checkpoint_id: str, websocket):
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


@pytest.mark.asyncio
async def test_lerobot_server_reports_missing_checkpoint(monkeypatch):
    monkeypatch.setattr(lerobot_server, 'list_checkpoints', lambda _path: ['41'])

    server = lerobot_server.InferenceServer(
        policy_factory=lambda _checkpoint: MagicMock(),
        codec=_PassthroughCodec(),
        checkpoints_dir='s3://bucket/exp',
        checkpoint='42',
    )
    server.policy_manager.get_policy = AsyncMock()
    server.policy_manager.release_session = AsyncMock()

    websocket = _DummyWebSocket()
    await server.websocket_endpoint(websocket)

    assert websocket.events == ['send_bytes', 'close']
    error_payload = websocket._send_bytes.await_args.args[0]
    error_response = deserialise(error_payload)
    assert error_response['status'] == 'error'
    assert 'Configured checkpoint not found: 42' in error_response['error']
    assert "Available: ['41']" in error_response['error']
    server.policy_manager.get_policy.assert_not_called()
    server.policy_manager.release_session.assert_not_called()


@pytest.mark.asyncio
async def test_lerobot_server_reports_unknown_checkpoint_id(monkeypatch):
    monkeypatch.setattr(lerobot_server, 'list_checkpoints', lambda _path: ['41'])

    server = lerobot_server.InferenceServer(
        policy_factory=lambda _checkpoint: MagicMock(),
        codec=_PassthroughCodec(),
        checkpoints_dir='s3://bucket/exp',
        checkpoint=None,
    )
    server.policy_manager.get_policy = AsyncMock()
    server.policy_manager.release_session = AsyncMock()

    websocket = _DummyWebSocket()
    await server.websocket_endpoint(websocket, checkpoint_id='42')

    assert websocket.events == ['send_bytes', 'close']
    error_payload = websocket._send_bytes.await_args.args[0]
    error_response = deserialise(error_payload)
    assert error_response['status'] == 'error'
    assert 'Checkpoint not found: 42' in error_response['error']
    assert "Available: ['41']" in error_response['error']
    server.policy_manager.get_policy.assert_not_called()
    server.policy_manager.release_session.assert_not_called()
