from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from starlette.datastructures import QueryParams

from positronic.offboard.server import PolicyServer
from positronic.policy.spec import remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.serialization import deserialise

pytest.importorskip('lerobot', minversion='0.4')

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402

from positronic.vendors.lerobot.policy import TASK, warm_observation  # noqa: E402
from positronic.vendors.lerobot.server import LerobotSource  # noqa: E402


class _DummyWebSocket:
    def __init__(self):
        self.client = ('test', 0)
        self.query_params = QueryParams()
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
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['42'])

    server = PolicyServer(ChunkedSchedule() | remote | LerobotSource('s3://bucket/exp', checkpoint='42'))

    requested = {}

    async def fake_get_policy(checkpoint_id: str, websocket=None):
        requested['checkpoint_id'] = checkpoint_id
        policy = MagicMock()
        policy.meta = {'model_name': 'test'}
        policy.new_session.return_value.meta = {}
        return policy

    server._manager.get_policy = fake_get_policy
    server._manager.release_session = AsyncMock()
    await server._startup()

    websocket = _DummyWebSocket()
    await server.default_session(websocket)

    assert requested['checkpoint_id'] == '42'
    assert websocket.events == ['send_bytes']
    ready = deserialise(websocket._send_bytes.await_args.args[0])
    assert ready['status'] == 'ready'
    assert ready['meta']['checkpoint_id'] == '42'
    server._manager.release_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_lerobot_server_reports_missing_checkpoint(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])

    server = PolicyServer(ChunkedSchedule() | remote | LerobotSource('s3://bucket/exp', checkpoint='42'))
    server._manager.get_policy = AsyncMock()

    with pytest.raises(ValueError, match=r"Configured checkpoint not found: 42. Available: \['41'\]"):
        await server._startup()

    server._manager.get_policy.assert_not_called()


@pytest.mark.asyncio
async def test_lerobot_server_reports_unknown_checkpoint_id(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])
    monkeypatch.setattr('positronic.utils.checkpoints.get_latest_checkpoint', lambda _path: '41')

    server = PolicyServer(ChunkedSchedule() | remote | LerobotSource('s3://bucket/exp'))
    server._manager.get_policy = AsyncMock(return_value=MagicMock())
    server._manager.release_session = AsyncMock()
    await server._startup()
    server._manager.get_policy.reset_mock()

    websocket = _DummyWebSocket()
    await server.model_session(websocket, '42')

    assert websocket.events == ['send_bytes', 'close']
    error_response = deserialise(websocket._send_bytes.await_args.args[0])
    assert error_response['status'] == 'error'
    assert 'Checkpoint not found: 42' in error_response['error']
    assert "Available: ['41']" in error_response['error']
    server._manager.get_policy.assert_not_called()
    server._manager.release_session.assert_not_called()


def test_warmup_observation_matches_the_features_the_policy_declares():
    config = ACTConfig(
        input_features={
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            'observation.images.left': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 320)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    obs = warm_observation(config)

    assert obs['observation.state'].shape == (8,)
    # Declared channels-first, handed over channels-last the way a session takes it.
    assert obs['observation.images.left'].shape == (224, 320, 3)
    assert obs[TASK] == ''
