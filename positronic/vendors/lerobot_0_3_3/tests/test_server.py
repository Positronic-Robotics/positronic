from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from starlette.datastructures import QueryParams

from positronic.offboard.protocol import deserialise
from positronic.policy.executor import blocking
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import remote

pytest.importorskip('torch')

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402

from positronic.offboard import PolicyServer  # noqa: E402
from positronic.policy.observation import TASK_FIELD
from positronic.vendors.lerobot_0_3_3 import server as lerobot_server  # noqa: E402
from positronic.vendors.lerobot_0_3_3.policy import warm_observation  # noqa: E402

STATE_FEATURE = 'observation.state'
CAMERA_FEATURE = 'observation.images.left'


def _act_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            STATE_FEATURE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            CAMERA_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 320)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


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


def test_handshake_metadata_does_not_depend_on_the_factory(monkeypatch):
    """A factory's whole contract is returning a policy, so a plain one carrying no extra attributes
    still yields complete metadata — ``checkpoint_path`` included, since sampling keys on it."""
    monkeypatch.setattr(lerobot_server.pos3, 'download', lambda path: path)
    # A mock cannot answer an inference, but the warm observation is still built from what the factory returned,
    # so the load reaches no checkpoint on disk.
    monkeypatch.setattr(lerobot_server, 'warmup', lambda *_args, **_kwargs: None)
    source = lerobot_server.LerobotSource(
        policy_factory=lambda _path: MagicMock(spec=lerobot_server.PreTrainedPolicy, config=_act_config()),
        checkpoints_dir='s3://bucket/exp',
    )
    session = blocking(source.load('42')).new_session()
    assert session.meta == {'type': 'act', 'checkpoint_path': 's3://bucket/exp/checkpoints/42/pretrained_model'}
    session.close()


def _make_server(checkpoint: str | None) -> PolicyServer:
    source = lerobot_server.LerobotSource(
        policy_factory=lambda _checkpoint: MagicMock(), checkpoints_dir='s3://bucket/exp', checkpoint=checkpoint
    )
    return PolicyServer(ChunkedSchedule() | remote | source)


@pytest.mark.asyncio
async def test_lerobot_server_uses_configured_checkpoint(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['42'])

    server = _make_server(checkpoint='42')

    requested = {}

    async def fake_get_policy(checkpoint_id: str, websocket=None):
        requested['checkpoint_id'] = checkpoint_id
        policy = MagicMock()
        policy.new_session.return_value.meta = {}
        return policy

    server._manager.get_policy = fake_get_policy
    server._manager.release_session = AsyncMock()

    await server._startup()
    websocket = _DummyWebSocket()
    await server.default_session(websocket)

    assert requested['checkpoint_id'] == '42'
    ready = deserialise(websocket._send_bytes.await_args_list[0].args[0])
    assert ready['status'] == 'ready'
    assert ready['meta']['checkpoint_id'] == '42'
    server._manager.release_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_lerobot_server_rejects_missing_configured_checkpoint_at_startup(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])

    server = _make_server(checkpoint='42')
    server._manager.get_policy = AsyncMock()

    with pytest.raises(ValueError) as excinfo:
        await server._startup()

    assert 'Configured checkpoint not found: 42' in str(excinfo.value)
    assert "Available: ['41']" in str(excinfo.value)
    server._manager.get_policy.assert_not_called()


@pytest.mark.asyncio
async def test_lerobot_server_reports_unknown_checkpoint_id(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])
    # Startup pins the default checkpoint via the latest-branch, which lists through get_latest_checkpoint.
    monkeypatch.setattr('positronic.utils.checkpoints.get_latest_checkpoint', lambda _path: '41')

    server = _make_server(checkpoint=None)
    server._manager.get_policy = AsyncMock(return_value=MagicMock())
    server._manager.release_session = AsyncMock()

    await server._startup()
    server._manager.get_policy.reset_mock()

    websocket = _DummyWebSocket()
    await server.model_session(websocket, '42')

    assert websocket.events == ['send_bytes', 'close']
    error_payload = websocket._send_bytes.await_args.args[0]
    error_response = deserialise(error_payload)
    assert error_response['status'] == 'error'
    assert 'Checkpoint not found: 42' in error_response['error']
    assert "Available: ['41']" in error_response['error']
    server._manager.get_policy.assert_not_called()
    server._manager.release_session.assert_not_called()


def test_warmup_observation_matches_the_features_the_policy_declares():
    obs = warm_observation(_act_config())

    assert obs[STATE_FEATURE].shape == (8,)
    # Declared channels-first, handed over channels-last the way a session takes it.
    assert obs[CAMERA_FEATURE].shape == (224, 320, 3)
    assert obs[TASK_FIELD] == ''
