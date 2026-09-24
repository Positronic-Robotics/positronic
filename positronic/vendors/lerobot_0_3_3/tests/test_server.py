from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect
from starlette.datastructures import QueryParams

from positronic.offboard import server_wire, websocket_wire
from positronic.offboard.protocol import deserialise
from positronic.offboard.spec import PolicyDeployment
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.observation import TASK_FIELD

torch = pytest.importorskip('torch')
pytest.importorskip('lerobot')

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402

from positronic.offboard.server import PolicyServer  # noqa: E402
from positronic.vendors.lerobot_0_3_3 import server as lerobot_server  # noqa: E402
from positronic.vendors.lerobot_0_3_3.policy import LerobotModel, warm_observation  # noqa: E402

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
        self.accept = AsyncMock()
        self._send_bytes = AsyncMock()
        self._close = AsyncMock()

    async def receive_bytes(self):
        raise WebSocketDisconnect()

    async def send_bytes(self, payload):
        await self._send_bytes(payload)

    async def close(self, **kwargs):
        await self._close(**kwargs)

    def as_connection(self) -> websocket_wire.WebsocketServerConnection:
        """What the websocket wire hands the server for one session it has accepted."""
        return websocket_wire.WebsocketServerConnection(self, server_wire.ServedHostPort('localhost', 8000))


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
        device='cpu',
    )
    model = source.load('42')
    assert model.meta() == {
        'type': 'act',
        'checkpoint_path': 's3://bucket/exp/checkpoints/42/pretrained_model',
        'experiment_name': 'exp',
        'device': 'cpu',
    }
    model.close()


def test_act_sessions_reuse_full_chunk_prediction_without_resetting_the_model():
    policy = MagicMock()
    policy.to.return_value = policy
    policy.predict_action_chunk.return_value = torch.zeros((1, 2, 7))
    model = LerobotModel(policy, device='cpu')
    for session_id in ('first', 'second'):
        chunk = model({TASK_FIELD: 'stack'}, session_id=session_id)
        assert len(chunk) == 2
        assert all(step['action'].shape == (7,) for step in chunk)
        model.end_session(session_id)
    assert policy.predict_action_chunk.call_count == 2
    policy.reset.assert_not_called()
    policy.select_action.assert_not_called()
    model.close()


def _make_server(checkpoint: str | None) -> PolicyServer:
    source = lerobot_server.LerobotSource(
        policy_factory=lambda _checkpoint: MagicMock(), checkpoints_dir='s3://bucket/exp', checkpoint=checkpoint
    )
    return PolicyServer(PolicyDeployment(source=source, local=ChunkedSchedule(fps=15)))


@pytest.mark.asyncio
async def test_lerobot_server_uses_configured_checkpoint(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41', '42'])
    model = MagicMock()
    model.meta.return_value = {}
    load = MagicMock(return_value=model)
    monkeypatch.setattr(lerobot_server.LerobotSource, 'load', load)

    server = _make_server(checkpoint='42')
    server._load()
    websocket = _DummyWebSocket()
    await server._serve_session(websocket.as_connection())

    assert load.call_args.args[0] == '42'
    ready = deserialise(websocket._send_bytes.await_args_list[0].args[0])
    assert ready['status'] == 'ready'
    assert ready['meta']['checkpoint_id'] == '42'


def test_lerobot_server_rejects_missing_configured_checkpoint_at_startup(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path: ['41'])
    load = MagicMock()
    monkeypatch.setattr(lerobot_server.LerobotSource, 'load', load)

    server = _make_server(checkpoint='42')

    with pytest.raises(ValueError) as excinfo:
        server._load()

    assert 'Configured checkpoint not found: 42' in str(excinfo.value)
    assert "Available: ['41']" in str(excinfo.value)
    load.assert_not_called()


def test_warmup_observation_matches_the_features_the_policy_declares():
    obs = warm_observation(_act_config())

    assert obs[STATE_FEATURE].shape == (8,)
    # The model callable accepts channels-last images.
    assert obs[CAMERA_FEATURE].shape == (224, 320, 3)
    assert obs[TASK_FIELD] == ''
