from unittest.mock import MagicMock

from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


class TestRemotePolicyHeaderPropagation:
    def test_headers_and_secure_forwarded_to_client(self):
        headers = {'Modal-Key': 'k'}
        policy = RemotePolicy('example.com', 443, headers=headers, secure=True)
        assert policy._client.headers == headers
        assert policy._client.base_uri == 'wss://example.com/api/v1/session'

    def test_defaults_match_pre_existing_behaviour(self):
        policy = RemotePolicy('localhost', 8000)
        assert policy._client.headers is None
        assert policy._client.base_uri == 'ws://localhost:8000/api/v1/session'


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        mock_ws = _mock_ws_session()
        mock_ws.infer.return_value = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        # Build: ActionHorizon wrapping a RemotePolicy
        policy = RemotePolicy('localhost', 0)
        policy._client = MagicMock()
        policy._client.new_session.return_value = mock_ws
        wrapped = ActionHorizon(0.5).wrap(policy)

        session = wrapped.new_session()
        actions = session({'obs_time_ns': 0})
        assert len(actions) == 3  # 2 within-horizon actions + horizon sentinel
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25
        assert actions[2] == {'timestamp': 0.5}  # horizon sentinel (timestamp = horizon_sec)

    def test_no_truncation_without_horizon(self):
        mock_ws = _mock_ws_session()
        mock_ws.infer.return_value = [{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}]
        policy = RemotePolicy('localhost', 0)
        policy._client = MagicMock()
        policy._client.new_session.return_value = mock_ws

        session = policy.new_session()
        actions = session({})
        assert len(actions) == 2


def test_remote_session_normalizes_single_dict():
    """Server returning a single action dict (legacy shape) is wrapped into a 1-element list."""
    mock_ws = _mock_ws_session()
    mock_ws.infer.return_value = {'robot_command': 'X', 'timestamp': 0.0}
    policy = RemotePolicy('localhost', 0)
    policy._client = MagicMock()
    policy._client.new_session.return_value = mock_ws

    session = policy.new_session()
    actions = session({})
    assert actions == [{'robot_command': 'X', 'timestamp': 0.0}]


def test_remote_session_passes_through_none():
    mock_ws = _mock_ws_session()
    mock_ws.infer.return_value = None
    policy = RemotePolicy('localhost', 0)
    policy._client = MagicMock()
    policy._client.new_session.return_value = mock_ws

    session = policy.new_session()
    assert session({}) is None


def test_remote_policy_meta_exposes_server_fields():
    """RemotePolicy.meta must expose server metadata so SampledPolicy._get_keys
    can read e.g. 'server.checkpoint_path' before a session is created."""
    mock_ws = _mock_ws_session({'checkpoint_path': '/ckpts/abc', 'model_name': 'foo'})
    policy = RemotePolicy('localhost', 0)
    policy._client = MagicMock()
    policy._client.new_session.return_value = mock_ws

    meta = policy.meta
    assert meta['type'] == 'remote'
    assert meta['server.checkpoint_path'] == '/ckpts/abc'
    assert meta['server.model_name'] == 'foo'


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """Test RemotePolicy.new_session() and session call."""
    host, port = inference_server

    policy = RemotePolicy(host, port)
    session = policy.new_session()

    meta = session.meta
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    obs = {'dataset': 'test'}
    action = session(obs)
    # Single-dict server response is normalized to a 1-element list (Session contract).
    assert action == [{'action_data': [1, 2, 3]}]

    session.close()

    # New session
    session2 = policy.new_session()
    session2.close()


def test_remote_policy_chunking(inference_server):
    """Test that RemotePolicy session returns action chunks correctly."""
    host, port = inference_server

    policy = RemotePolicy(host, port)
    session = policy.new_session()

    # Inject a mock ws session
    mock_ws = MagicMock()
    chunk = [{'a': 1}, {'a': 2}, {'a': 3}]
    mock_ws.infer.return_value = chunk
    mock_ws.metadata = {'model_name': 'test_model'}
    session._session = mock_ws

    actions = session({'obs': 1})
    assert actions == chunk
    assert mock_ws.infer.call_count == 1

    new_chunk = [{'b': 1}]
    mock_ws.infer.return_value = new_chunk
    actions2 = session({'obs': 2})
    assert actions2 == new_chunk
    assert mock_ws.infer.call_count == 2

    session.close()


def test_remote_session_meta(inference_server):
    """Session meta must include server metadata."""
    host, port = inference_server
    policy = RemotePolicy(host, port)
    session = policy.new_session()

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
