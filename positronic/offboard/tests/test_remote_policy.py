from unittest.mock import MagicMock, patch

import numpy as np

from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.remote import RemoteSession


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for RemoteSession._prepare_obs image resize logic."""

    def test_server_tuple_resizes_all_images(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(240, 320), 'state': np.array([1.0])}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], obs['state'])

    def test_server_dict_resizes_per_key(self):
        sizes = {'cam_a': (64, 48), 'cam_b': (32, 24)}
        session = RemoteSession(_mock_ws_session({'image_sizes': sizes}), resize=None)
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (24, 32, 3)

    def test_fallback_resize_scales_by_max_dim(self):
        session = RemoteSession(_mock_ws_session(), resize=160)
        obs = {'cam': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam'].shape == (120, 160, 3)

    def test_no_resize_when_already_correct_size(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        img = _make_image(48, 64)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_no_resize_without_server_sizes_or_fallback(self):
        session = RemoteSession(_mock_ws_session(), resize=None)
        img = _make_image(480, 640)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_drops_client_only_model_keys(self):
        """The harness injects ``urdf``/``control_frame`` for client-side frame codecs; they must never wire out."""
        session = RemoteSession(_mock_ws_session(), resize=None)
        result = session._prepare_obs({'state': np.array([1.0]), 'urdf': '<robot/>', 'control_frame': 'end_effector'})
        assert 'urdf' not in result and 'control_frame' not in result
        np.testing.assert_array_equal(result['state'], np.array([1.0]))

    def test_normalizes_list_to_tuple(self):
        """Wire format (msgpack) turns tuples into lists — must normalize."""
        session = RemoteSession(_mock_ws_session({'image_sizes': [64, 48]}), resize=None)
        assert session._default_image_size == (64, 48)
        assert isinstance(session._default_image_size, tuple)

    def test_normalizes_dict_values(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': {'cam_a': [64, 48], 'cam_b': [32, 24]}}), resize=None)
        assert session._image_sizes == {'cam_a': (64, 48), 'cam_b': (32, 24)}
        assert all(isinstance(v, tuple) for v in session._image_sizes.values())

    def test_non_image_values_pass_through(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = session._prepare_obs(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True


class TestInferenceClientHeaders:
    def test_default_headers_empty_and_ws_scheme(self):
        client = InferenceClient('localhost', 8000)
        assert client.headers is None
        assert client.base_uri == 'ws://localhost:8000/api/v1/session'
        assert client.api_url == 'http://localhost:8000/api/v1'

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient('localhost', 8000, headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers['Modal-Key'] == 'k'

    def test_secure_switches_scheme_and_omits_default_port(self):
        client = InferenceClient('example.com', 443, secure=True)
        assert client.base_uri == 'wss://example.com/api/v1/session'
        assert client.api_url == 'https://example.com/api/v1'

    def test_secure_keeps_non_default_port(self):
        client = InferenceClient('example.com', 8443, secure=True)
        assert client.base_uri == 'wss://example.com:8443/api/v1/session'
        assert client.api_url == 'https://example.com:8443/api/v1'

    def test_insecure_omits_default_port(self):
        client = InferenceClient('example.com', 80, secure=False)
        assert client.base_uri == 'ws://example.com/api/v1/session'
        assert client.api_url == 'http://example.com/api/v1'

    def test_new_session_passes_additional_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession') as mock_session_cls,
        ):
            client = InferenceClient('localhost', 8000, headers=headers)
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] == headers
            mock_session_cls.assert_called_once_with(mock_connect.return_value, infer_timeout=DEFAULT_INFER_TIMEOUT)

    def test_new_session_without_headers_omits_additional_headers(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000)
            client.new_session()

            mock_connect.assert_called_once()
            assert 'additional_headers' not in mock_connect.call_args.kwargs

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': ['m1']}
            client = InferenceClient('localhost', 8000, headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert mock_get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': []}
            client = InferenceClient('localhost', 8000)
            client.list_models()

            assert mock_get.call_args.kwargs['headers'] is None


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
        assert len(actions) == 2
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25

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
