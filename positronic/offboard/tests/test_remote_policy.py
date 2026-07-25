import urllib.parse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.remote import RemoteSession
from positronic.policy.wrappers import ChunkedSchedule

EMPTY_STACK = {'local_stack': {'seq': []}}


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None, **kwargs):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_ws)."""
    mock_ws = _mock_ws_session(metadata)
    if infer_return is not None:
        mock_ws.infer.return_value = infer_return
    policy = RemotePolicy('localhost', 0, **kwargs)
    policy._endpoint._client = MagicMock()
    policy._endpoint._client.new_session.return_value = mock_ws
    return policy, mock_ws


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for RemoteSession's optional JPEG compression. Image geometry is the declared
    stack's business (see RestrictImageSize) — the session only compresses."""

    def test_images_pass_through_untouched_by_default(self):
        session = RemoteSession(_mock_ws_session())
        obs = {'cam': _make_image(480, 640), 'state': np.array([1.0])}
        assert session._prepare_obs(obs) is obs

    def test_compression_reaches_nested_images(self):
        session = RemoteSession(_mock_ws_session(), compress_images=True)
        result = session._prepare_obs({
            'cam': _make_image(48, 64),
            'video': {'wrist': _make_image(48, 64)},
            'state': np.array([1.0, 2.0]),
            'task': 'pick cube',
        })
        assert isinstance(result['cam'], dict)
        assert isinstance(result['video']['wrist'], dict)
        np.testing.assert_array_equal(result['state'], np.array([1.0, 2.0]))
        assert result['task'] == 'pick cube'


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
        assert client.headers is not None and client.headers['Modal-Key'] == 'k'

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


class TestInferenceClientParams:
    def test_params_encoded_at_construction(self):
        params = {'codec.fps': 10, 'tag': 'hello'}
        client = InferenceClient('localhost', 8000, params=params)
        assert client._query is not None
        # Encoded once at construction — mutating the caller's dict must not affect the client.
        params['codec.fps'] = 99
        assert urllib.parse.parse_qs(client._query) == {'codec.fps': ['10'], 'tag': ['"hello"']}

    def test_str_params_forwarded_verbatim(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000, params='codec.fps=10&pad=false')
            client.new_session()

            # A raw query string is not re-encoded: 'false' stays the JSON literal the caller wrote.
            assert mock_connect.call_args.args[0] == 'ws://localhost:8000/api/v1/session?codec.fps=10&pad=false'

    def test_new_session_encodes_params_as_json_query(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000, params={'fps': 10, 'flag': False, 'tag': 'true'})
            client.new_session()

            base, query = mock_connect.call_args.args[0].split('?')
            assert base == 'ws://localhost:8000/api/v1/session'
            # Every value is JSON: numbers/bools arrive typed, while the string 'true' stays quoted.
            assert urllib.parse.parse_qs(query) == {'fps': ['10'], 'flag': ['false'], 'tag': ['"true"']}

    def test_new_session_without_params_leaves_uri_bare(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000)
            client.new_session()

            assert '?' not in mock_connect.call_args.args[0]

    def test_new_session_appends_params_after_model_id(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000, params={'fps': 10})
            client.new_session(model_id='m1')

            assert mock_connect.call_args.args[0] == 'ws://localhost:8000/api/v1/session/m1?fps=10'


class TestRemotePolicyHeaderPropagation:
    def test_headers_and_secure_forwarded_to_client(self):
        headers = {'Modal-Key': 'k'}
        policy = RemotePolicy('example.com', 443, headers=headers, secure=True)
        client = policy._endpoint._client
        assert client is not None
        assert client.headers == headers
        assert client.base_uri == 'wss://example.com/api/v1/session'

    def test_default_headers_and_ws_uri(self):
        policy = RemotePolicy('localhost', 8000)
        client = policy._endpoint._client
        assert client is not None
        assert client.headers is None
        assert client.base_uri == 'ws://localhost:8000/api/v1/session'


class TestRemotePolicyParamsPropagation:
    def test_params_forwarded_to_client(self):
        policy = RemotePolicy('localhost', 8000, params={'codec.fps': 10})
        client = policy._endpoint._client
        assert client is not None
        assert client._query == 'codec.fps=10'

    def test_default_params_empty(self):
        policy = RemotePolicy('localhost', 8000)
        client = policy._endpoint._client
        assert client is not None
        assert client._query is None


class TestRemotePolicyFromUrl:
    def test_bare_host_defaults(self):
        policy = RemotePolicy.from_url('gpu-host')
        client = policy._endpoint._client
        assert client is not None
        assert client.base_uri == 'ws://gpu-host:8000/api/v1/session'
        assert client._query is None

    def test_host_port_and_query_verbatim(self):
        policy = RemotePolicy.from_url('gpu-host:9000?codec.fps=10&pad=false')
        client = policy._endpoint._client
        assert client is not None
        assert client.base_uri == 'ws://gpu-host:9000/api/v1/session'
        assert client._query == 'codec.fps=10&pad=false'

    def test_full_url_with_model_id(self):
        policy = RemotePolicy.from_url('https://gpu-host:8443/api/v1/session/10000?fps=2.5')
        client = policy._endpoint._client
        assert client is not None
        assert client.base_uri == 'wss://gpu-host:8443/api/v1/session'
        assert policy._endpoint._model_id == '10000'
        assert client._query == 'fps=2.5'

    def test_session_path_without_model_id(self):
        policy = RemotePolicy.from_url('http://gpu-host/api/v1/session')
        client = policy._endpoint._client
        assert client is not None
        assert policy._endpoint._model_id is None
        assert client.base_uri == 'ws://gpu-host:8000/api/v1/session'

    def test_model_id_keeps_its_slashes(self):
        policy = RemotePolicy.from_url('http://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID')
        assert policy._endpoint._model_id == 'GEAR-Dreams/DreamZero-DROID'

    def test_model_id_is_held_decoded(self):
        """The client percent-encodes the id per session URL, so holding it encoded would double-encode."""
        policy = RemotePolicy.from_url('http://gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231')
        assert policy._endpoint._model_id == 's3://bucket/ckpt#1'

    def test_unexpected_path_rejected(self):
        with pytest.raises(ValueError, match='/api/v1/session'):
            RemotePolicy.from_url('gpu-host:8000/api/v2/other')
        with pytest.raises(ValueError, match='/api/v1/session'):
            RemotePolicy.from_url('gpu-host:8000/api/v1/sessions/10000')

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match='scheme'):
            RemotePolicy.from_url('ftp://gpu-host:8000')


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        actions = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        # Build: ActionHorizon wrapping a RemotePolicy with no local stack of its own
        policy, _ = _mock_remote_policy(EMPTY_STACK, infer_return=actions)
        wrapped = ActionHorizon(0.5).wrap(policy)

        session = wrapped.new_session()
        actions = session({'obs_time_ns': 0})
        assert actions is not None
        assert len(actions) == 3  # 2 within-horizon actions + horizon sentinel
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25
        assert actions[2] == {'timestamp': 0.5}  # horizon sentinel (timestamp = horizon_sec)

    def test_no_truncation_without_horizon(self):
        policy, _ = _mock_remote_policy(
            EMPTY_STACK, infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}]
        )

        session = policy.new_session()
        actions = session({})
        assert actions is not None
        assert len(actions) == 2


def test_remote_session_normalizes_single_dict():
    """Server returning a single action dict (legacy shape) is wrapped into a 1-element list."""
    policy, _ = _mock_remote_policy(EMPTY_STACK, infer_return={'robot_command': 'X', 'timestamp': 0.0})

    session = policy.new_session()
    actions = session({})
    assert actions == [{'robot_command': 'X', 'timestamp': 0.0}]


def test_remote_session_passes_through_none():
    policy, mock_ws = _mock_remote_policy(EMPTY_STACK)
    mock_ws.infer.return_value = None

    session = policy.new_session()
    assert session({}) is None


def test_remote_policy_meta_exposes_server_fields():
    """RemotePolicy.meta must expose server metadata so SampledPolicy._get_keys
    can read e.g. 'server.checkpoint_path' before a session is created."""
    policy, _ = _mock_remote_policy({'checkpoint_path': '/ckpts/abc', 'model_name': 'foo', **EMPTY_STACK})

    meta = policy.meta
    assert meta['type'] == 'remote'
    assert meta['server.checkpoint_path'] == '/ckpts/abc'
    assert meta['server.model_name'] == 'foo'


def test_no_declaration_falls_back_to_chunked_schedule():
    """A server that declares no ``local_stack`` in the handshake gets the standard ChunkedSchedule."""
    clock = [0.0]
    policy, _ = _mock_remote_policy(infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 0.5}])
    session = policy.new_session(now=lambda: clock[0])
    actions = session({'obs_time_ns': 0})
    # ChunkedSchedule anchored the chunk to now=0.0 and gates re-inference until it is consumed.
    assert actions == [{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 0.5}]
    clock[0] = 0.2
    assert session({'obs_time_ns': 0}) is None


def test_declared_stack_built_at_session_open():
    """The server-declared local stack runs in front of the connection."""
    declared = {'local_stack': {'name': 'chunked_schedule'}}
    clock = [1.0]
    policy, mock_ws = _mock_remote_policy(declared, infer_return=[{'a': 1, 'timestamp': 0.0}])
    session = policy.new_session(now=lambda: clock[0])
    actions = session({'obs_time_ns': 0})
    assert actions == [{'a': 1, 'timestamp': 1.0}]


def test_unknown_declared_entry_fails_before_motion():
    policy, _ = _mock_remote_policy({'local_stack': {'name': 'run_arbitrary_code'}, 'positronic_version': '9.9.9'})
    with pytest.raises(ValueError, match='9.9.9'):
        policy.new_session()


def test_operator_local_bypasses_declaration():
    """An operator-supplied ``local`` stack ignores the server declaration entirely."""
    declared = {'local_stack': {'name': 'temporal_stack', 'args': {'keys': ['v'], 'offsets_sec': [0.0]}}}
    policy, _ = _mock_remote_policy(declared, infer_return=[{'a': 1}], local=ChunkedSchedule())
    session = policy.new_session(now=lambda: 5.0)
    actions = session({'obs_time_ns': 0})
    # The declared TemporalStack would have stacked 'v'; instead the operator's ChunkedSchedule ran.
    assert actions == [{'a': 1, 'timestamp': 5.0}]


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """RemotePolicy against a live server whose pipeline declares a chunked_schedule local stack."""
    host, port = inference_server

    policy = RemotePolicy(host, port)
    session = policy.new_session(now=lambda: 0.0)

    meta = session.meta
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    obs = {'dataset': 'test'}
    action = session(obs)
    # Single-dict server response is normalized to a 1-element list (Session contract) and
    # anchored to absolute time by the declared ChunkedSchedule.
    assert action == [{'action_data': [1, 2, 3], 'timestamp': 0.0}]

    session.close()

    # New session
    session2 = policy.new_session(now=lambda: 0.0)
    session2.close()


def test_remote_session_meta(inference_server):
    """Session meta must include server metadata."""
    host, port = inference_server
    policy = RemotePolicy(host, port)
    session = policy.new_session(now=lambda: 0.0)

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
