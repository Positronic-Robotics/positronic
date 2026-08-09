import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from positronic import keys, telemetry, telemetry_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.remote import RemoteSession

# These fixtures stand in for a server, so they spell the handshake fields rather than importing the
# ``keys`` constants the client reads: sharing a constant makes the two agree whatever its value, which
# would leave nothing pinning the client to the wire.
CHUNKED_STACK = {'local_stack': {'name': 'chunked_schedule'}}


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_ws)."""
    mock_ws = _mock_ws_session(metadata)
    if infer_return is not None:
        mock_ws.infer.return_value = infer_return
    policy = RemotePolicy('localhost:0')
    policy._endpoint._client = MagicMock()
    policy._endpoint._client.new_session.return_value = mock_ws
    return policy, mock_ws


def _mock_endpoint(metadata=None, infer_return=None):
    """The bare wire connection, with no declared stack in front of it."""
    policy, mock_ws = _mock_remote_policy(metadata, infer_return)
    return policy._endpoint, mock_ws


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """The border's own settings. Image geometry is the declared stack's business (see RestrictImageSize)."""

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
            keys.TASK: 'pick cube',
        })
        assert isinstance(result['cam'], dict)
        assert isinstance(result['video']['wrist'], dict)
        np.testing.assert_array_equal(result['state'], np.array([1.0, 2.0]))
        assert result[keys.TASK] == 'pick cube'


class TestInferenceClientHeaders:
    def test_default_headers_empty(self):
        assert InferenceClient('localhost:8000').headers is None

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient('localhost:8000', headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers is not None and client.headers['Modal-Key'] == 'k'

    def test_new_session_passes_additional_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession') as mock_session_cls,
        ):
            client = InferenceClient('localhost:8000', headers=headers)
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] == headers
            mock_session_cls.assert_called_once_with(mock_connect.return_value, infer_timeout=DEFAULT_INFER_TIMEOUT)

    def test_new_session_without_headers_passes_none(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost:8000')
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] is None

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': ['m1']}
            client = InferenceClient('localhost:8000', headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert mock_get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': []}
            client = InferenceClient('localhost:8000')
            client.list_models()

            assert mock_get.call_args.kwargs['headers'] is None


class TestInferenceClientUrl:
    """One URL carries host, port, TLS, model id, and session params; headers stay their own argument."""

    def test_bare_host_defaults_to_the_scheme_port(self):
        client = InferenceClient('gpu-host')
        assert client.session_url == 'ws://gpu-host/api/v1/session'
        assert client.api_url == 'http://gpu-host/api/v1'

    def test_explicit_port_is_kept(self):
        client = InferenceClient('localhost:8000')
        assert client.session_url == 'ws://localhost:8000/api/v1/session'
        assert client.api_url == 'http://localhost:8000/api/v1'

    def test_query_rides_along_verbatim(self):
        """Nothing re-encodes the query: 'false' stays the JSON literal whoever wrote the URL meant."""
        client = InferenceClient('gpu-host:9000?codec.fps=10&pad=false')
        assert client.session_url == 'ws://gpu-host:9000/api/v1/session?codec.fps=10&pad=false'
        assert client.api_url == 'http://gpu-host:9000/api/v1'

    def test_tls_scheme_defaults_to_443(self):
        """`https://` is the scheme a fronted endpoint hands out; `wss://` names the same connection."""
        for url in ('https://example.com', 'wss://example.com'):
            client = InferenceClient(url)
            assert client.session_url == 'wss://example.com/api/v1/session'
            assert client.api_url == 'https://example.com/api/v1'

    def test_full_url_keeps_model_id_and_query(self):
        client = InferenceClient('https://gpu-host:8443/api/v1/session/10000?fps=2.5')
        assert client.session_url == 'wss://gpu-host:8443/api/v1/session/10000?fps=2.5'
        assert client.api_url == 'https://gpu-host:8443/api/v1'

    @pytest.mark.parametrize('url', ['gpu-host/', 'http://gpu-host/api/v1/session', 'http://gpu-host/api/v1/session/'])
    def test_url_naming_no_model_is_the_bare_endpoint(self, url):
        assert InferenceClient(url).session_url == 'ws://gpu-host/api/v1/session'

    def test_trailing_slash_belongs_to_the_model_id(self):
        """Sources advertise pinned checkpoint dirs verbatim, and `resolve` matches ids exactly."""
        client = InferenceClient('http://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/')
        assert client.session_url == 'ws://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/'

    def test_model_id_keeps_its_slashes(self):
        client = InferenceClient('http://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID'

    def test_percent_encoding_survives_as_written(self):
        """The server decodes the id whoever handed out the URL meant, so the client normalizes nothing."""
        client = InferenceClient('gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231'

    def test_unexpected_path_rejected(self):
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient('gpu-host:8000/api/v2/other')
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient('gpu-host:8000/api/v1/sessions/10000')

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match='scheme'):
            InferenceClient('ftp://gpu-host:8000')

    def test_every_session_dials_the_session_url(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost:8000/api/v1/session/10000?fps=10')
            client.new_session()
            client.new_session()

            assert mock_connect.call_count == 2
            for call in mock_connect.call_args_list:
                assert call.args[0] == client.session_url == 'ws://localhost:8000/api/v1/session/10000?fps=10'


def test_remote_policy_hands_the_url_and_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    client = RemotePolicy('https://example.com/api/v1/session/10000', headers=headers)._endpoint._client
    assert client is not None
    assert client.session_url == 'wss://example.com/api/v1/session/10000'
    assert client.headers == headers


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        actions = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        endpoint, _ = _mock_endpoint(infer_return=actions)
        wrapped = ActionHorizon(0.5).wrap(endpoint)

        session = wrapped.new_session()
        actions = session({'obs_time_ns': 0})
        assert actions is not None
        assert len(actions) == 3  # 2 within-horizon actions + horizon sentinel
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25
        assert actions[2] == {'timestamp': 0.5}  # horizon sentinel (timestamp = horizon_sec)

    def test_no_truncation_without_horizon(self):
        endpoint, _ = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}])

        session = endpoint.new_session()
        actions = session({})
        assert actions is not None
        assert len(actions) == 2


def test_remote_session_normalizes_single_dict():
    """Server returning a single action dict (legacy shape) is wrapped into a 1-element list."""
    endpoint, _ = _mock_endpoint(infer_return={keys.ROBOT_COMMAND: 'X', 'timestamp': 0.0})

    session = endpoint.new_session()
    actions = session({})
    assert actions == [{keys.ROBOT_COMMAND: 'X', 'timestamp': 0.0}]


def test_remote_session_passes_through_none():
    endpoint, mock_ws = _mock_endpoint()
    mock_ws.infer.return_value = None

    session = endpoint.new_session()
    assert session({}) is None


def test_records_infer_span_without_scheduling_wrapper(tmp_path):
    """The ``policy.infer`` span is recorded at the remote inference boundary itself, not by a wrapper in
    front of it."""
    endpoint, _ = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    session = endpoint.new_session()
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-span'):
        assert session({'obs_time_ns': 0}) is not None
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert [s.name for s in spans] == [telemetry_keys.SPAN_POLICY_INFER]


def test_infer_span_excludes_client_side_image_preparation(tmp_path):
    """``policy.infer`` is the remote round-trip, so JPEG-encoding the observation stays outside it: folding
    client CPU work into the span would inflate the inference percentiles and the policy-server capacity
    estimate the report derives from them."""
    endpoint, _ = _mock_endpoint({'compress_images': True}, infer_return=[])
    session = endpoint.new_session()
    encoded_at: list[int] = []

    def _stamp_encode(image):
        encoded_at.append(time.time_ns())
        return {'jpeg': b''}

    with patch('positronic.policy.remote.encode_jpeg', side_effect=_stamp_encode):
        with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-prep'):
            session({'cam': _make_image(48, 64), 'obs_time_ns': 0})

    (span,) = telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))
    assert span.name == telemetry_keys.SPAN_POLICY_INFER
    assert encoded_at, 'the observation carried an image to compress'
    assert span.start_ns >= encoded_at[-1]  # every encode finishes before the span opens, not inside it


def test_records_infer_span_when_inference_raises(tmp_path):
    """A raising round-trip (a stalled server surfaces ``TimeoutError``) still records its time-to-failure —
    the span is timed in a ``finally`` — and the exception propagates."""
    endpoint, mock_ws = _mock_endpoint()
    mock_ws.infer.side_effect = TimeoutError('server stalled')
    session = endpoint.new_session()
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-raise'):
        with pytest.raises(TimeoutError):
            session({'obs_time_ns': 0})
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert [s.name for s in spans] == [telemetry_keys.SPAN_POLICY_INFER]


def test_remote_policy_meta_exposes_server_fields():
    """RemotePolicy.meta must expose server metadata so SampledPolicy._get_keys
    can read e.g. 'server.checkpoint_path' before a session is created."""
    policy, _ = _mock_remote_policy({'checkpoint_path': '/ckpts/abc', 'model_name': 'foo', **CHUNKED_STACK})

    meta = policy.meta
    assert meta['type'] == 'remote'
    assert meta['server.checkpoint_path'] == '/ckpts/abc'
    assert meta['server.model_name'] == 'foo'


def test_missing_declaration_fails_before_motion():
    """A handshake carrying no ``local_stack`` leaves nothing to build, so no session opens."""
    policy, _ = _mock_remote_policy({'positronic_version': '0.1.0'})
    with pytest.raises(ValueError, match='0.1.0'):
        policy.new_session()


@pytest.mark.parametrize(
    'declared',
    [
        {'seq': []},
        {'name': 'restrict_image_size'},
        {'seq': [{'name': 'chunked_schedule'}] * 2},
        {'seq': [{'name': 'action_timestamp', 'args': {'fps': 10.0}}, {'name': 'chunked_schedule'}]},
        {'seq': [{'seq': [{'name': 'action_horizon', 'args': {'horizon_sec': 0.5}}]}, {'name': 'chunked_schedule'}]},
    ],
    ids=['empty', 'unscheduled', 'double', 'misordered', 'misordered-nested'],
)
def test_unanchored_stack_fails_before_motion(declared):
    """Whatever leaves actions out of wall time: no scheduler, two of them, or one stamping over the anchor."""
    policy, _ = _mock_remote_policy({'local_stack': declared})
    with pytest.raises(ValueError, match='ChunkedSchedule'):
        policy.new_session()


def test_declared_stack_built_at_session_open():
    """The server-declared local stack runs in front of the connection."""
    clock = [1.0]
    policy, mock_ws = _mock_remote_policy(CHUNKED_STACK, infer_return=[{'a': 1, 'timestamp': 0.0}])
    session = policy.new_session(now=lambda: clock[0])
    actions = session({'obs_time_ns': 0})
    assert actions == [{'a': 1, 'timestamp': 1.0}]


def test_unknown_declared_entry_fails_before_motion():
    policy, _ = _mock_remote_policy({'local_stack': {'name': 'run_arbitrary_code'}, 'positronic_version': '9.9.9'})
    with pytest.raises(ValueError, match='9.9.9'):
        policy.new_session()


def test_compression_follows_the_server_declaration():
    """A server behind a message-size cap declares ``remote(compress_images=True)`` and the rig obeys."""
    endpoint, mock_ws = _mock_endpoint({'compress_images': True}, infer_return=[])
    endpoint.new_session()({'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], dict)


def test_frames_stay_raw_where_the_server_declares_no_compression():
    endpoint, mock_ws = _mock_endpoint({'compress_images': False}, infer_return=[])
    endpoint.new_session()({'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], np.ndarray)


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """RemotePolicy against a live server whose pipeline declares a chunked_schedule local stack."""
    host, port = inference_server

    policy = RemotePolicy(f'{host}:{port}')
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
    policy = RemotePolicy(f'{host}:{port}')
    session = policy.new_session(now=lambda: 0.0)

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
