import threading
import time
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from positronic import keys, telemetry, telemetry_keys
from positronic.drivers.roboarm import command
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, _ConnectRetries
from positronic.offboard.tests.conftest import ANSWER_SEC, round_trip
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.remote import _prepare_obs
from positronic.policy.spec import PolicySource, remote

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
        obs = {'cam': _make_image(480, 640), 'state': np.array([1.0])}
        prepared = _prepare_obs(obs, compress_images=False)
        assert prepared.keys() == obs.keys()
        assert all(prepared[key] is value for key, value in obs.items())

    def test_compression_reaches_nested_images(self):
        result = _prepare_obs(
            {
                'cam': _make_image(48, 64),
                'video': {'wrist': _make_image(48, 64)},
                'state': np.array([1.0, 2.0]),
                keys.TASK: 'pick cube',
            },
            compress_images=True,
        )
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
            patch('positronic.offboard.client._handshake') as mock_handshake,
        ):
            client = InferenceClient('localhost:8000', headers=headers)
            session = client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] == headers
            assert session._websocket is mock_connect.return_value
            assert session.metadata is mock_handshake.return_value
            assert session._infer_timeout == DEFAULT_INFER_TIMEOUT

    def test_new_session_without_headers_passes_none(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client._handshake'),
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
            patch('positronic.offboard.client._handshake'),
        ):
            client = InferenceClient('localhost:8000/api/v1/session/10000?fps=10')
            client.new_session()
            client.new_session()

            assert mock_connect.call_count == 2
            for call in mock_connect.call_args_list:
                assert call.args[0] == client.session_url == 'ws://localhost:8000/api/v1/session/10000?fps=10'


def _refused(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, 'refused', Headers()))


class TestNewSessionRetriesRefusedUpgrades:
    """Which non-101 upgrade responses are a backend still coming up, and which are the endpoint saying no."""

    def test_a_403_retries_and_the_session_that_follows_is_returned(self):
        with (
            patch(
                'positronic.offboard.client.connect', side_effect=[_refused(HTTPStatus.FORBIDDEN), MagicMock()]
            ) as mock_connect,
            patch('positronic.offboard.client._handshake') as mock_handshake,
            patch('positronic.offboard.client.time.sleep'),
        ):
            session = InferenceClient('localhost:8000').new_session()

            assert mock_connect.call_count == 2
            assert session.metadata is mock_handshake.return_value

    def test_a_403_gives_up_once_its_attempts_are_spent(self):
        with (
            patch(
                'positronic.offboard.client.connect',
                side_effect=[_refused(HTTPStatus.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS + 5),
            ) as mock_connect,
            patch('positronic.offboard.client._handshake'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(InvalidStatus),
        ):
            InferenceClient('localhost:8000').new_session()

        assert mock_connect.call_count == _ConnectRetries.MAX_FORBIDDEN_ATTEMPTS

    @pytest.mark.parametrize('status', [HTTPStatus.UNAUTHORIZED, HTTPStatus.NOT_FOUND])
    def test_a_refusal_that_no_warm_up_clears_is_raised_at_once(self, status):
        with (
            patch('positronic.offboard.client.connect', side_effect=_refused(status)) as mock_connect,
            patch('positronic.offboard.client._handshake'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(InvalidStatus),
        ):
            InferenceClient('localhost:8000').new_session()

        assert mock_connect.call_count == 1

    def test_each_session_opens_on_a_full_budget(self):
        """A client that spent 403s opening one session still gets all of them for the next."""
        one_session = [_refused(HTTPStatus.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1) + [MagicMock()]
        with (
            patch('positronic.offboard.client.connect', side_effect=one_session * 2) as mock_connect,
            patch('positronic.offboard.client._handshake'),
            patch('positronic.offboard.client.time.sleep'),
        ):
            client = InferenceClient('localhost:8000')
            client.new_session()
            client.new_session()

            assert mock_connect.call_count == 2 * len(one_session)


def test_remote_policy_hands_the_url_and_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    client = RemotePolicy('https://example.com/api/v1/session/10000', headers=headers)._endpoint._client
    assert client is not None
    assert client.session_url == 'wss://example.com/api/v1/session/10000'
    assert client.headers == headers


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self, open_session):
        actions = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        endpoint, _ = _mock_endpoint(infer_return=actions)
        session, rt = open_session(ActionHorizon(0.5).wrap(endpoint))

        actions = round_trip(session, rt, {keys.OBS_TIME_NS: 0})
        assert actions is not None
        assert len(actions) == 3  # 2 within-horizon actions + horizon sentinel
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25
        assert actions[2] == {'timestamp': 0.5}  # horizon sentinel (timestamp = horizon_sec)

    def test_no_truncation_without_horizon(self, open_session):
        endpoint, _ = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}])

        session, rt = open_session(endpoint)

        actions = round_trip(session, rt, {})
        assert actions is not None
        assert len(actions) == 2


def test_remote_session_normalizes_single_dict(open_session):
    """Server returning a single action dict is wrapped into a 1-element list."""
    endpoint, _ = _mock_endpoint(infer_return={keys.ROBOT_COMMAND: 'X', 'timestamp': 0.0})
    session, rt = open_session(endpoint)

    assert round_trip(session, rt, {}) == [{keys.ROBOT_COMMAND: 'X', 'timestamp': 0.0}]


def test_remote_session_passes_through_none(open_session):
    endpoint, mock_ws = _mock_endpoint()
    mock_ws.infer.return_value = None
    session, rt = open_session(endpoint)

    assert round_trip(session, rt, {}) is None


def test_a_call_while_a_round_trip_is_in_flight_answers_none(open_session):
    """A session never waits. Every call while the round trip is in flight answers ``None``, and none of
    them starts a second round trip."""
    chunk = [{'a': 1, 'timestamp': 0.0}]
    endpoint, mock_ws = _mock_endpoint()
    started, release = threading.Event(), threading.Event()

    def blocked(obs):
        started.set()
        assert release.wait(ANSWER_SEC), 'the test never released the round-trip'
        return chunk

    mock_ws.infer.side_effect = blocked
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    assert started.wait(ANSWER_SEC), 'the round-trip never started'
    assert session({}, 0) is None
    assert mock_ws.infer.call_count == 1

    release.set()
    rt.wait(ANSWER_SEC)
    assert session({}, 0) == chunk


def test_opening_a_session_without_a_runtime_is_refused():
    """Nothing serves the round trip without a runtime, so the session is refused where it is opened, and not
    at the first observation it is given."""
    endpoint, _ = _mock_endpoint()

    with pytest.raises(ValueError, match='runs its inference on a runtime'):
        endpoint.new_session()


def test_cancel_drops_the_chunk_of_the_round_trip_in_flight(open_session):
    """A cancelled session drops the chunk it waited for, because that chunk applies to a world the cancel
    says has gone, and it asks for a new one."""
    endpoint, mock_ws = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()

    assert session({}, 0) is None  # the cancelled answer, read and thrown away
    assert session({}, 0) is None  # a round-trip of its own
    rt.wait(ANSWER_SEC)
    assert mock_ws.infer.call_count == 2


def test_a_cancelled_round_trip_still_raises_what_it_failed_with(open_session):
    """A dropped chunk drops no failure. The session reads a cancelled answer, so a stalled server raises
    to the caller that asked for the episode."""
    endpoint, mock_ws = _mock_endpoint()
    mock_ws.infer.side_effect = TimeoutError('server stalled')
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()

    with pytest.raises(TimeoutError, match='server stalled'):
        session({}, 0)


def test_a_cancel_dies_with_the_answer_it_was_made_against(open_session):
    """A cancel ends with the round trip it was made against, even when that round trip fails. A caller
    that catches the failure and keeps the session gets the next chunk."""
    endpoint, mock_ws = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    mock_ws.infer.side_effect = [TimeoutError('server stalled'), [{'a': 1, 'timestamp': 0.0}]]
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()
    with pytest.raises(TimeoutError, match='server stalled'):
        session({}, 0)

    assert round_trip(session, rt, {}) == [{'a': 1, 'timestamp': 0.0}]


def test_closing_a_session_with_a_round_trip_in_flight_is_refused(open_session):
    """A runtime closes before the session it serves. A caller that closes the websocket under a round trip
    gets an error that names the order, and not a failure on a dead socket."""
    endpoint, mock_ws = _mock_endpoint()
    release = threading.Event()

    def blocked(obs):
        assert release.wait(ANSWER_SEC), 'the test never released the round-trip'
        return None

    mock_ws.infer.side_effect = blocked
    session, _rt = open_session(endpoint)

    assert session({}, 0) is None
    with pytest.raises(AssertionError, match='close the runtime'):
        session.close()

    release.set()


def test_records_infer_span_without_scheduling_layer(tmp_path, open_session):
    """The ``policy.infer`` span is recorded at the remote inference boundary itself, not by a layer in
    front of it."""
    endpoint, _ = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    session, rt = open_session(endpoint)
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-span'):
        assert round_trip(session, rt, {keys.OBS_TIME_NS: 0}) is not None
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert [s.name for s in spans] == [telemetry_keys.SPAN_POLICY_INFER]


def test_infer_span_excludes_client_side_image_preparation(tmp_path, open_session):
    """``policy.infer`` is the remote round-trip, so JPEG-encoding the observation stays outside it: folding
    client CPU work into the span would inflate the inference percentiles and the policy-server capacity
    estimate the report derives from them."""
    endpoint, _ = _mock_endpoint({'compress_images': True}, infer_return=[])
    session, rt = open_session(endpoint)
    encoded_at: list[int] = []

    def _stamp_encode(image):
        encoded_at.append(time.time_ns())
        return {'jpeg': b''}

    with patch('positronic.policy.remote.encode_jpeg', side_effect=_stamp_encode):
        with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-prep'):
            round_trip(session, rt, {'cam': _make_image(48, 64), keys.OBS_TIME_NS: 0})

    (span,) = telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))
    assert span.name == telemetry_keys.SPAN_POLICY_INFER
    assert encoded_at, 'the observation carried an image to compress'
    assert span.start_ns >= encoded_at[-1]  # every encode finishes before the span opens, not inside it


def test_records_infer_span_when_inference_raises(tmp_path, open_session):
    """A round trip that raises still records the time it took to fail, and the answer raises it again at
    the call that reads it."""
    endpoint, mock_ws = _mock_endpoint()
    mock_ws.infer.side_effect = TimeoutError('server stalled')
    session, rt = open_session(endpoint)
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-raise'):
        with pytest.raises(TimeoutError):
            round_trip(session, rt, {keys.OBS_TIME_NS: 0})
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert [s.name for s in spans] == [telemetry_keys.SPAN_POLICY_INFER]


def test_missing_declaration_fails_before_motion():
    """A handshake carrying no ``local_stack`` leaves nothing to build, so no session opens."""
    policy, _ = _mock_remote_policy({'positronic_version': '0.1.0'})
    with pytest.raises(ValueError, match='0.1.0'):
        policy.new_session()


def test_empty_declaration_fails_before_motion():
    """An empty stack declares nothing to build, so it is refused like an absent one."""
    policy, _ = _mock_remote_policy({'local_stack': {'seq': []}})
    with pytest.raises(ValueError, match='declares no rig-side stack'):
        policy.new_session()


def test_declared_stack_built_at_session_open(open_session):
    """The server-declared local stack runs in front of the connection."""
    policy, mock_ws = _mock_remote_policy(CHUNKED_STACK, infer_return=[{'a': 1, 'timestamp': 0.0}])
    session, rt = open_session(policy)

    assert round_trip(session, rt, {keys.OBS_TIME_NS: 0}, int(1e9)) == [{'a': 1, 'timestamp': 1.0}]


def test_unknown_declared_entry_fails_before_motion():
    policy, _ = _mock_remote_policy({
        'local_stack': {'name': 'run_arbitrary_code'},
        offboard_keys.POSITRONIC_VERSION: '9.9.9',
    })
    with pytest.raises(ValueError, match='9.9.9'):
        policy.new_session()


def test_compression_follows_the_server_declaration(open_session):
    """A server behind a message-size cap declares ``remote(compress_images=True)`` and the rig obeys."""
    endpoint, mock_ws = _mock_endpoint({'compress_images': True}, infer_return=[])
    session, rt = open_session(endpoint)

    round_trip(session, rt, {'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], dict)


def test_frames_stay_raw_where_the_server_declares_no_compression(open_session):
    endpoint, mock_ws = _mock_endpoint({'compress_images': False}, infer_return=[])
    session, rt = open_session(endpoint)

    round_trip(session, rt, {'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], np.ndarray)


# rules-allow: hardcoded-keys — the command mapping below is spelled the way a server sends it. Reading
# the decoder's own constants would make test and decoder agree whatever those names became, leaving the
# wire itself unpinned.
def test_a_command_crossing_a_live_websocket_arrives_typed(start_server, make_mock_policy, open_session):
    """A command served as a bare mapping — no ``__cmd__`` envelope, the vector a plain sequence — survives a
    real msgpack round trip over the socket and reaches the rig typed, under the stack the handshake declares."""
    pose = [0.4, 0.0, 0.6, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # translation + a 3x3 rotation, the wire's own layout
    wire_action = [{keys.ROBOT_COMMAND: {'type': 'cartesian_pos', 'pose': pose}, 'timestamp': 0.0}]
    served = make_mock_policy(wire_action, {'model_name': 'm'})
    host, port, _ = start_server(ChunkedSchedule() | remote | PolicySource(served))

    session, rt = open_session(RemotePolicy(f'{host}:{port}'))
    actions = round_trip(session, rt, {keys.OBS_TIME_NS: 0})

    assert actions is not None, 'the chunk was swallowed before any command reached a driver'
    decoded = actions[0][keys.ROBOT_COMMAND]
    assert isinstance(decoded, command.CartesianPosition), f'the driver would be handed {decoded!r}'
    np.testing.assert_allclose(decoded.pose.translation, [0.4, 0.0, 0.6], atol=1e-6)


def test_remote_policy_lifecycle(inference_server, mock_policy, open_session):
    """RemotePolicy against a live server whose pipeline declares a chunked_schedule local stack."""
    host, port = inference_server

    policy = RemotePolicy(f'{host}:{port}')
    session, rt = open_session(policy)

    meta = session.meta
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    action = round_trip(session, rt, {'dataset': 'test'})
    # Single-dict server response is normalized to a 1-element list (Session contract) and
    # anchored to absolute time by the declared ChunkedSchedule.
    assert action == [{'action_data': [1, 2, 3], 'timestamp': 0.0}]

    session.close()

    # New session
    session2, _ = open_session(policy)
    session2.close()


def test_remote_session_meta(inference_server, open_session):
    """Session meta must include server metadata."""
    host, port = inference_server
    session, _ = open_session(RemotePolicy(f'{host}:{port}'))

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
