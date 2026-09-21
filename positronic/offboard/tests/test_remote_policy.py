import contextlib
import pathlib
import threading
import time
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from positronic_wire import websocket, wire

from positronic import keys, telemetry, telemetry_keys
from positronic.drivers.roboarm import command
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, DEFAULT_OPEN_TIMEOUT, InferenceClient, _ConnectRetries
from positronic.offboard.tests.conftest import ANSWER_SEC, round_trip
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.remote import prepare_obs
from positronic.policy.spec import PolicySource, remote

# These fixtures stand in for a server, so they spell the handshake fields rather than importing the
# ``keys`` constants the client reads: sharing a constant makes the two agree whatever its value, which
# would leave nothing pinning the client to the wire.
CHUNKED_STACK = {'local_stack': {'name': 'chunked_schedule'}}


class _FakeWire(wire.ClientWire):
    """A client wire that answers each dial from ``outcomes``: a connection to return, or a refusal to raise."""

    NAME = 'fake'
    DEFAULT_PORT = 80

    def __init__(self, *outcomes: wire.ClientConnection | wire.ConnectRefused):
        self._outcomes = list(outcomes)
        self.dials: list[tuple[wire.SessionAddress, Mapping[str, str] | None, float]] = []

    def session_url(self, address: wire.SessionAddress) -> str:
        query = f'?{address.query}' if address.query else ''
        return f'fake://{self.netloc(address)}{address.path}{query}'

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'http://{self.netloc(address)}{wire.API_PATH}'

    def probe(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        return None

    def dial(self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float):
        self.dials.append((address, headers, open_timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, wire.ConnectRefused):
            raise outcome
        return outcome


_ADDRESS = wire.SessionAddress('localhost', 8000, wire.SESSION_PATH, '')


def _mock_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_session)."""
    mock_session = _mock_session(metadata)
    if infer_return is not None:
        mock_session.infer.return_value = infer_return
    policy = RemotePolicy('websocket', 'localhost', 0)
    policy._endpoint._client = MagicMock()
    policy._endpoint._client.new_session.return_value = mock_session
    return policy, mock_session


def _mock_endpoint(metadata=None, infer_return=None):
    """The bare wire connection, with no declared stack in front of it."""
    policy, mock_session = _mock_remote_policy(metadata, infer_return)
    return policy._endpoint, mock_session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """The border's own settings. Image geometry is the declared stack's business (see RestrictImageSize)."""

    def test_images_pass_through_untouched_by_default(self):
        obs = {'cam': _make_image(480, 640), 'state': np.array([1.0])}
        prepared = prepare_obs(obs, compress_images=False)
        assert prepared.keys() == obs.keys()
        assert all(prepared[key] is value for key, value in obs.items())

    def test_compression_reaches_nested_images(self):
        result = prepare_obs(
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


@contextlib.contextmanager
def _catalogue_call(models: list[str]):
    """``httpx.Client`` patched so ``list_models`` reads ``models``. Yields the class and its ``get``."""
    with patch('positronic.offboard.client.httpx.Client') as client_cls:
        get = client_cls.return_value.__enter__.return_value.get
        get.return_value.json.return_value = {'models': models}
        yield client_cls, get


class TestInferenceClientHeaders:
    def test_default_headers_empty(self):
        assert InferenceClient(websocket.WebsocketClientWire(), _ADDRESS).headers is None

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient(websocket.WebsocketClientWire(), _ADDRESS, headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers is not None and client.headers['Modal-Key'] == 'k'

    def test_new_session_dials_with_the_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        conn = MagicMock()
        fake = _FakeWire(conn)
        with patch('positronic.offboard.client.InferenceSession') as mock_session_cls:
            InferenceClient(fake, _ADDRESS, headers=headers).new_session()

        assert fake.dials == [(_ADDRESS, headers, DEFAULT_OPEN_TIMEOUT)]
        assert mock_session_cls.call_args.args[0] is conn
        assert mock_session_cls.call_args.kwargs['infer_timeout'] == DEFAULT_INFER_TIMEOUT

    def test_new_session_without_headers_dials_with_none(self):
        fake = _FakeWire(MagicMock())
        with patch('positronic.offboard.client.InferenceSession'):
            InferenceClient(fake, _ADDRESS).new_session()

        assert fake.dials == [(_ADDRESS, None, DEFAULT_OPEN_TIMEOUT)]

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with _catalogue_call(['m1']) as (_client_cls, get):
            client = InferenceClient(websocket.WebsocketClientWire(), _ADDRESS, headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with _catalogue_call([]) as (_client_cls, get):
            InferenceClient(websocket.WebsocketClientWire(), _ADDRESS).list_models()

            assert get.call_args.kwargs['headers'] is None

    def test_list_models_over_the_network_builds_no_transport_of_its_own(self):
        with _catalogue_call([]) as (client_cls, _get):
            InferenceClient(websocket.WebsocketClientWire(), _ADDRESS).list_models()

            assert client_cls.call_args.kwargs['transport'] is None

    def test_list_models_over_a_socket_reads_the_catalogue_through_that_socket(self):
        """The catalogue is an HTTP route beside the session, so it has to take the socket too."""
        address = _ADDRESS._replace(uds=pathlib.Path('/run/policy.sock'))
        with _catalogue_call([]) as (client_cls, _get):
            InferenceClient(websocket.WebsocketUnixClientWire(), address).list_models()

            transport = client_cls.call_args.kwargs['transport']
            assert isinstance(transport, httpx.HTTPTransport)


def test_every_session_dials_the_same_address():
    address = wire.SessionAddress('localhost', 8000, wire.session_path('10000'), 'fps=10')
    fake = _FakeWire(MagicMock(), MagicMock())
    with patch('positronic.offboard.client.InferenceSession'):
        client = InferenceClient(fake, address)
        client.new_session()
        client.new_session()

    assert [dialed for dialed, _headers, _timeout in fake.dials] == [address, address]
    assert client.session_url == 'fake://localhost:8000/api/v1/session/10000?fps=10'


def _refused(refusal: wire.Refusal) -> wire.ConnectRefused:
    return wire.ConnectRefused(refusal, 'refused')


class TestNewSessionRetriesRefusedConnects:
    """Which refusals are a backend still coming up, and which are the endpoint saying no."""

    def test_a_forbidden_refusal_retries_and_the_session_that_follows_is_returned(self):
        fake = _FakeWire(_refused(wire.Refusal.FORBIDDEN), MagicMock())
        with (
            patch('positronic.offboard.client.InferenceSession') as mock_session_cls,
            patch('positronic.offboard.client.time.sleep'),
        ):
            session = InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == 2
        assert session is mock_session_cls.return_value

    def test_a_forbidden_refusal_gives_up_once_its_attempts_are_spent(self):
        fake = _FakeWire(*[_refused(wire.Refusal.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS + 5))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(wire.ConnectRefused),
        ):
            InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == _ConnectRetries.MAX_FORBIDDEN_ATTEMPTS

    def test_a_final_refusal_is_raised_at_once(self):
        fake = _FakeWire(_refused(wire.Refusal.FINAL))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(wire.ConnectRefused) as refused,
        ):
            InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == 1
        assert refused.value.refusal is wire.Refusal.FINAL

    def test_a_cold_refusal_retries_to_the_deadline(self):
        fake = _FakeWire(_refused(wire.Refusal.COLD))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(TimeoutError, match='fake://localhost:8000'),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=0.0).new_session()

        assert len(fake.dials) == 1

    def test_each_session_opens_on_a_full_budget(self):
        """A client that spent forbidden refusals opening one session still gets all of them for the next."""
        one_session = [_refused(wire.Refusal.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1) + [MagicMock()]
        fake = _FakeWire(*one_session * 2)
        with patch('positronic.offboard.client.InferenceSession'), patch('positronic.offboard.client.time.sleep'):
            client = InferenceClient(fake, _ADDRESS)
            client.new_session()
            client.new_session()

        assert len(fake.dials) == 2 * len(one_session)


def test_a_websocket_port_that_never_answers_is_named_at_the_deadline():
    """Nothing listens on port 1; the refused connect is a backend that is not ready, and the deadline ends it."""
    address = _ADDRESS._replace(port=1)
    client = InferenceClient(websocket.WebsocketClientWire(), address, open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='ws://localhost:1'):
        client.new_session()


def test_remote_policy_hands_the_wire_the_server_the_model_and_the_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    policy = RemotePolicy('websocket_tls', 'example.com', 443, model='10000', query='fps=2.5', headers=headers)
    client = policy._endpoint._client
    assert client.session_url == 'wss://example.com/api/v1/session/10000?fps=2.5'
    assert client.api_url == 'https://example.com/api/v1'
    assert client.headers == headers


def test_a_wire_no_registry_member_carries_is_refused():
    with pytest.raises(ValueError, match="No wire is called 'ws'"):
        RemotePolicy('ws', 'localhost', 8000)


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
    endpoint, mock_session = _mock_endpoint()
    mock_session.infer.return_value = None
    session, rt = open_session(endpoint)

    assert round_trip(session, rt, {}) is None


def test_a_call_while_a_round_trip_is_in_flight_answers_none(open_session):
    """A session never waits. Every call while the round trip is in flight answers ``None``, and none of
    them starts a second round trip."""
    chunk = [{'a': 1, 'timestamp': 0.0}]
    endpoint, mock_session = _mock_endpoint()
    started, release = threading.Event(), threading.Event()

    def blocked(obs):
        started.set()
        assert release.wait(ANSWER_SEC), 'the test never released the round-trip'
        return chunk

    mock_session.infer.side_effect = blocked
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    assert started.wait(ANSWER_SEC), 'the round-trip never started'
    assert session({}, 0) is None
    assert mock_session.infer.call_count == 1

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
    endpoint, mock_session = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()

    assert session({}, 0) is None  # the cancelled answer, read and thrown away
    assert session({}, 0) is None  # a round-trip of its own
    rt.wait(ANSWER_SEC)
    assert mock_session.infer.call_count == 2


def test_a_cancelled_round_trip_still_raises_what_it_failed_with(open_session):
    """A dropped chunk drops no failure. The session reads a cancelled answer, so a stalled server raises
    to the caller that asked for the episode."""
    endpoint, mock_session = _mock_endpoint()
    mock_session.infer.side_effect = TimeoutError('server stalled')
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()

    with pytest.raises(TimeoutError, match='server stalled'):
        session({}, 0)


def test_a_cancel_dies_with_the_answer_it_was_made_against(open_session):
    """A cancel ends with the round trip it was made against, even when that round trip fails. A caller
    that catches the failure and keeps the session gets the next chunk."""
    endpoint, mock_session = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    mock_session.infer.side_effect = [TimeoutError('server stalled'), [{'a': 1, 'timestamp': 0.0}]]
    session, rt = open_session(endpoint)

    assert session({}, 0) is None
    rt.wait(ANSWER_SEC)
    session.cancel()
    with pytest.raises(TimeoutError, match='server stalled'):
        session({}, 0)

    assert round_trip(session, rt, {}) == [{'a': 1, 'timestamp': 0.0}]


def test_closing_a_session_with_a_round_trip_in_flight_is_refused(open_session):
    """A runtime closes before the session it serves. A caller that closes the connection under a round trip
    gets an error that names the order, and not a failure on a dead connection."""
    endpoint, mock_session = _mock_endpoint()
    release = threading.Event()

    def blocked(obs):
        assert release.wait(ANSWER_SEC), 'the test never released the round-trip'
        return None

    mock_session.infer.side_effect = blocked
    session, _rt = open_session(endpoint)

    assert session({}, 0) is None
    with pytest.raises(AssertionError, match='close the runtime'):
        session.close()

    release.set()


def test_records_infer_span_without_scheduling_layer(tmp_path, open_session):
    """The remote inference boundary records ``policy.infer``, and the preparation before it records
    ``policy.prepare``."""
    endpoint, _ = _mock_endpoint(infer_return=[{'a': 1, 'timestamp': 0.0}])
    session, rt = open_session(endpoint)
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-span'):
        assert round_trip(session, rt, {keys.OBS_TIME_NS: 0}) is not None
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert {s.name for s in spans} == {telemetry_keys.SPAN_POLICY_PREPARE, telemetry_keys.SPAN_POLICY_INFER}


def test_infer_span_excludes_client_side_image_preparation(tmp_path, open_session):
    """``policy.infer`` is the remote round-trip, so JPEG-encoding the observation stays outside it: folding
    client CPU work into the span would inflate the inference percentiles and the policy-server capacity
    estimate the report derives from them."""
    endpoint, _ = _mock_endpoint({offboard_keys.COMPRESS_IMAGES: True}, infer_return=[])
    session, rt = open_session(endpoint)
    encoded_at: list[int] = []

    def _stamp_encode(image):
        encoded_at.append(time.time_ns())
        return {'jpeg': b''}

    with patch('positronic.policy.remote.encode_jpeg', side_effect=_stamp_encode):
        with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-prep'):
            round_trip(session, rt, {'cam': _make_image(48, 64), keys.OBS_TIME_NS: 0})

    spans = {s.name: s for s in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    assert encoded_at, 'the observation carried an image to compress'
    # Every encode finishes before the infer span opens, and falls inside the span that does measure it.
    assert spans[telemetry_keys.SPAN_POLICY_INFER].start_ns >= encoded_at[-1]
    prepare = spans[telemetry_keys.SPAN_POLICY_PREPARE]
    assert prepare.start_ns <= encoded_at[0] and prepare.end_ns >= encoded_at[-1]


def test_records_infer_span_when_inference_raises(tmp_path, open_session):
    """A round trip that raises still records the time it took to fail, and the answer raises it again at
    the call that reads it."""
    endpoint, mock_session = _mock_endpoint()
    mock_session.infer.side_effect = TimeoutError('server stalled')
    session, rt = open_session(endpoint)
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-infer-raise'):
        with pytest.raises(TimeoutError):
            round_trip(session, rt, {keys.OBS_TIME_NS: 0})
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    assert telemetry_keys.SPAN_POLICY_INFER in {s.name for s in spans}


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
    policy, mock_session = _mock_remote_policy(CHUNKED_STACK, infer_return=[{'a': 1, 'timestamp': 0.0}])
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
    endpoint, mock_session = _mock_endpoint({offboard_keys.COMPRESS_IMAGES: True}, infer_return=[])
    session, rt = open_session(endpoint)

    round_trip(session, rt, {'cam': _make_image(48, 64)})
    assert isinstance(mock_session.infer.call_args.args[0]['cam'], dict)


def test_frames_stay_raw_where_the_server_declares_no_compression(open_session):
    endpoint, mock_session = _mock_endpoint({offboard_keys.COMPRESS_IMAGES: False}, infer_return=[])
    session, rt = open_session(endpoint)

    round_trip(session, rt, {'cam': _make_image(48, 64)})
    assert isinstance(mock_session.infer.call_args.args[0]['cam'], np.ndarray)


# rules-allow: hardcoded-keys — the command mapping below is spelled the way a server sends it. Reading
# the decoder's own constants would make test and decoder agree whatever those names became, leaving the
# wire itself unpinned.
def test_a_command_crossing_a_live_websocket_arrives_typed(start_server, make_mock_policy, open_session):
    """A command served as a bare mapping — no ``__cmd__`` envelope, the vector a plain sequence — survives a
    real msgpack round trip over the socket and reaches the rig typed, under the stack the handshake declares."""
    pose = [0.4, 0.0, 0.6, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # translation + a 3x3 rotation, the wire's own layout
    wire_action = [{keys.ROBOT_COMMAND: {'type': 'cartesian_pos', 'pose': pose}, 'timestamp': 0.0}]
    policy = make_mock_policy(wire_action, {'model_name': 'm'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy))

    session, rt = open_session(RemotePolicy('websocket', served.host, served.port))
    actions = round_trip(session, rt, {keys.OBS_TIME_NS: 0})

    assert actions is not None, 'the chunk was swallowed before any command reached a driver'
    decoded = actions[0][keys.ROBOT_COMMAND]
    assert isinstance(decoded, command.CartesianPosition), f'the driver would be handed {decoded!r}'
    np.testing.assert_allclose(decoded.pose.translation, [0.4, 0.0, 0.6], atol=1e-6)


def test_remote_policy_lifecycle(inference_server, mock_policy, open_session):
    """RemotePolicy against a live server whose pipeline declares a chunked_schedule local stack."""
    served = inference_server

    policy = RemotePolicy('websocket', served.host, served.port)
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
    served = inference_server
    session, _ = open_session(RemotePolicy('websocket', served.host, served.port))

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
