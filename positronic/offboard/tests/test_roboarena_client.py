"""The roboarena client, driven against a fake connection."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from positronic_model_server.serialization import deserialize, serialize
from positronic_wire import roboarena as roboarena_wire
from positronic_wire import wire
from websockets.exceptions import ConnectionClosedError

from positronic import telemetry, telemetry_keys
from positronic.offboard import roboarena
from positronic.offboard.client import ConnectRetries


def _client(connection) -> roboarena.RoboarenaClient:
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    client._connection = connection
    return client


_ANNOUNCEMENT = serialize({'resolution': [180, 320]})


def _dialing(*connections):
    """The roboarena wire, patched to open each of ``connections`` in turn. Each one announces a config first."""
    for connection in connections:
        connection.recv.side_effect = [_ANNOUNCEMENT, *connection.recv.side_effect]
    return patch.object(roboarena_wire.RoboarenaClientWire, 'dial', side_effect=list(connections))


# Each way an exchange on an open connection fails, as the connection raises it.
_FAILURES = {
    'send-disconnect': ({'send.side_effect': wire.PeerDisconnected('gone')}, wire.PeerDisconnected),
    'read-timeout': ({'recv.side_effect': TimeoutError('timed out')}, TimeoutError),
    'read-disconnect': ({'recv.side_effect': wire.PeerDisconnected('gone')}, wire.PeerDisconnected),
    'read-text': ({'recv.side_effect': roboarena_wire.TextAnswer('CUDA out of memory')}, roboarena_wire.TextAnswer),
}


@pytest.mark.parametrize('failure', _FAILURES.values(), ids=_FAILURES.keys())
@pytest.mark.parametrize(
    'exchange',
    [lambda client: client.infer({}), lambda client: client.reset(session_id='an-episode')],
    ids=['infer', 'reset'],
)
def test_a_failed_exchange_drops_the_connection_and_raises(exchange, failure):
    """A reply after the failure stays queued, so no later exchange may read it as its own."""
    effects, raised = failure
    connection = MagicMock(**effects)
    client = _client(connection)

    with pytest.raises(raised):
        exchange(client)

    connection.close.assert_called_once()
    assert client._connection is None


@pytest.mark.parametrize(
    ('exchange', 'limit'),
    [
        (lambda client: client.infer({}), roboarena.INFER_TIMEOUT_S),
        (lambda client: client.reset(session_id='an-episode'), roboarena.RESET_TIMEOUT_S),
    ],
    ids=['infer', 'reset'],
)
def test_the_send_and_the_answer_of_an_exchange_share_its_timeout(exchange, limit, fake_clock):
    """Each half fits the timeout alone, and the two together do not."""
    connection = fake_clock.connection((limit * 0.6, serialize({})), send_s=limit * 0.6)

    with pytest.raises(TimeoutError):
        exchange(_client(connection))

    assert fake_clock.now == limit


@pytest.mark.parametrize(
    'failure', [TimeoutError('timed out'), wire.PeerDisconnected('gone'), roboarena_wire.TextAnswer('x')]
)
def test_a_handshake_that_fails_closes_the_connection_it_opened_and_raises(failure):
    connection = MagicMock(**{'recv.side_effect': failure})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        with pytest.raises(type(failure)):
            client.connect()

    connection.close.assert_called_once()
    assert client._connection is None


# What a wall-clock test grants past a deadline: a thread start and a closed socket, on a loaded machine.
_DEADLINE_SLACK_S = 2.0


def test_a_port_that_never_accepts_ends_the_connect_by_its_deadline(never_accepting_port):
    client = roboarena.RoboarenaClient('localhost', never_accepting_port, connect_deadline=0.5)
    started = time.monotonic()
    with pytest.raises(wire.ConnectRefused):
        client.connect()

    assert time.monotonic() - started < 0.5 + _DEADLINE_SLACK_S


def test_a_server_that_never_announces_its_config_ends_the_connect_by_its_deadline(silent_websocket_port):
    client = roboarena.RoboarenaClient('localhost', silent_websocket_port, connect_deadline=0.5)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.connect()

    assert time.monotonic() - started < 0.5 + _DEADLINE_SLACK_S


def test_a_server_that_answers_no_close_ends_the_connect_by_its_deadline(unanswering_websocket_port):
    client = roboarena.RoboarenaClient('localhost', unanswering_websocket_port, connect_deadline=0.5)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.connect()

    assert time.monotonic() - started < 0.5 + _DEADLINE_SLACK_S


def test_the_default_deadline_gives_the_dial_and_the_config_read_a_handshake_timeout_each(fake_clock):
    """The server answers the dial and announces its config each just inside one handshake timeout."""
    almost = roboarena.HANDSHAKE_TIMEOUT_S - 1.0
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    def dial(address, headers, open_timeout):
        assert fake_clock.answer(almost, open_timeout), 'the dial timed out'
        return fake_clock.connection((almost, _ANNOUNCEMENT))

    with patch.object(roboarena_wire.RoboarenaClientWire, 'dial', side_effect=dial):
        assert client.connect() == deserialize(_ANNOUNCEMENT)

    assert fake_clock.now == 2 * almost


def test_the_client_hands_its_headers_to_every_handshake_it_opens():
    """A server that checks a token reads it on the handshake, so connect and probe carry the same headers."""
    headers = {'Authorization': 'Bearer run-token'}
    client = roboarena.RoboarenaClient('a-partner-host', 8000, headers)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = MagicMock(**{'recv.return_value': _ANNOUNCEMENT})
        client_wire.probe.return_value = None
        client.connect()
        client.probe()

    assert client_wire.dial.call_args.args[1] == headers
    assert client_wire.probe.call_args.args[1] == headers


def test_a_client_given_no_headers_sends_none():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = MagicMock(**{'recv.return_value': _ANNOUNCEMENT})
        client_wire.probe.return_value = None
        client.connect()
        client.probe()

    assert client_wire.dial.call_args.args[1] is None
    assert client_wire.probe.call_args.args[1] is None


def test_an_inference_with_no_connection_dials_one_and_keeps_it():
    connection = MagicMock(**{'recv.side_effect': [serialize([0.0] * 8)]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with _dialing(connection):
        assert client.infer({}) == [0.0] * 8

    assert client._connection is connection
    connection.close.assert_not_called()


def test_a_reset_with_no_connection_dials_one_and_sends_the_keyed_reset():
    connection = MagicMock(**{'recv.side_effect': [_ANNOUNCEMENT, serialize({'ok': True})]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        client.reset(session_id='an-episode')

    sent = deserialize(connection.send.call_args.args[0])
    assert sent == {roboarena.ENDPOINT: roboarena.RESET, roboarena.SESSION_ID: 'an-episode'}


def test_a_reset_with_no_connection_raises_when_the_server_refuses_the_dial():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.side_effect = wire.ConnectRefused(wire.Refusal.COLD, 'refused')
        with pytest.raises(wire.ConnectRefused):
            client.reset(session_id='an-episode')


def test_a_reset_acknowledged_in_text_ends_the_session():
    websocket = MagicMock(**{'recv.return_value': roboarena.RESET_ACKNOWLEDGEMENT})

    client = _client(roboarena_wire.RoboarenaClientConnection(websocket))

    client.reset(session_id='an-episode')

    sent = deserialize(websocket.send.call_args.args[0])
    assert sent == {roboarena.ENDPOINT: roboarena.RESET, roboarena.SESSION_ID: 'an-episode'}
    assert client._connection is None


def test_a_reset_answered_with_error_text_reaches_the_caller():
    """Error text propagates; it does not count as the reset acknowledgement."""
    websocket = MagicMock(**{'recv.return_value': 'CUDA out of memory'})

    client = _client(roboarena_wire.RoboarenaClientConnection(websocket))

    with pytest.raises(roboarena_wire.TextAnswer, match='CUDA out of memory'):
        client.reset(session_id='an-episode')

    assert client._connection is None


def test_an_error_text_that_ends_with_the_acknowledgement_reaches_the_caller():
    """Only the exact acknowledgement ends the reset. Text that ends with it is still a failure."""
    near_miss = f'expected {roboarena.RESET_ACKNOWLEDGEMENT}'
    websocket = MagicMock(**{'recv.return_value': near_miss})

    with pytest.raises(roboarena_wire.TextAnswer, match=near_miss):
        _client(roboarena_wire.RoboarenaClientConnection(websocket)).reset()


def test_a_reset_the_peer_never_answered_reaches_the_caller():
    closed = ConnectionClosedError(None, None)
    websocket = MagicMock(**{'recv.side_effect': closed})

    with pytest.raises(wire.PeerDisconnected) as ended:
        _client(roboarena_wire.RoboarenaClientConnection(websocket)).reset()

    assert ended.value.__cause__ is closed


def test_a_reset_acknowledged_in_a_frame_keeps_the_connection():
    connection = MagicMock(**{'recv.return_value': serialize({'ok': True})})
    client = _client(connection)

    client.reset()

    assert deserialize(connection.send.call_args.args[0]) == {roboarena.ENDPOINT: roboarena.RESET}
    connection.close.assert_not_called()
    assert client._connection is connection


def test_a_readiness_poll_waits_far_less_than_a_handshake():
    """Readiness reads on the short probe timeout, not the handshake's."""
    assert roboarena.READY_PROBE_TIMEOUT_S < roboarena.HANDSHAKE_TIMEOUT_S

    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = None
        assert client.probe() is None
    assert client_wire.probe.call_args.args[2] == roboarena.READY_PROBE_TIMEOUT_S


def test_readiness_raises_the_text_a_server_answers_in():
    """A backend that reports a failure is not a backend still starting, so a readiness poll does not retry it."""
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch('positronic_wire.roboarena.connected_socket'), patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = 'CUDA out of memory'
        with pytest.raises(roboarena_wire.TextAnswer, match='CUDA out of memory'):
            client.probe()


def test_a_peer_that_closes_before_announcing_is_not_ready():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch('positronic_wire.roboarena.connected_socket'), patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = ConnectionClosedError(None, None)
        assert client.probe() is wire.Refusal.COLD


def test_readiness_raises_a_final_refusal():
    """A retry does not change a permanent refusal, so a readiness poll does not wait out its deadline."""
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = wire.Refusal.FINAL
        with pytest.raises(wire.ConnectRefused) as refused:
            client.probe()
    assert refused.value.refusal is wire.Refusal.FINAL


def test_readiness_raises_a_forbidden_refusal_once_its_attempts_are_spent():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = wire.Refusal.FORBIDDEN
        for _ in range(ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1):
            assert client.probe() is wire.Refusal.FORBIDDEN
        with pytest.raises(wire.ConnectRefused) as refused:
            client.probe()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN


@pytest.mark.parametrize('refusal', [wire.Refusal.COLD, wire.Refusal.SILENT])
def test_a_backend_not_up_yet_stays_not_ready_past_the_forbidden_attempts(refusal):
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = refusal
        for _ in range(ConnectRetries.MAX_FORBIDDEN_ATTEMPTS + 1):
            assert client.probe() is refusal


def _probes_at(client: roboarena.RoboarenaClient, *answers: tuple[float, wire.Refusal | None]) -> list:
    """Probe `client` once at each instant, with the wire answering as given. Returns each probe's result."""
    clock = [0.0]
    results = []
    with (
        patch('positronic.offboard.client.time.monotonic', side_effect=lambda: clock[0]),
        patch.object(client, '_wire') as client_wire,
    ):
        for now, answer in answers:
            clock[0], client_wire.probe.return_value = now, answer
            try:
                results.append(client.probe())
            except wire.ConnectRefused as e:
                results.append(e)
    return results


def test_readiness_raises_an_unresolved_host_at_the_limit_the_client_was_given():
    client = roboarena.RoboarenaClient('a-partner-host', 8000, unresolved_host_deadline=60.0)

    *waiting, surfaced = _probes_at(client, *[(at, wire.Refusal.UNRESOLVED) for at in (0.0, 59.0, 60.0)])

    assert waiting == [wire.Refusal.UNRESOLVED] * 2
    assert isinstance(surfaced, wire.ConnectRefused) and surfaced.refusal is wire.Refusal.UNRESOLVED


def test_a_server_that_answers_ends_the_run_of_unresolved_probes():
    """The name resolves at 200 s and stops resolving at 250 s: the limit counts from 250 s, not from 0 s."""
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    results = _probes_at(
        client,
        (0.0, wire.Refusal.UNRESOLVED),
        (200.0, None),
        (250.0, wire.Refusal.UNRESOLVED),
        (549.0, wire.Refusal.UNRESOLVED),
        (550.0, wire.Refusal.UNRESOLVED),
    )

    assert results[:4] == [wire.Refusal.UNRESOLVED, None, wire.Refusal.UNRESOLVED, wire.Refusal.UNRESOLVED]
    assert isinstance(results[4], wire.ConnectRefused)


def test_a_server_that_answers_gives_the_next_forbidden_refusals_their_full_attempts():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    forbidden = [(0.0, wire.Refusal.FORBIDDEN)] * (ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1)

    results = _probes_at(client, *forbidden, (0.0, None), *forbidden)

    assert not any(isinstance(result, wire.ConnectRefused) for result in results)


def _wire_sends(tmp_path) -> list:
    spans = telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))
    return sorted((s for s in spans if s.name == telemetry_keys.SPAN_WIRE_SEND), key=lambda s: s.start_ns)


def _connection_marks(send) -> tuple:
    return (
        send.attrs[telemetry_keys.ATTR_WIRE_NEW_CONNECTION],
        send.attrs.get(telemetry_keys.ATTR_WIRE_ENDPOINT),
        telemetry_keys.ATTR_WIRE_CONNECT_MS in send.attrs,
    )


def test_each_inference_records_its_send_and_its_wait_for_the_answer(tmp_path):
    answer = serialize([0.0] * 8)
    connection = MagicMock(**{'recv.side_effect': [answer]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with _dialing(connection), telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-roboarena-wire'):
        client.infer({'prompt': 'pick'})

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    [send] = [s for s in spans if s.name == telemetry_keys.SPAN_WIRE_SEND]
    [recv] = [s for s in spans if s.name == telemetry_keys.SPAN_WIRE_RECV]
    assert send.attrs[telemetry_keys.ATTR_WIRE_BYTES] == len(connection.send.call_args.args[0])
    assert send.end_ns == recv.start_ns


def test_only_the_first_send_on_a_roboarena_connection_carries_the_dialed_endpoint_and_connect_time(tmp_path):
    answer = serialize([0.0] * 8)
    first = MagicMock(**{'recv.side_effect': [answer, answer]})
    second = MagicMock(**{'recv.side_effect': [answer]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with _dialing(first, second), telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-roboarena-connection'):
        client.infer({})
        client.infer({})
        client.close()
        client.infer({})

    assert [_connection_marks(send) for send in _wire_sends(tmp_path)] == [
        (True, client.url, True),
        (False, None, False),
        (True, client.url, True),
    ]


def test_an_inference_sent_while_another_waits_for_its_answer_is_not_sent_alone():
    """The first call waits for its answer until the second has sent, so the second goes out behind it."""
    first_waiting, second_sent, release = threading.Event(), threading.Event(), threading.Event()
    sends = []
    answer = serialize([0.0] * 8)

    def send(data):
        sends.append(data)
        if len(sends) == 2:
            second_sent.set()

    def recv(timeout):
        first_waiting.set()
        assert release.wait(timeout=5)
        return answer

    client = _client(MagicMock(**{'send.side_effect': send, 'recv.side_effect': recv}))
    marks = {}

    def call(name):
        client.infer({})
        marks[name] = client.sent_alone

    first = threading.Thread(target=call, args=('first',))
    first.start()
    assert first_waiting.wait(timeout=5)
    second = threading.Thread(target=call, args=('second',))
    second.start()
    assert second_sent.wait(timeout=5)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert marks == {'first': True, 'second': False}


def test_an_inference_that_fails_before_its_send_carries_no_alone_mark():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.side_effect = wire.ConnectRefused(wire.Refusal.COLD, 'refused')
        with pytest.raises(wire.ConnectRefused):
            client.infer({})

    assert client.sent_alone is None
