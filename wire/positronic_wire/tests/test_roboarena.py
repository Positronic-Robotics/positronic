"""The client side of the roboarena wire."""

import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
from positronic_wire import roboarena, wire
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidHandshake, InvalidStatus
from websockets.http11 import Response
from websockets.sync.server import ServerConnection, serve

_ADDRESS = roboarena.RoboarenaAddress('a-partner-host', 8000)
_ANNOUNCEMENT = b'\x81\xa8endpoint\xa5infer'


def _refused_upgrade(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, 'refused', Headers()))


@pytest.fixture
def opened(monkeypatch) -> Iterator[MagicMock]:
    """Stands in for the socket a probe opens, so the handshake a test patches runs on it. No proxy applies."""
    monkeypatch.setenv('no_proxy', '*')
    with patch('positronic_wire.roboarena.connected_socket') as connected_socket:
        yield connected_socket


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        (TimeoutError('timed out'), wire.Refusal.SILENT),
        (ConnectionRefusedError(111, 'Connection refused'), wire.Refusal.SILENT),
        (ConnectionClosedError(None, None), wire.Refusal.SILENT),
        (InvalidHandshake('dropped'), wire.Refusal.COLD),
        (socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution'), wire.Refusal.SILENT),
        (ssl.SSLCertVerificationError('unknown issuer'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NONAME, 'Name or service not known'), wire.Refusal.FINAL),
    ],
)
def test_a_handshake_that_does_not_open_is_a_refusal_naming_the_root(raised, refusal):
    with (
        patch('positronic_wire.roboarena.connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused, match='ws://a-partner-host:8000') as refused,
    ):
        roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, wire.Refusal.FORBIDDEN),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
    ],
)
def test_an_edge_in_front_of_the_server_refuses_as_it_does_on_the_websocket_wire(status, refusal):
    """A partner may publish the port behind a proxy, and this wire reads that answer the same way."""
    with (
        patch('positronic_wire.roboarena.connect', side_effect=_refused_upgrade(status)),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal


def test_a_dial_opens_the_bare_root_and_leaves_the_first_frame_unread():
    """The server announces its configuration on connect, and ``dial`` leaves it for ``recv``."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        connection = roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 3.0)

    assert connect.call_args.args == ('ws://a-partner-host:8000',)
    assert connect.call_args.kwargs['open_timeout'] == 3.0
    assert connect.call_args.kwargs['compression'] is None
    assert connect.call_args.kwargs['max_size'] == wire.MAX_MESSAGE_BYTES
    assert connection.recv() == _ANNOUNCEMENT


def test_a_probe_reads_the_announcement_on_the_socket_it_opened_and_closes(opened):
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 3.0) is None

    assert opened.call_args.args == ('a-partner-host', 8000, 3.0)
    assert connect.call_args.args == ('ws://a-partner-host:8000',)
    settings = connect.call_args.kwargs
    assert settings['sock'] is opened.return_value
    assert 0 < settings['open_timeout'] <= 3.0, 'the handshake gets what the connect left'
    assert settings['close_timeout'] == 0
    assert 0 < connect.return_value.recv.call_args.kwargs['timeout'] <= settings['open_timeout']
    connect.return_value.close.assert_called_once()


@pytest.mark.usefixtures('opened')
@pytest.mark.parametrize('verb', [roboarena.RoboarenaClientWire.dial, roboarena.RoboarenaClientWire.probe])
def test_the_handshake_carries_the_headers_the_caller_gives(verb):
    """The caller decides which headers reach the server, so the wire sends exactly the ones it is given."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        verb(roboarena.RoboarenaClientWire(), _ADDRESS, {'Authorization': 'Bearer run-token'}, 3.0)
    assert connect.call_args.kwargs['additional_headers'] == {'Authorization': 'Bearer run-token'}


@pytest.mark.usefixtures('opened')
@pytest.mark.parametrize('verb', [roboarena.RoboarenaClientWire.dial, roboarena.RoboarenaClientWire.probe])
def test_a_caller_giving_no_headers_opens_a_handshake_with_none(verb):
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        verb(roboarena.RoboarenaClientWire(), _ADDRESS, None, 3.0)
    assert connect.call_args.kwargs['additional_headers'] is None


def test_a_port_that_accepts_and_announces_nothing_is_cold(opened):
    """The server upgraded, so it answered; the announcement it has not sent yet is a backend still starting."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = TimeoutError('timed out')
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.COLD
    connect.return_value.close.assert_called_once()


def test_a_probe_reports_the_refusal_a_dial_would_have_raised(opened):
    with patch('positronic_wire.roboarena.connect', side_effect=_refused_upgrade(HTTPStatus.UNAUTHORIZED)):
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.FINAL


def test_a_probe_of_a_port_nothing_answers_on_is_no_answer():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    assert roboarena.RoboarenaClientWire().probe(roboarena.RoboarenaAddress(host, port), None, 1.0) is (
        wire.Refusal.SILENT
    )


def test_a_probe_whose_name_lookup_outlasts_the_timeout_is_no_answer_in_time(monkeypatch):
    def slow_lookup(*_args, **_kwargs):
        time.sleep(3.0)
        raise socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution')

    monkeypatch.setattr(socket, 'getaddrinfo', slow_lookup)
    started = time.monotonic()
    assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 0.3) is wire.Refusal.SILENT
    assert time.monotonic() - started < 2.0, 'the timeout did not cover the lookup'


def test_a_text_frame_raises_the_servers_text():
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'recv.return_value': 'CUDA out of memory'}))
    with pytest.raises(roboarena.TextAnswer, match='CUDA out of memory') as answered:
        connection.recv()
    assert answered.value.text == 'CUDA out of memory'


def test_a_probe_answered_in_text_raises_the_servers_text(opened):
    """A backend that reports a failure is not cold: a retry does not change the text."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = 'CUDA out of memory'
        with pytest.raises(roboarena.TextAnswer, match='CUDA out of memory'):
            roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0)
    connect.return_value.close.assert_called_once()


def test_a_probe_whose_peer_closes_before_the_announcement_is_cold(opened):
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = ConnectionClosedError(None, None)
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.COLD
    connect.return_value.close.assert_called_once()


def test_a_read_on_a_closed_connection_says_the_peer_ended_the_session():
    closed = ConnectionClosedError(None, None)
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'recv.side_effect': closed}))
    with pytest.raises(wire.PeerDisconnected) as ended:
        connection.recv()
    assert ended.value.__cause__ is closed


def test_a_send_on_a_closed_connection_says_the_peer_ended_the_session():
    closed = ConnectionClosedError(None, None)
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'send.side_effect': closed}))
    with pytest.raises(wire.PeerDisconnected) as ended:
        connection.send(b'')
    assert ended.value.__cause__ is closed


@pytest.mark.parametrize(
    ('address', 'session_url'),
    [
        (roboarena.RoboarenaAddress('a-partner-host', 8000), 'ws://a-partner-host:8000'),
        (roboarena.RoboarenaAddress('127.0.0.1', 9000), 'ws://127.0.0.1:9000'),
        (roboarena.RoboarenaAddress('::1', 9000), 'ws://[::1]:9000'),
    ],
)
def test_the_wire_names_the_root_it_dials_with_the_port_the_partner_published(address, session_url):
    """No port is left out: a roboarena server publishes no default, so every address states one."""
    assert roboarena.RoboarenaClientWire().session_url(address) == session_url


def test_the_address_carries_the_host_and_the_port_alone():
    """The protocol routes on a key inside each frame, so a session names no route and no params."""
    address = roboarena.RoboarenaAddress('a-partner-host', 8000)
    assert (address.path, address.query) == ('', '')
    assert address.at_root() is address
    assert roboarena.RoboarenaClientWire().ADDRESS is roboarena.RoboarenaAddress


def test_a_keepalive_is_refused_without_a_dial():
    with patch('positronic_wire.roboarena.connect') as dialled, pytest.raises(wire.KeepaliveUnsupported):
        roboarena.RoboarenaClientWire().keepalive(_ADDRESS, None, 1.0)
    dialled.assert_not_called()


# ─── a roboarena server on loopback ──────────────────────────────────────────


def _announcing(first: str | bytes):
    """A roboarena handler that sends ``first`` as its first frame, then waits for the client to go."""

    def handler(connection: ServerConnection) -> None:
        try:
            connection.send(first)
            connection.recv()
        except ConnectionClosed:
            pass

    return handler


@pytest.fixture
def roboarena_on() -> Iterator[Callable[..., roboarena.RoboarenaAddress]]:
    """Serves a roboarena server on loopback until the test ends."""
    servers = []

    def start(handler) -> roboarena.RoboarenaAddress:
        server = serve(handler, '127.0.0.1', 0)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return roboarena.RoboarenaAddress(*server.socket.getsockname())

    yield start
    for server in servers:
        server.shutdown()


def test_a_server_that_announces_itself_answers_the_probe(roboarena_on):
    assert roboarena.RoboarenaClientWire().probe(roboarena_on(_announcing(b'config')), None, 5.0) is None


def test_a_server_that_announces_a_failure_in_text_raises_it(roboarena_on):
    with pytest.raises(roboarena.TextAnswer, match='the policy failed to load'):
        roboarena.RoboarenaClientWire().probe(roboarena_on(_announcing('the policy failed to load')), None, 5.0)


def _tunnelling_proxy(asked: list[bytes]) -> str:
    """A loopback proxy that opens one tunnel and refuses the upgrade sent in it with 401. Returns its URL."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)

    def tunnel() -> None:
        with listener, listener.accept()[0] as conn:
            asked.append(conn.recv(4096))
            conn.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
            asked.append(conn.recv(4096))
            conn.sendall(b'HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n')

    threading.Thread(target=tunnel, daemon=True).start()
    host, port = listener.getsockname()
    return f'http://{host}:{port}'


def test_a_probe_goes_through_the_proxy_the_environment_names(monkeypatch):
    """Nothing listens on the target; only the proxy, which refuses the upgrade as an edge would."""
    asked: list[bytes] = []
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    monkeypatch.setenv('ws_proxy', _tunnelling_proxy(asked))
    monkeypatch.setenv('no_proxy', 'unrelated.invalid')
    assert roboarena.RoboarenaClientWire().probe(roboarena.RoboarenaAddress(host, port), None, 5.0) is (
        wire.Refusal.FINAL
    )
    assert asked[0].startswith(f'CONNECT {host}:{port} HTTP/1.1\r\n'.encode())
