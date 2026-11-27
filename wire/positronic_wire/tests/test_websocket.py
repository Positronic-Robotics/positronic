"""The client side of the websocket wire."""

import dataclasses
import datetime
import errno
import ipaddress
import json
import socket
import ssl
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from positronic_wire import roboarena, websocket, wire
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, InvalidMessage, InvalidStatus
from websockets.http11 import Response
from websockets.sync.client import connect, unix_connect
from websockets.sync.connection import Connection
from websockets.sync.server import serve
from websockets.utils import accept_key

_ADDRESS = wire.HostPortAddress('localhost', 8000, wire.SESSION_PATH, '')
_WEBSOCKET = websocket.WebsocketClientWire()
# A header a caller sends with its call.
_HEADER = ('X-Caller', 'SECRET-VALUE')
_ALIVE = json.dumps({wire.ALIVE_SECONDS: 60}).encode()


def _refused_upgrade(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, 'refused', Headers()))


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, wire.Refusal.FORBIDDEN),
        (HTTPStatus.TOO_MANY_REQUESTS, wire.Refusal.COLD),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.BAD_GATEWAY, wire.Refusal.COLD),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
    ],
)
def test_a_non_101_answer_to_the_upgrade_says_what_the_server_is(status, refusal):
    refused_upgrade = _refused_upgrade(status)
    with (
        patch('positronic_wire.websocket.connect', side_effect=refused_upgrade),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is refused_upgrade


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        (TimeoutError('timed out'), wire.Refusal.SILENT),
        (ssl.SSLError('reset'), wire.Refusal.SILENT),
        (ConnectionClosedError(None, None), wire.Refusal.SILENT),
        (InvalidMessage('did not receive a valid HTTP response'), wire.Refusal.SILENT),
        (InvalidHandshake('dropped'), wire.Refusal.COLD),
        (ConnectionRefusedError(111, 'Connection refused'), wire.Refusal.SILENT),
        (socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution'), wire.Refusal.UNRESOLVED),
        (ssl.SSLCertVerificationError('unknown issuer'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NONAME, 'Name or service not known'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NODATA, 'No address associated with hostname'), wire.Refusal.FINAL),
    ],
)
def test_a_handshake_that_does_not_open_is_a_refusal_naming_the_url(raised, refusal):
    with (
        patch('positronic_wire.websocket.connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused, match='ws://localhost:8000/api/v1/session') as refused,
    ):
        websocket.WebsocketClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


def test_a_dial_carries_the_headers_on_the_handshake():
    with patch('positronic_wire.websocket.connect') as connect:
        websocket.WebsocketClientWire().dial(_ADDRESS, {'Modal-Key': 'k'}, 3.0)
    assert connect.call_args.kwargs['additional_headers'] == {'Modal-Key': 'k'}


def test_a_dial_negotiates_no_deflate_with_a_server_that_offers_it():
    """A stock websockets server offers permessage-deflate, and the session still opens uncompressed."""
    negotiated = []

    def echo(connection):
        negotiated.append(connection.protocol.extensions)
        connection.send(connection.recv())

    with serve(echo, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        connection = websocket.WebsocketClientWire().dial(
            dataclasses.replace(_ADDRESS, host='127.0.0.1', port=port), None, 5.0
        )
        try:
            connection.send(b'frame')
            assert connection.recv(timeout=5.0) == b'frame'
        finally:
            connection.close()
        server.shutdown()
    assert negotiated == [[]]


@pytest.fixture
def opened(monkeypatch) -> Iterator[MagicMock]:
    """Stands in for the socket a probe opens, so the handshake a test patches runs on it."""
    with patch('positronic_wire.websocket.connected_socket') as connected_socket:
        yield connected_socket


def test_a_probe_asks_the_host_root_with_the_headers_on_the_socket_it_opened(opened):
    with patch('positronic_wire.websocket.connect', side_effect=_refused_upgrade(HTTPStatus.FORBIDDEN)) as connect:
        probed = websocket.WebsocketClientWire().probe(
            dataclasses.replace(_ADDRESS, query='fps=10'), dict([_HEADER]), 3.0
        )
    assert probed is None
    assert opened.call_args.args == ('localhost', 8000, 3.0)
    assert connect.call_args.args == ('ws://localhost:8000',)
    settings = connect.call_args.kwargs
    assert settings['sock'] is opened.return_value
    assert settings['additional_headers'] == dict([_HEADER])
    assert 0 < settings['open_timeout'] <= 3.0, 'the handshake gets what the connect left'
    assert settings['close_timeout'] == 0


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, None),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
        (HTTPStatus.BAD_GATEWAY, wire.Refusal.COLD),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.TOO_MANY_REQUESTS, wire.Refusal.COLD),
    ],
)
def test_a_probe_reads_the_servers_own_403_as_the_server_and_every_other_status_as_dial_does(status, refusal, opened):
    with patch('positronic_wire.websocket.connect', side_effect=_refused_upgrade(status)):
        assert websocket.WebsocketClientWire().probe(_ADDRESS, None, 1.0) is refusal


def test_a_probe_reads_a_handshake_that_opened_as_the_server(opened):
    with patch('positronic_wire.websocket.connect') as connect:
        assert websocket.WebsocketClientWire().probe(_ADDRESS, None, 1.0) is None
    connect.return_value.close.assert_called_once()


def test_a_probe_of_a_port_nothing_answers_on_is_no_answer():
    assert websocket.WebsocketClientWire().probe(_at(*_unbound()), None, 1.0) is wire.Refusal.SILENT


def test_the_tls_member_dials_wss_and_probes_it(opened):
    with patch('positronic_wire.websocket.connect') as connect:
        websocket.WebsocketTlsClientWire().dial(_ADDRESS, None, 1.0)
        websocket.WebsocketTlsClientWire().probe(_ADDRESS, None, 1.0)
    assert [call.args[0] for call in connect.call_args_list] == [
        'wss://localhost:8000/api/v1/session',
        'wss://localhost:8000',
    ]


@pytest.mark.parametrize(
    ('client_wire', 'address', 'session_url'),
    [
        (websocket.WebsocketClientWire(), _ADDRESS, 'ws://localhost:8000/api/v1/session'),
        (websocket.WebsocketClientWire(), dataclasses.replace(_ADDRESS, port=80), 'ws://localhost/api/v1/session'),
        (websocket.WebsocketTlsClientWire(), dataclasses.replace(_ADDRESS, port=443), 'wss://localhost/api/v1/session'),
        (
            websocket.WebsocketTlsClientWire(),
            dataclasses.replace(_ADDRESS, port=8443),
            'wss://localhost:8443/api/v1/session',
        ),
        (websocket.WebsocketClientWire(), dataclasses.replace(_ADDRESS, host='::1'), 'ws://[::1]:8000/api/v1/session'),
        (
            websocket.WebsocketClientWire(),
            dataclasses.replace(_ADDRESS, host='127.0.0.1'),
            'ws://127.0.0.1:8000/api/v1/session',
        ),
        (
            websocket.WebsocketClientWire(),
            dataclasses.replace(_ADDRESS, query='codec.fps=10&pad=false'),
            'ws://localhost:8000/api/v1/session?codec.fps=10&pad=false',
        ),
    ],
)
def test_the_member_spells_the_session_and_leaves_out_its_default_port(client_wire, address, session_url):
    assert client_wire.session_url(address) == session_url


def test_the_socket_wire_names_the_socket_it_dials_and_claims_no_authority():
    """A socket names no host and no port, so the log names the socket and the handshake a stand-in."""
    address = wire.UnixSocketAddress(Path('/run/policy.sock'), wire.SESSION_PATH, 'fps=10')
    unix = websocket.WebsocketUnixClientWire()

    assert unix.session_url(address) == 'ws+unix:///run/policy.sock/api/v1/session?fps=10'
    assert unix.handshake_url(address) == 'ws://localhost/api/v1/session?fps=10'


def test_each_member_declares_the_address_it_dials():
    """A caller builds the address its wire names, and the other wire's address is a type error."""
    assert websocket.WebsocketClientWire().ADDRESS is wire.HostPortAddress
    assert websocket.WebsocketTlsClientWire().ADDRESS is wire.HostPortAddress
    assert websocket.WebsocketUnixClientWire().ADDRESS is wire.UnixSocketAddress


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        # Reached a live socket: a server that is starting or restarting raises each of these.
        (TimeoutError('handshake timed out'), wire.Refusal.SILENT),
        (ConnectionResetError(104, 'Connection reset by peer'), wire.Refusal.SILENT),
        (ConnectionAbortedError(103, 'Software caused connection abort'), wire.Refusal.SILENT),
        (BrokenPipeError(32, 'Broken pipe'), wire.Refusal.SILENT),
        (FileNotFoundError(2, 'No such file or directory'), wire.Refusal.SILENT),
        # This process's own, and no retry reaches any of them.
        (PermissionError(13, 'Permission denied'), wire.Refusal.FINAL),
        (OSError(24, 'Too many open files'), wire.Refusal.FINAL),
    ],
)
def test_a_socket_dial_that_did_not_open_says_whether_a_retry_can_reach_it(raised, refusal, tmp_path):
    """A connection-level failure reached the socket, so it reads as it does on a port. An absent path
    is no answer because a misspelt one and a socket nobody has bound yet look the same from here."""
    address = wire.UnixSocketAddress(tmp_path / 'absent.sock', wire.SESSION_PATH, '')

    with (
        patch('positronic_wire.websocket.unix_connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketUnixClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


def test_a_refusal_from_a_path_that_is_not_a_socket_is_final(tmp_path):
    """A path holding a regular file is refused in the same words as a socket nobody listens on, so
    the refusal reads the path: only one of the two can ever come good."""
    not_a_socket = tmp_path / 'regular.file'
    not_a_socket.write_text('not a socket')
    address = wire.UnixSocketAddress(not_a_socket, wire.SESSION_PATH, '')

    with (
        patch('positronic_wire.websocket.unix_connect', side_effect=ConnectionRefusedError(111, 'Connection refused')),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketUnixClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is wire.Refusal.FINAL


def test_a_socket_address_refuses_a_relative_path():
    """`--policy.address.uds=policy.sock` names a different socket to each caller, so the address refuses it."""
    with pytest.raises(ValueError, match='relative socket path'):
        wire.UnixSocketAddress(Path('policy.sock'), wire.SESSION_PATH, '')


# ─── keepalive and probe against a server on loopback ────────────────────────


def _at(host: str, port: int) -> wire.HostPortAddress:
    return wire.HostPortAddress(host, port, wire.SESSION_PATH, '')


def _unbound() -> tuple[str, int]:
    """A loopback port nothing listens on."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    return host, port


def _served(*handlers: Callable[[socket.socket], None], family: int = socket.AF_INET) -> tuple[str, int]:
    """A loopback server running each of ``handlers`` against one connection, in order."""
    listener = socket.socket(family)
    listener.bind(('::1' if family == socket.AF_INET6 else '127.0.0.1', 0))
    listener.listen(len(handlers))

    def serve_each() -> None:
        try:
            for handler in handlers:
                conn, _ = listener.accept()
                with conn:
                    handler(conn)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client closed its end at its deadline or at its byte cap, mid-answer
        finally:
            listener.close()

    threading.Thread(target=serve_each, daemon=True).start()
    host, port = listener.getsockname()[:2]
    return host, port


def _answering(head: bytes, asked: list[bytes], body: bytes = b''):
    """A handler answering ``head`` and ``body``, recording the request it read."""

    def handler(conn: socket.socket) -> None:
        asked.append(conn.recv(4096))
        conn.sendall(head + b'\r\ncontent-length: %d\r\n\r\n' % len(body) + body)

    return handler


@pytest.fixture
def dropping() -> Iterator[tuple[str, int]]:
    """A loopback port whose listen queue is full, so the kernel drops every further connect, as a firewall does."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(0)
    filler = socket.create_connection(listener.getsockname())
    yield listener.getsockname()
    filler.close()
    listener.close()


def test_a_keepalive_posts_to_its_route_with_the_callers_headers_and_reads_the_seconds():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE))
    assert _WEBSOCKET.keepalive(_at(host, port), dict([_HEADER]), 5.0) == 60
    assert asked[0].startswith(f'POST {wire.KEEPALIVE_PATH} HTTP/1.1\r\n'.encode())
    assert b'Content-Length: 0\r\n' in asked[0]
    assert f'{_HEADER[0]}: {_HEADER[1]}\r\n'.encode() in asked[0]


def test_a_chunked_keepalive_answer_carries_the_seconds():
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        chunked = b'%x\r\n%s\r\n0\r\n\r\n' % (len(_ALIVE), _ALIVE)
        conn.sendall(b'HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n' + chunked)

    assert _WEBSOCKET.keepalive(_at(*_served(handler)), None, 5.0) == 60


def test_a_keepalive_answer_of_a_server_with_no_idle_timeout_carries_none():
    body = json.dumps({wire.ALIVE_SECONDS: None}).encode()
    assert _WEBSOCKET.keepalive(_at(*_served(_answering(b'HTTP/1.1 200 OK', [], body))), None, 5.0) is None


@pytest.mark.parametrize(
    'body',
    [b'not json', b'[1]', b'{"other": 1}', b'{"alive_seconds": "soon"}', b'{"alive_seconds": true}'],
    ids=['not-json', 'a-list', 'no-key', 'a-string', 'a-bool'],
)
def test_a_200_that_does_not_carry_the_keepalive_answer_is_final(body):
    """Something other than a policy server answered, so the call does not read it as the server."""
    host, port = _served(_answering(b'HTTP/1.1 200 OK', [], body))
    with pytest.raises(wire.ConnectRefused) as refused:
        _WEBSOCKET.keepalive(_at(host, port), None, 5.0)
    assert refused.value.refusal is wire.Refusal.FINAL


@pytest.mark.parametrize(
    ('head', 'refusal'),
    [
        (b'HTTP/1.1 401 Unauthorized', wire.Refusal.FORBIDDEN),
        (b'HTTP/1.1 403 Forbidden', wire.Refusal.FORBIDDEN),
        (b'HTTP/1.1 429 Too Many Requests', wire.Refusal.COLD),
        (b'HTTP/1.1 500 Internal Server Error', wire.Refusal.COLD),
        (b'HTTP/1.1 502 Bad Gateway', wire.Refusal.COLD),
        (b'HTTP/1.1 400 Bad Request', wire.Refusal.FINAL),
    ],
)
def test_a_keepalive_status_says_what_the_server_is(head, refusal):
    with pytest.raises(wire.ConnectRefused) as refused:
        _WEBSOCKET.keepalive(_at(*_served(_answering(head, []))), None, 5.0)
    assert refused.value.refusal is refusal


def test_a_keepalive_redirect_is_final_and_reaches_no_other_origin():
    """A client that followed one would copy the caller's headers onto a request to the host the server named."""
    reached: list[bytes] = []
    elsewhere_host, elsewhere_port = _served(_answering(b'HTTP/1.1 200 OK', reached, _ALIVE))

    def redirects(conn: socket.socket) -> None:
        conn.recv(4096)
        location = f'http://{elsewhere_host}:{elsewhere_port}/'
        conn.sendall(f'HTTP/1.1 302 Found\r\nLocation: {location}\r\ncontent-length: 0\r\n\r\n'.encode())

    with pytest.raises(wire.ConnectRefused) as refused:
        _WEBSOCKET.keepalive(_at(*_served(redirects)), dict([_HEADER]), 5.0)
    assert refused.value.refusal is wire.Refusal.FINAL
    assert reached == [], 'nothing reached the other origin'


@pytest.mark.parametrize(
    ('root', 'answered'),
    [(b'HTTP/1.1 403 Forbidden', None), (b'HTTP/1.1 404 Not Found', wire.Refusal.FINAL)],
    ids=['a-session-server', 'something-else'],
)
def test_a_404_on_the_keepalive_call_is_read_off_the_upgrade_on_the_root(root, answered):
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 404 Not Found', asked), _answering(root, asked))
    if answered is None:
        with pytest.raises(wire.KeepaliveUnsupported):
            _WEBSOCKET.keepalive(_at(host, port), None, 5.0)
    else:
        with pytest.raises(wire.ConnectRefused) as refused:
            _WEBSOCKET.keepalive(_at(host, port), None, 5.0)
        assert refused.value.refusal is answered
    assert asked[1].startswith(b'GET / HTTP/1.1\r\n')
    assert b'Upgrade: websocket\r\n' in asked[1]


def test_a_tls_keepalive_sends_nothing_in_the_clear():
    """Nothing here presents a certificate, so the handshake fails before a request goes out."""
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE))
    with pytest.raises(wire.ConnectRefused) as refused:
        websocket.WebsocketTlsClientWire().keepalive(_at(host, port), dict([_HEADER]), 5.0)
    assert refused.value.refusal is wire.Refusal.SILENT
    assert all(_HEADER[1].encode() not in request for request in asked)


def test_the_keepalive_host_header_names_an_ipv6_literal_in_brackets():
    asked: list[bytes] = []
    try:
        host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE), family=socket.AF_INET6)
    except OSError as refused:
        if refused.errno not in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT):
            raise
        pytest.skip('this host has no IPv6 loopback')
    assert _WEBSOCKET.keepalive(_at(host, port), None, 5.0) == 60
    assert f'Host: [::1]:{port}\r\n'.encode() in asked[0]


def test_a_host_name_resolves_inside_the_call():
    host, port = _served(_answering(b'HTTP/1.1 200 OK', [], _ALIVE))
    assert _WEBSOCKET.keepalive(_at('localhost' if host == '127.0.0.1' else host, port), None, 5.0) == 60


# ─── no answer ───────────────────────────────────────────────────────────────


def _refusal_of_keepalive(address: wire.HostPortAddress, timeout: float) -> wire.Refusal:
    with pytest.raises(wire.ConnectRefused) as refused:
        _WEBSOCKET.keepalive(address, None, timeout)
    return refused.value.refusal


def test_a_keepalive_to_a_port_nothing_listens_on_is_no_answer():
    assert _refusal_of_keepalive(_at(*_unbound()), 2.0) is wire.Refusal.SILENT


@pytest.mark.parametrize(
    ('errno_', 'refusal'),
    [(socket.EAI_AGAIN, wire.Refusal.UNRESOLVED), (socket.EAI_NONAME, wire.Refusal.FINAL)],
    ids=['the-resolver-timed-out', 'no-such-host'],
)
def test_a_name_lookup_that_fails_reads_as_a_dial_reads_it(errno_, refusal, monkeypatch):
    """A name can start resolving, where a misspelt one never does."""

    def failed_lookup(*_args, **_kwargs):
        raise socket.gaierror(errno_, 'lookup failed')

    monkeypatch.setattr(socket, 'getaddrinfo', failed_lookup)
    assert _refusal_of_keepalive(_at('a-host.example', 8000), 2.0) is refusal
    assert _WEBSOCKET.probe(_at('a-host.example', 8000), None, 2.0) is refusal


def test_a_name_lookup_that_outlasts_the_timeout_is_no_answer_in_time(monkeypatch):
    def slow_lookup(*_args, **_kwargs):
        time.sleep(3.0)
        raise socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution')

    monkeypatch.setattr(socket, 'getaddrinfo', slow_lookup)
    started = time.monotonic()
    assert _refusal_of_keepalive(_at('slow.example', 8000), 0.3) is wire.Refusal.SILENT
    assert _WEBSOCKET.probe(_at('slow.example', 8000), None, 0.3) is wire.Refusal.SILENT
    assert time.monotonic() - started < 2.0, 'the timeout did not cover the lookup'


def test_one_timeout_covers_every_address_a_name_resolves_to(dropping, monkeypatch):
    """``socket.create_connection`` gives each address the whole timeout; four would take four times as long."""
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', dropping)] * 4
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *_args, **_kwargs: resolved)
    started = time.monotonic()
    assert _refusal_of_keepalive(_at('four-records.example', 8000), 0.4) is wire.Refusal.SILENT
    assert time.monotonic() - started < 1.0, 'the timeout did not cover every address'


def test_a_host_that_drops_every_connect_is_no_answer(dropping):
    assert _refusal_of_keepalive(_at(*dropping), 0.3) is wire.Refusal.SILENT
    assert _WEBSOCKET.probe(_at(*dropping), None, 0.3) is wire.Refusal.SILENT


@pytest.mark.parametrize('status_field', [b'2000', b'9' * 5000], ids=['four-digits', 'past-the-int-limit'])
def test_a_status_line_that_is_not_three_digits_is_no_answer(status_field):
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(b'HTTP/1.1 ' + status_field + b' Whatever\r\n\r\n')

    assert _refusal_of_keepalive(_at(*_served(handler)), 5.0) is wire.Refusal.SILENT


def test_a_server_trickling_its_answer_cannot_outlast_the_timeout():
    """Every byte arrives well inside a per-read timeout, so only the wall clock ends this."""
    ended: list[float] = []

    def trickling(conn: socket.socket) -> None:
        conn.recv(4096)
        for byte in b'HTTP/1.1 200 OK':
            try:
                conn.sendall(bytes([byte]))
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.15)
        ended.append(time.monotonic())

    host, port = _served(trickling)
    started = time.monotonic()
    refusal = _refusal_of_keepalive(_at(host, port), 0.3)
    returned = time.monotonic()
    assert refusal is wire.Refusal.SILENT
    assert returned - started < 1.0, 'the timeout did not end it'
    deadline = time.monotonic() + 2.0
    while not ended and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ended and ended[0] - returned < 1.0, 'the reader kept reading past the timeout'


def test_a_server_that_says_nothing_cannot_outlast_the_timeout():
    def silent(conn: socket.socket) -> None:
        conn.recv(4096)
        time.sleep(5.0)

    started = time.monotonic()
    assert _refusal_of_keepalive(_at(*_served(silent)), 0.3) is wire.Refusal.SILENT
    assert _WEBSOCKET.probe(_at(*_served(silent)), None, 0.3) is wire.Refusal.SILENT
    assert time.monotonic() - started < 2.0, 'the timeout did not end it'


def test_a_server_flooding_its_answer_is_cut_off_by_the_byte_cap():
    """The body never ends, so a read that kept it all would spend the caller's memory."""
    stopped: list[float] = []

    def flood(conn: socket.socket) -> None:
        conn.recv(4096)
        try:
            conn.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
            while True:
                conn.sendall(b'x' * 65536)
        except (BrokenPipeError, ConnectionResetError):
            stopped.append(time.monotonic())

    started = time.monotonic()
    assert _refusal_of_keepalive(_at(*_served(flood)), 30.0) is wire.Refusal.FINAL
    assert time.monotonic() - started < 5.0, 'the byte cap did not end it'
    deadline = time.monotonic() + 5.0
    while not stopped and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stopped, 'the server never saw its reader go'


@pytest.mark.parametrize('timeout', [0.0, -1.0], ids=['spent', 'past'])
def test_a_timeout_already_spent_asks_nothing_and_is_no_answer(timeout):
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE))
    assert _refusal_of_keepalive(_at(host, port), timeout) is wire.Refusal.SILENT
    assert _WEBSOCKET.probe(_at(host, port), None, timeout) is wire.Refusal.SILENT
    assert asked == []


def test_a_probe_asks_the_root_for_an_upgrade_with_the_callers_headers():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 403 Forbidden', asked))
    assert _WEBSOCKET.probe(_at(host, port), dict([_HEADER]), 5.0) is None
    assert asked[0].startswith(b'GET / HTTP/1.1\r\n')
    assert f'{_HEADER[0]}: {_HEADER[1]}\r\n'.encode() in asked[0]


def test_a_probe_ignores_a_proxy_the_environment_names(monkeypatch):
    via_proxy: list[bytes] = []
    proxy_host, proxy_port = _served(_answering(b'HTTP/1.1 502 Bad Gateway', via_proxy))
    host, port = _served(_answering(b'HTTP/1.1 403 Forbidden', []))
    monkeypatch.setenv('ws_proxy', f'http://{proxy_host}:{proxy_port}')
    assert _WEBSOCKET.probe(_at(host, port), None, 5.0) is None
    assert via_proxy == []


def _answer_the_upgrade(conn: socket.socket) -> None:
    """Read the upgrade request and answer it with 101."""
    request = b''
    while b'\r\n\r\n' not in request and (chunk := conn.recv(4096)):
        request += chunk
    key = next(
        line.split(b':', 1)[1].strip().decode()
        for line in request.split(b'\r\n')
        if line.lower().startswith(b'sec-websocket-key:')
    )
    conn.sendall(
        b'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
        b'Sec-WebSocket-Accept: ' + accept_key(key).encode() + b'\r\n\r\n'
    )


def _upgrade_then_ignore_the_close(conn: socket.socket) -> None:
    """Answer the upgrade with 101, then read every frame and answer none, not even a close."""
    _answer_the_upgrade(conn)
    while conn.recv(4096):
        pass


# A wire bound far past what any close in these tests may take, and one well inside it.
_LONG_CLOSE_S = 30.0
_SHORT_CLOSE_S = 0.2
# What a wall-clock test grants past a bound: a thread start and a closed socket, on a loaded machine.
_CLOSE_SLACK_S = 2.0


def _seconds_to_close(wire_bound: float | None, timeout: float | None = None) -> float:
    """Seconds ``close(timeout)`` takes on a wire bound to ``wire_bound``, against a peer that never answers it."""
    host, port = _served(_upgrade_then_ignore_the_close)
    connection = websocket.WebsocketClientConnection(connect(f'ws://{host}:{port}/', close_timeout=wire_bound))
    closer = threading.Thread(target=connection.close, kwargs={'timeout': timeout}, daemon=True)
    started = time.monotonic()
    closer.start()
    closer.join(timeout=_LONG_CLOSE_S / 2)
    assert not closer.is_alive(), 'the close waited past every bound it was given'
    return time.monotonic() - started


def test_a_close_timeout_longer_than_the_wires_own_keeps_the_wires_bound():
    assert _seconds_to_close(_SHORT_CLOSE_S, timeout=_LONG_CLOSE_S) < _SHORT_CLOSE_S + _CLOSE_SLACK_S


def test_a_close_timeout_shorter_than_the_wires_own_bounds_the_close():
    assert _seconds_to_close(_LONG_CLOSE_S, timeout=_SHORT_CLOSE_S) < _SHORT_CLOSE_S + _CLOSE_SLACK_S


def test_a_close_with_no_timeout_keeps_the_wires_bound():
    assert _seconds_to_close(_SHORT_CLOSE_S) < _SHORT_CLOSE_S + _CLOSE_SLACK_S


def test_a_close_timeout_bounds_a_close_the_wire_left_unbounded():
    assert _seconds_to_close(None, timeout=_SHORT_CLOSE_S) < _SHORT_CLOSE_S + _CLOSE_SLACK_S


# More than the send and receive buffers of a loopback connection hold, so the write waits for the peer to read.
_BUFFER_FILLING_BYTES = 64 * 1024 * 1024
_SEND_LIMIT_S = 1.0


def _upgrade_then_read_nothing(released: threading.Event) -> Callable[[socket.socket], None]:
    def handler(conn: socket.socket) -> None:
        _answer_the_upgrade(conn)
        released.wait(_LONG_CLOSE_S)

    return handler


def _upgrade_then_read_slowly(released: threading.Event) -> Callable[[socket.socket], None]:
    """Read 64 KiB each 50 ms: the peer takes the data, but too slowly for the message to fit the timeout."""

    def handler(conn: socket.socket) -> None:
        _answer_the_upgrade(conn)
        while not released.is_set() and conn.recv(64 * 1024):
            time.sleep(0.05)

    return handler


def _over_tls(handler: Callable[[socket.socket], None], context: ssl.SSLContext) -> Callable[[socket.socket], None]:
    def over_tls(conn: socket.socket) -> None:
        with context.wrap_socket(conn, server_side=True) as tls:
            handler(tls)

    return over_tls


def _tls_contexts(directory: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A server context with a certificate for 127.0.0.1, and a client context that trusts only it."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, '127.0.0.1')])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509
        .CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = directory / 'cert.pem', directory / 'key.pem'
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    # The server sends no TLS traffic after the handshake, so a test sees only application data on the connection.
    server.num_tickets = 0
    return server, ssl.create_default_context(cafile=str(cert_file))


def _served_on_unix(handler: Callable[[socket.socket], None], path: Path) -> None:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def serve() -> None:
        with listener, listener.accept()[0] as conn:
            handler(conn)

    threading.Thread(target=serve, daemon=True).start()


@contextmanager
def _tcp_session(
    handler: Callable[[socket.socket], None],
    session: Callable[[Connection], websocket.WebsocketClientConnection] = websocket.WebsocketClientConnection,
) -> Iterator[websocket.WebsocketClientConnection]:
    host, port = _served(handler)
    yield session(connect(f'ws://{host}:{port}/', max_size=None, open_timeout=_LONG_CLOSE_S))


@contextmanager
def _tls_session(handler: Callable[[socket.socket], None]) -> Iterator[websocket.WebsocketClientConnection]:
    with tempfile.TemporaryDirectory() as directory:
        server, client = _tls_contexts(Path(directory))
        host, port = _served(_over_tls(handler, server))
        yield websocket.WebsocketClientConnection(
            connect(f'wss://{host}:{port}/', ssl=client, max_size=None, open_timeout=_LONG_CLOSE_S)
        )


@contextmanager
def _unix_session(handler: Callable[[socket.socket], None]) -> Iterator[websocket.WebsocketClientConnection]:
    # A short directory: a Unix socket path has a length limit.
    with tempfile.TemporaryDirectory(prefix='ws') as directory:
        path = Path(directory) / 's'
        _served_on_unix(handler, path)
        yield websocket.WebsocketClientConnection(
            unix_connect(str(path), 'ws://localhost/', max_size=None, open_timeout=_LONG_CLOSE_S)
        )


# The roboarena connection keeps the websocket send, on its own class.
_roboarena_session = partial(_tcp_session, session=roboarena.RoboarenaClientConnection)


@pytest.mark.parametrize(
    'session', [_tcp_session, _tls_session, _unix_session, _roboarena_session], ids=['tcp', 'tls', 'unix', 'roboarena']
)
def test_a_send_to_a_peer_that_reads_nothing_raises_a_timeout_at_its_timeout(session):
    released = threading.Event()
    try:
        with session(_upgrade_then_read_nothing(released)) as connection:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                connection.send(bytes(_BUFFER_FILLING_BYTES), timeout=_SEND_LIMIT_S)
            assert time.monotonic() - started < _SEND_LIMIT_S + _CLOSE_SLACK_S
    finally:
        released.set()


def test_a_send_to_a_peer_that_reads_slowly_raises_a_timeout_at_its_timeout():
    released = threading.Event()
    try:
        with _tcp_session(_upgrade_then_read_slowly(released)) as connection:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                connection.send(bytes(_BUFFER_FILLING_BYTES), timeout=_SEND_LIMIT_S)
            assert time.monotonic() - started < _SEND_LIMIT_S + _CLOSE_SLACK_S
    finally:
        released.set()


# Longer than the bound of the first send, so a bound left on the socket would end the second send.
_STALL_S = 3.0


def _read_a_chunk_then_stall(conn: socket.socket) -> None:
    """Answer the upgrade, read one chunk, read nothing for ``_STALL_S``, then read until the end."""
    _answer_the_upgrade(conn)
    conn.recv(4096)
    time.sleep(_STALL_S)
    while conn.recv(1 << 20):
        pass


def test_a_send_with_no_timeout_outlasts_the_timeout_of_an_earlier_send():
    host, port = _served(_read_a_chunk_then_stall)
    connection = websocket.WebsocketClientConnection(
        connect(f'ws://{host}:{port}/', max_size=None, close_timeout=_SHORT_CLOSE_S)
    )
    try:
        connection.send(b'observation', timeout=_SHORT_CLOSE_S)
        connection.send(bytes(_BUFFER_FILLING_BYTES))
    finally:
        connection.close()
