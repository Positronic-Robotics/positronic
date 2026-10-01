"""What counts as a policy server that serves."""

import json
import logging
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http import HTTPStatus

import pytest
from positronic_wire import probe
from positronic_wire.grpc import GrpcClientWire
from positronic_wire.probe import Answer
from positronic_wire.roboarena import RoboarenaClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketTlsClientWire, WebsocketUnixClientWire
from positronic_wire.wire import (
    ALIVE_SECONDS,
    KEEPALIVE_PATH,
    SESSION_PATH,
    ConnectRefused,
    HostPortAddress,
    KeepaliveUnsupported,
    Refusal,
)
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

forks = pytest.mark.skipif(sys.platform != 'linux', reason='a readiness call forks, which the probe runs on Linux only')

_WEBSOCKET = WebsocketClientWire()
_ROBOARENA = RoboarenaClientWire()

# ─── what a wire's outcome reads as ──────────────────────────────────────────


@dataclass
class _Wire:
    """A client wire answering one readiness call as a test says, recording the probe's timeout."""

    NAME = WebsocketClientWire.NAME
    keepalive_raises: Exception | None = None
    probed: Refusal | None = None
    keepalive_takes_s: float = 0.0
    probe_timeouts: list[float] = field(default_factory=list)

    def keepalive(self, _address, _headers, _timeout) -> int | None:
        time.sleep(self.keepalive_takes_s)
        if self.keepalive_raises is not None:
            raise self.keepalive_raises
        return 60

    def probe(self, _address, _headers, timeout) -> Refusal | None:
        self.probe_timeouts.append(timeout)
        return self.probed


def _answer(wire: _Wire, timeout: float = 5.0) -> Answer:
    address = HostPortAddress('10.0.0.5', 8000, '', '')
    return probe.answer_of(wire, address, None, timeout)  # pyright: ignore[reportArgumentType]


def _raised_from(refusal: Refusal, cause: Exception) -> ConnectRefused:
    try:
        raise ConnectRefused(refusal, 'refused') from cause
    except ConnectRefused as refused:
        return refused


def test_a_keepalive_call_that_answers_admits_the_caller():
    assert _answer(_Wire()) is Answer.admitted


@pytest.mark.parametrize(
    ('probed', 'answer'),
    [
        (None, Answer.no_keepalive),
        (Refusal.FORBIDDEN, Answer.refused),
        (Refusal.FINAL, Answer.final),
        (Refusal.COLD, Answer.silent),
    ],
    ids=['a-server', 'refused', 'final', 'no-server-yet'],
)
def test_a_server_without_the_keepalive_call_is_read_off_the_wires_probe(probed, answer):
    assert _answer(_Wire(keepalive_raises=KeepaliveUnsupported('none'), probed=probed)) is answer


def test_the_probe_is_given_what_is_left_of_the_timeout():
    wire = _Wire(keepalive_raises=KeepaliveUnsupported('none'), keepalive_takes_s=0.2)
    _answer(wire, timeout=1.0)
    assert wire.probe_timeouts[0] < 0.85


@pytest.mark.parametrize(
    ('refusal', 'answer'),
    [(Refusal.COLD, Answer.cold), (Refusal.FORBIDDEN, Answer.refused), (Refusal.FINAL, Answer.final)],
)
def test_a_refusal_read_off_a_status_is_an_answer(refusal, answer):
    assert _answer(_Wire(keepalive_raises=ConnectRefused(refusal, 'answers a status'))) is answer


@pytest.mark.parametrize('refusal', list(Refusal))
def test_a_refusal_a_library_error_caused_is_no_answer(refusal):
    """A refused connect, a timeout, a certificate the edge does not hold, a name that does not resolve."""
    assert _answer(_Wire(keepalive_raises=_raised_from(refusal, OSError('refused')))) is Answer.silent


@pytest.mark.parametrize('answer', [Answer.cold, Answer.silent])
def test_an_answer_to_wait_on_leaves_the_server_coming_up(answer):
    assert probe.not_up_yet(answer) is True


@pytest.mark.parametrize('answer', [Answer.admitted, Answer.no_keepalive, Answer.refused, Answer.final])
def test_any_other_answer_settles_the_question(answer):
    """A refused token and a wrong server are verdicts, and waiting on either one only spends the deadline."""
    assert probe.not_up_yet(answer) is False


def test_a_wire_that_dials_no_host_and_port_has_no_address_on_one():
    with pytest.raises(ValueError, match='websocket_unix'):
        probe.address_on(WebsocketUnixClientWire(), '127.0.0.1', 8000)


def test_a_readiness_call_refuses_a_wire_whose_refusals_chain_the_status_the_server_sent():
    """A gRPC server's `PERMISSION_DENIED` arrives as the cause, which reads as no answer."""
    with pytest.raises(ValueError, match='not grpc'):
        probe.readiness_of(GrpcClientWire(), '127.0.0.1', 8000, 1.0)
    with pytest.raises(ValueError, match='not grpc'):
        probe.answer_of(GrpcClientWire(), HostPortAddress('127.0.0.1', 8000, SESSION_PATH, ''), None, 1.0)


def test_a_readiness_call_refuses_a_system_other_than_linux(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(NotImplementedError, match='darwin'):
        probe.readiness_of(_WEBSOCKET, '127.0.0.1', 8000, 1.0)


# ─── loopback servers ────────────────────────────────────────────────────────


def _served(*handlers: Callable[[socket.socket], None]) -> tuple[str, int]:
    """A loopback server running each of `handlers` against one connection, in order."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(len(handlers))

    def serve_each() -> None:
        try:
            for handler in handlers:
                conn, _ = listener.accept()
                with conn:
                    handler(conn)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the probe's child was killed at its deadline, mid-answer
        finally:
            listener.close()

    threading.Thread(target=serve_each, daemon=True).start()
    return listener.getsockname()


def _answering(head: bytes, asked: list[bytes], body: bytes = b''):
    """A handler answering `head` and `body`, recording the request it read."""

    def handler(conn: socket.socket) -> None:
        asked.append(conn.recv(4096))
        conn.sendall(head + b'\r\ncontent-length: %d\r\n\r\n' % len(body) + body)

    return handler


def _trickling(ended: list[float]):
    """A handler sending a status line one byte at a time, recording when its reader went away."""

    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        for byte in b'HTTP/1.1 200 OK':
            try:
                conn.sendall(bytes([byte]))
            except OSError:
                break
            time.sleep(0.15)
        ended.append(time.monotonic())

    return handler


# ─── the readiness call, through the child ───────────────────────────────────


@forks
def test_the_readiness_call_is_a_post_to_the_keepalive_route_that_states_no_body():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 401 Unauthorized', asked))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.refused
    assert asked[0].startswith(f'POST {KEEPALIVE_PATH} HTTP/1.1\r\n'.encode())
    assert b'Content-Length: 0\r\n' in asked[0]


# A header a caller sends with its readiness call.
_HEADER = ('X-Caller', 'SECRET-VALUE')


@forks
def test_a_keepalive_answer_admits_the_caller_and_carries_its_headers():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, json.dumps({ALIVE_SECONDS: 60}).encode()))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0, dict([_HEADER])) is Answer.admitted
    assert f'{_HEADER[0]}: {_HEADER[1]}\r\n'.encode() in asked[0]


@forks
def test_a_gateway_status_asks_the_caller_to_wait():
    host, port = _served(_answering(b'HTTP/1.1 502 Bad Gateway', []))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.cold


@forks
def test_a_keepalive_answer_the_wire_cannot_read_is_final():
    host, port = _served(_answering(b'HTTP/1.1 200 OK', [], b'not json'))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.final


@forks
def test_a_redirect_is_an_answer_and_reaches_no_other_origin():
    """A client that followed one would copy the caller's headers onto the redirected request and hand
    them to whichever host the server named."""
    reached: list[bytes] = []
    elsewhere_host, elsewhere_port = _served(_answering(b'HTTP/1.1 200 OK', reached))

    def redirects(conn: socket.socket) -> None:
        conn.recv(4096)
        location = f'http://{elsewhere_host}:{elsewhere_port}/'
        conn.sendall(f'HTTP/1.1 302 Found\r\nLocation: {location}\r\ncontent-length: 0\r\n\r\n'.encode())

    host, port = _served(redirects)
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0, dict([_HEADER])) is Answer.final
    assert reached == [], 'nothing reached the other origin'


@forks
def test_a_tls_readiness_call_sends_nothing_in_the_clear():
    """Nothing here presents a certificate, so the handshake fails before a request goes out."""
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked))
    assert probe.readiness_of(WebsocketTlsClientWire(), host, port, 5.0, dict([_HEADER])) is Answer.silent
    assert all(_HEADER[1].encode() not in request for request in asked)


@forks
def test_a_port_nothing_listens_on_is_no_answer():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    assert probe.readiness_of(_WEBSOCKET, host, port, 2.0) is Answer.silent


@forks
def test_a_server_trickling_its_answer_cannot_outlast_the_deadline():
    """Every byte arrives well inside the wire's per-read timeout, so only the child's kill ends this,
    and the killed child reads nothing more."""
    ended: list[float] = []
    host, port = _served(_trickling(ended))
    started = time.monotonic()
    answer = probe.readiness_of(_WEBSOCKET, host, port, 0.3)
    returned = time.monotonic()
    assert answer is Answer.silent
    assert returned - started < 1.0, 'the deadline did not end it'
    deadline = time.monotonic() + 2.0
    while not ended and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ended and ended[0] - returned < 1.0, 'the reader kept reading past the deadline'


@forks
def test_a_server_flooding_its_answer_is_cut_off_by_the_childs_memory_limit():
    """The body never ends, so a read that accumulated it would spend the caller's memory. The child
    stops at its limit well inside the deadline, and the server sees its reader go."""
    sent: list[int] = []

    def flood(conn: socket.socket) -> None:
        conn.recv(4096)
        total = 0
        try:
            conn.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
            while True:
                conn.sendall(b'x' * 65536)
                total += 65536
        except OSError:
            sent.append(total)

    host, port = _served(flood)
    started = time.monotonic()
    answer = probe.readiness_of(_WEBSOCKET, host, port, 30.0)
    assert time.monotonic() - started < 10.0, 'the memory limit did not end it'
    assert answer is Answer.final
    deadline = time.monotonic() + 5.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sent and sent[0] < 16 * probe.CHILD_HEADROOM_BYTES


@forks
def test_a_server_that_says_nothing_cannot_outlast_the_deadline():
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        time.sleep(5.0)

    host, port = _served(handler)
    started = time.monotonic()
    assert probe.readiness_of(_WEBSOCKET, host, port, 0.3) is Answer.silent
    assert time.monotonic() - started < 2.0, 'the deadline did not end it'


@forks
def test_a_call_that_raises_what_no_answer_explains_is_logged_and_no_answer(caplog):
    def raises() -> Answer:
        raise RuntimeError('a fault of ours')

    with caplog.at_level(logging.ERROR, logger=probe.__name__):
        assert probe._ask_in_child(raises, 5.0, 'the box') is Answer.silent  # noqa: SLF001
    assert 'a fault of ours' in caplog.text


@forks
@pytest.mark.parametrize(
    ('head', 'up'), [(b'HTTP/1.1 401 Unauthorized', True), (b'HTTP/1.1 503 Service Unavailable', False)]
)
def test_a_server_is_up_once_its_answer_is_not_one_to_wait_on(head, up):
    host, port = _served(_answering(head, []))
    assert probe.serving(_WEBSOCKET, host, port, 5.0) is up


# ─── a roboarena server ──────────────────────────────────────────────────────


def _announce(connection: ServerConnection) -> None:
    """What a roboarena server sends first on every connection: its configuration, as a binary frame."""
    try:
        connection.send(b'config')
        connection.recv()
    except ConnectionClosed:
        pass


def _refuses_every_handshake(_connection: ServerConnection, _request: Request) -> Response:
    return Response(HTTPStatus.UNAUTHORIZED, 'Unauthorized', Headers(), b'')


@pytest.fixture
def roboarena_on() -> Iterator[Callable[..., tuple[str, int]]]:
    """Serves a roboarena server on loopback until the test ends; `process_request` may refuse a handshake."""
    servers = []

    def start(process_request=None) -> tuple[str, int]:
        server = serve(_announce, '127.0.0.1', 0, process_request=process_request)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server.socket.getsockname()

    yield start
    for server in servers:
        server.shutdown()


@forks
def test_a_roboarena_server_that_announces_itself_serves_no_keepalive_call(roboarena_on):
    host, port = roboarena_on()
    assert probe.readiness_of(_ROBOARENA, host, port, 5.0) is Answer.no_keepalive
    assert probe.serving(_ROBOARENA, host, port, 5.0)


@forks
def test_a_roboarena_server_that_refuses_the_handshake_is_up(roboarena_on):
    """It refuses with 401, which no retry changes: a server answered."""
    host, port = roboarena_on(_refuses_every_handshake)
    assert probe.readiness_of(_ROBOARENA, host, port, 5.0) is Answer.final
    assert probe.serving(_ROBOARENA, host, port, 5.0)
