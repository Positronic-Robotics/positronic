"""What counts as a policy server that serves, and whether its token gate holds.

The readiness calls run against real servers on loopback, because the property under test is a
wall-clock and a memory bound against a server that misbehaves, and no fake can hold that.
"""

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
from positronic_wire.probe import Answer, Gate
from positronic_wire.roboarena import RoboarenaClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketUnixClientWire
from positronic_wire.wire import (
    ALIVE_SECONDS,
    AUTH_HEADER,
    KEEPALIVE_PATH,
    SESSION_PATH,
    ConnectRefused,
    HostPortAddress,
    KeepaliveUnsupported,
    Refusal,
    bearer,
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
    assert probe.warming(answer) is True


@pytest.mark.parametrize('answer', [Answer.admitted, Answer.no_keepalive, Answer.refused, Answer.final])
def test_any_other_answer_settles_the_question(answer):
    """A refused token and a wrong server are verdicts, and waiting on either one only spends the deadline."""
    assert probe.warming(answer) is False


def test_a_wire_that_dials_no_host_and_port_has_no_address_on_one():
    with pytest.raises(ValueError, match='websocket_unix'):
        probe.address_on(WebsocketUnixClientWire(), '127.0.0.1', 8000)


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
        except OSError:
            pass
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


@forks
def test_a_keepalive_answer_admits_the_caller_and_the_token_travels_as_a_bearer():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, json.dumps({ALIVE_SECONDS: 60}).encode()))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0, {AUTH_HEADER: bearer('tok')}) is Answer.admitted
    assert f'{AUTH_HEADER}: {bearer("tok")}\r\n'.encode() in asked[0]


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
    """A client that followed one would copy the bearer onto the redirected request and hand it to
    whichever host the server named."""
    reached: list[bytes] = []
    elsewhere_host, elsewhere_port = _served(_answering(b'HTTP/1.1 200 OK', reached))

    def redirects(conn: socket.socket) -> None:
        conn.recv(4096)
        location = f'http://{elsewhere_host}:{elsewhere_port}/'
        conn.sendall(f'HTTP/1.1 302 Found\r\nLocation: {location}\r\ncontent-length: 0\r\n\r\n'.encode())

    host, port = _served(redirects)
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0, {AUTH_HEADER: bearer('tok')}) is Answer.final
    assert reached == [], 'nothing reached the other origin'


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


# ─── the GET the gate's session probes send ──────────────────────────────────


def test_a_server_that_answers_is_read_off_its_status_line():
    host, port = _served(_answering(b'HTTP/1.1 401 Unauthorized', []))
    assert probe.status_of(host, port, '/', {}, 5.0) == HTTPStatus.UNAUTHORIZED


def test_a_session_probe_trickling_its_headers_cannot_outlast_the_deadline():
    host, port = _served(_trickling([]))
    started = time.monotonic()
    assert probe.status_of(host, port, '/', {}, 0.3) is None
    assert time.monotonic() - started < 2.0, 'the deadline did not end it'


def test_a_name_answering_with_several_addresses_gets_one_budget_between_them(monkeypatch):
    """`socket.create_connection` arms its timeout per address, so a name with four records would spend
    four budgets in connect alone before anything read the clock."""
    tried: list = []

    class Blackhole(socket.socket):
        """Accepts the connect and answers nothing, which is what a budget has to end."""

        def connect(self, address):
            tried.append(address)
            time.sleep(self.gettimeout() or 0.0)
            raise TimeoutError('timed out')

    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (f'192.0.2.{n}', 443)) for n in (1, 2, 3, 4)]
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **kw: records)
    monkeypatch.setattr(socket, 'socket', Blackhole)

    started = time.monotonic()
    assert probe.status_of('endpoint.example', 443, '/', {}, 0.4) is None
    elapsed = time.monotonic() - started
    assert elapsed < 1.2, f'{len(tried)} addresses spent {elapsed:.1f}s of a 0.4s budget'


def test_a_refused_connection_is_no_status():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    assert probe.status_of(host, port, '/', {}, 1.0) is None


def test_something_that_is_not_http_on_the_port_is_no_status():
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(b'not http at all\r\n')

    host, port = _served(handler)
    assert probe.status_of(host, port, '/', {}, 5.0) is None


@pytest.mark.parametrize(
    'status_field',
    [b'2000', b'20', b'', b'9' * 5000, b'20x'],
    ids=['four-digits', 'two-digits', 'empty', 'past-the-int-limit', 'not-digits'],
)
def test_a_status_field_that_is_not_three_digits_is_no_status(status_field):
    """At 5000 digits `isdigit` passes and `int` refuses, so a server could raise out of a probe and into
    the caller's loop."""

    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(b'HTTP/1.1 ' + status_field + b' Whatever\r\n\r\n')

    host, port = _served(handler)
    assert probe.status_of(host, port, '/', {}, 5.0) is None


def test_the_host_header_names_an_ipv6_literal_in_brackets():
    asked: list[bytes] = []
    listener = socket.socket(socket.AF_INET6)
    try:
        listener.bind(('::1', 0))
    except OSError:
        pytest.skip('this host has no IPv6 loopback')
    listener.listen(1)
    port = listener.getsockname()[1]

    def answer() -> None:
        conn, _ = listener.accept()
        with conn:
            _answering(b'HTTP/1.1 403 Forbidden', asked)(conn)
        listener.close()

    threading.Thread(target=answer, daemon=True).start()
    assert probe.status_of('::1', port, '/', {}, 5.0) == HTTPStatus.FORBIDDEN
    assert f'Host: [::1]:{port}\r\n'.encode() in asked[0]


# ─── the gate, on a server shaped like a positronic one ──────────────────────

_TOKEN = 'tok'
_KEEPALIVE = 'keepalive'


@dataclass
class _Positronic:
    """A loopback server answering the readiness call and the session upgrade the way a gated positronic
    server does, with the deviations a test names.

    `opens` names routes that serve any caller, `refuses` names routes that refuse the server's own token.
    A route is `_KEEPALIVE` or a session path. A server without the keepalive call answers it 404.
    """

    keepalive: bool = True
    opens: frozenset[str] = frozenset()
    refuses: frozenset[str] = frozenset()
    authorizations: list[str] = field(default_factory=list)

    def answer(self, method: str, path: str, authorization: str) -> bytes:
        self.authorizations.append(authorization)
        route = _KEEPALIVE if (method, path) == ('POST', KEEPALIVE_PATH) else path
        own = authorization == bearer(_TOKEN) and route not in self.refuses
        served = own or route in self.opens
        if route == _KEEPALIVE:
            if not self.keepalive:
                return b'HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n'
            if served:
                body = json.dumps({ALIVE_SECONDS: 60}).encode()
                return b'HTTP/1.1 200 OK\r\ncontent-length: %d\r\n\r\n' % len(body) + body
            return b'HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n'
        if served and route in probe.SESSION_PATHS_OF_WIRE[WebsocketClientWire.NAME]:
            return b'HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\nconnection: Upgrade\r\n\r\n'
        # A positronic server refuses every other upgrade with 403: the root, and a caller with no valid token.
        return b'HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n'

    def handle(self, conn: socket.socket) -> None:
        request = b''
        while b'\r\n\r\n' not in request:
            chunk = conn.recv(4096)
            if not chunk:
                return
            request += chunk
        line, *header_lines = request.split(b'\r\n\r\n', 1)[0].decode().split('\r\n')
        method, path, _version = line.split(' ', 2)
        headers = {name.lower(): value.strip() for name, _, value in (h.partition(':') for h in header_lines)}
        conn.sendall(self.answer(method, path, headers.get(AUTH_HEADER.lower(), '')))


@pytest.fixture
def positronic_like() -> Iterator[Callable[..., tuple[_Positronic, str, int]]]:
    """Serves a `_Positronic` on loopback, one thread per connection, until the test ends."""
    listeners: list[socket.socket] = []

    def start(**deviations) -> tuple[_Positronic, str, int]:
        server = _Positronic(**deviations)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        listeners.append(listener)

        def accept_each() -> None:
            while True:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    return
                threading.Thread(target=_close_after, args=(server.handle, conn), daemon=True).start()

        threading.Thread(target=accept_each, daemon=True).start()
        host, port = listener.getsockname()
        return server, host, port

    yield start
    for listener in listeners:
        listener.close()


def _close_after(handle: Callable[[socket.socket], None], conn: socket.socket) -> None:
    with conn:
        handle(conn)


def _verdict(started: tuple[_Positronic, str, int], *, prove_own_token: bool = True) -> Gate:
    _server, host, port = started
    return probe.gate(_WEBSOCKET, host, port, _TOKEN, 5.0, prove_own_token=prove_own_token)


# Each way a route serves a caller with no valid token.
_SERVES_A_STRANGER = [
    pytest.param(frozenset({_KEEPALIVE}), id='keepalive'),
    pytest.param(frozenset({SESSION_PATH}), id='session'),
    pytest.param(frozenset({probe.MODEL_SESSION_PATH}), id='model-session'),
]


@forks
@pytest.mark.parametrize('keepalive', [True, False], ids=['keepalive', 'no-keepalive'])
def test_a_gate_that_refuses_strangers_and_serves_the_server_holds(positronic_like, keepalive):
    assert _verdict(positronic_like(keepalive=keepalive)) is Gate.holds


@forks
@pytest.mark.parametrize('opens', _SERVES_A_STRANGER)
def test_a_route_that_serves_a_caller_with_no_valid_token_is_an_open_gate(positronic_like, opens):
    assert _verdict(positronic_like(opens=opens)) is Gate.open


@forks
def test_a_server_without_the_keepalive_call_is_caught_open_on_its_session_route(positronic_like):
    assert _verdict(positronic_like(keepalive=False, opens=frozenset({SESSION_PATH}))) is Gate.open


@forks
@pytest.mark.parametrize('refuses', [frozenset({_KEEPALIVE}), frozenset({SESSION_PATH})], ids=['keepalive', 'session'])
def test_a_route_that_does_not_serve_the_servers_own_token_is_a_rejected_token(positronic_like, refuses):
    assert _verdict(positronic_like(refuses=refuses)) is Gate.token_rejected


@forks
def test_a_gate_already_proven_presents_the_servers_own_token_on_no_route(positronic_like):
    """The own-token session probe upgrades, so it takes a session: a server serving one at a time refuses
    its client while a probe holds it."""
    started = positronic_like()
    assert _verdict(started, prove_own_token=False) is Gate.holds
    assert bearer(_TOKEN) not in started[0].authorizations


@forks
@pytest.mark.parametrize('opens', _SERVES_A_STRANGER)
def test_a_gate_already_proven_still_catches_a_route_that_serves_a_stranger(positronic_like, opens):
    assert _verdict(positronic_like(opens=opens), prove_own_token=False) is Gate.open


def test_the_gate_names_the_wires_it_proves_when_asked_of_another():
    with pytest.raises(ValueError, match='websocket, roboarena'):
        probe.gate(GrpcClientWire(), '127.0.0.1', 8000, _TOKEN, 1.0)


# ─── the gate, on a roboarena server ─────────────────────────────────────────


def _announce(connection: ServerConnection) -> None:
    """What a roboarena server sends first on every connection: its configuration, as a binary frame."""
    try:
        connection.send(b'config')
        connection.recv()
    except ConnectionClosed:
        pass


def _roboarena_gate(admits: Callable[[str], bool]):
    """A `process_request` that refuses a handshake whose bearer `admits` refuses, with 401."""

    def process_request(_connection: ServerConnection, request: Request) -> Response | None:
        if admits(request.headers.get(AUTH_HEADER, '')):
            return None
        return Response(HTTPStatus.UNAUTHORIZED, 'Unauthorized', Headers(), b'')

    return process_request


@pytest.fixture
def roboarena_on() -> Iterator[Callable[[Callable[[str], bool]], tuple[str, int]]]:
    """Serves a roboarena server behind a gate that admits a bearer `admits` takes, until the test ends."""
    servers = []

    def start(admits: Callable[[str], bool]) -> tuple[str, int]:
        server = serve(_announce, '127.0.0.1', 0, process_request=_roboarena_gate(admits))
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server.socket.getsockname()

    yield start
    for server in servers:
        server.shutdown()


@forks
def test_a_roboarena_gate_that_refuses_strangers_and_serves_the_server_holds(roboarena_on):
    host, port = roboarena_on(lambda held: held == bearer(_TOKEN))
    assert probe.serving(_ROBOARENA, host, port, 5.0)
    assert probe.gate(_ROBOARENA, host, port, _TOKEN, 5.0) is Gate.holds


@forks
def test_a_roboarena_server_with_no_gate_is_an_open_gate(roboarena_on):
    host, port = roboarena_on(lambda held: True)
    assert probe.gate(_ROBOARENA, host, port, _TOKEN, 5.0) is Gate.open


@forks
def test_a_roboarena_server_that_refuses_its_own_token_is_a_rejected_token(roboarena_on):
    host, port = roboarena_on(lambda held: False)
    assert probe.gate(_ROBOARENA, host, port, _TOKEN, 5.0) is Gate.token_rejected
