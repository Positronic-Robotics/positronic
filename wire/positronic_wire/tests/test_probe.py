"""What counts as a policy server that serves."""

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus

import pytest
from positronic_wire import probe
from positronic_wire.grpc import GrpcClientWire
from positronic_wire.probe import Answer
from positronic_wire.roboarena import RoboarenaClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketTlsClientWire, WebsocketUnixClientWire
from positronic_wire.wire import ALIVE_SECONDS, KEEPALIVE_PATH
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

_WEBSOCKET = WebsocketClientWire()
_ROBOARENA = RoboarenaClientWire()
# A header a caller sends with its readiness call.
_HEADER = ('X-Caller', 'SECRET-VALUE')
_ALIVE = json.dumps({ALIVE_SECONDS: 60}).encode()


def _served(*handlers: Callable[[socket.socket], None], family: int = socket.AF_INET) -> tuple[str, int]:
    """A loopback server running each of `handlers` against one connection, in order."""
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
            pass  # the probe closed its end at its deadline or at its byte cap, mid-answer
        finally:
            listener.close()

    threading.Thread(target=serve_each, daemon=True).start()
    host, port = listener.getsockname()[:2]
    return host, port


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
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.15)
        ended.append(time.monotonic())

    return handler


# ─── the keepalive call ──────────────────────────────────────────────────────


def test_the_readiness_call_is_a_post_to_the_keepalive_route_that_states_no_body():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 401 Unauthorized', asked))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.refused
    assert asked[0].startswith(f'POST {KEEPALIVE_PATH} HTTP/1.1\r\n'.encode())
    assert b'Content-Length: 0\r\n' in asked[0]


def test_a_keepalive_answer_admits_the_caller_and_carries_its_headers():
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0, dict([_HEADER])) is Answer.admitted
    assert f'{_HEADER[0]}: {_HEADER[1]}\r\n'.encode() in asked[0]


def test_a_chunked_keepalive_answer_admits_the_caller():
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        chunked = b'%x\r\n%s\r\n0\r\n\r\n' % (len(_ALIVE), _ALIVE)
        conn.sendall(b'HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n' + chunked)

    host, port = _served(handler)
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.admitted


@pytest.mark.parametrize(
    ('head', 'answer'),
    [
        (b'HTTP/1.1 401 Unauthorized', Answer.refused),
        (b'HTTP/1.1 403 Forbidden', Answer.refused),
        (b'HTTP/1.1 429 Too Many Requests', Answer.cold),
        (b'HTTP/1.1 500 Internal Server Error', Answer.cold),
        (b'HTTP/1.1 502 Bad Gateway', Answer.cold),
        (b'HTTP/1.1 400 Bad Request', Answer.final),
    ],
)
def test_a_keepalive_status_says_what_the_server_is(head, answer):
    host, port = _served(_answering(head, []))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is answer


@pytest.mark.parametrize('body', [b'not json', b'{"other": 1}', b'{"alive_seconds": "soon"}'])
def test_a_keepalive_answer_that_is_not_the_calls_object_is_final(body):
    host, port = _served(_answering(b'HTTP/1.1 200 OK', [], body))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.final


@pytest.mark.parametrize(
    ('root', 'answer'),
    [(b'HTTP/1.1 403 Forbidden', Answer.no_keepalive), (b'HTTP/1.1 404 Not Found', Answer.final)],
    ids=['a-session-server', 'something-else'],
)
def test_a_404_on_the_keepalive_call_is_read_off_the_upgrade_on_the_root(root, answer):
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 404 Not Found', asked), _answering(root, asked))
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is answer
    assert asked[1].startswith(b'GET / HTTP/1.1\r\n')
    assert b'Upgrade: websocket\r\n' in asked[1]


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


def test_a_tls_readiness_call_sends_nothing_in_the_clear():
    """Nothing here presents a certificate, so the handshake fails before a request goes out."""
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked))
    assert probe.readiness_of(WebsocketTlsClientWire(), host, port, 5.0, dict([_HEADER])) is Answer.silent
    assert all(_HEADER[1].encode() not in request for request in asked)


def test_the_host_header_names_an_ipv6_literal_in_brackets():
    asked: list[bytes] = []
    try:
        host, port = _served(_answering(b'HTTP/1.1 401 Unauthorized', asked), family=socket.AF_INET6)
    except OSError:
        pytest.skip('this host has no IPv6 loopback')
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.refused
    assert f'Host: [::1]:{port}\r\n'.encode() in asked[0]


# ─── no answer ───────────────────────────────────────────────────────────────


def test_a_port_nothing_listens_on_is_no_answer():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    host, port = listener.getsockname()
    listener.close()
    assert probe.readiness_of(_WEBSOCKET, host, port, 2.0) is Answer.silent


@pytest.mark.parametrize('wire', [_WEBSOCKET, _ROBOARENA], ids=['websocket', 'roboarena'])
def test_a_host_that_does_not_resolve_is_no_answer(wire, monkeypatch):
    def no_such_host(*_args, **_kwargs):
        raise socket.gaierror(socket.EAI_NONAME, 'Name or service not known')

    monkeypatch.setattr(socket, 'getaddrinfo', no_such_host)
    assert probe.readiness_of(wire, 'no-such-host.invalid', 8000, 2.0) is Answer.silent
    assert not probe.serving(wire, 'no-such-host.invalid', 8000, 2.0)


@pytest.mark.parametrize('wire', [_WEBSOCKET, _ROBOARENA], ids=['websocket', 'roboarena'])
def test_a_host_that_resolves_and_refuses_is_an_answer(wire):
    """The same call, on a host that resolves: the refusal comes from a server."""
    host, port = _served(_answering(b'HTTP/1.1 401 Unauthorized', []))
    assert probe.readiness_of(wire, 'localhost' if host == '127.0.0.1' else host, port, 5.0) is Answer.refused


def test_a_name_lookup_that_outlasts_the_deadline_is_no_answer(monkeypatch):
    def slow_lookup(*_args, **_kwargs):
        time.sleep(3.0)
        raise socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution')

    monkeypatch.setattr(socket, 'getaddrinfo', slow_lookup)
    started = time.monotonic()
    assert probe.readiness_of(_WEBSOCKET, 'slow.example', 8000, 0.3) is Answer.silent
    assert time.monotonic() - started < 1.5, 'the deadline did not cover the lookup'


@pytest.mark.parametrize('status_field', [b'2000', b'9' * 5000], ids=['four-digits', 'past-the-int-limit'])
def test_a_status_line_that_is_not_three_digits_is_no_answer(status_field):
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(b'HTTP/1.1 ' + status_field + b' Whatever\r\n\r\n')

    host, port = _served(handler)
    assert probe.readiness_of(_WEBSOCKET, host, port, 5.0) is Answer.silent


def test_a_server_trickling_its_answer_cannot_outlast_the_deadline():
    """Every byte arrives well inside a per-read timeout, so only the wall clock ends this."""
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


def test_a_server_that_says_nothing_cannot_outlast_the_deadline():
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        time.sleep(5.0)

    host, port = _served(handler)
    started = time.monotonic()
    assert probe.readiness_of(_WEBSOCKET, host, port, 0.3) is Answer.silent
    assert time.monotonic() - started < 2.0, 'the deadline did not end it'


def test_a_server_flooding_its_answer_is_cut_off_by_the_byte_cap():
    """The body never ends, so a read that accumulated it would spend the caller's memory."""
    stopped: list[float] = []

    def flood(conn: socket.socket) -> None:
        conn.recv(4096)
        try:
            conn.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
            while True:
                conn.sendall(b'x' * 65536)
        except OSError:
            stopped.append(time.monotonic())

    host, port = _served(flood)
    started = time.monotonic()
    assert probe.readiness_of(_WEBSOCKET, host, port, 30.0) is Answer.final
    assert time.monotonic() - started < 5.0, 'the byte cap did not end it'
    deadline = time.monotonic() + 5.0
    while not stopped and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stopped, 'the server never saw its reader go'


@pytest.mark.parametrize('deadline_s', [0.0, -1.0], ids=['spent', 'past'])
def test_a_deadline_already_spent_asks_nothing_and_is_no_answer(deadline_s):
    asked: list[bytes] = []
    host, port = _served(_answering(b'HTTP/1.1 200 OK', asked, _ALIVE))
    assert probe.readiness_of(_WEBSOCKET, host, port, deadline_s) is Answer.silent
    assert asked == []


# ─── the wires it speaks ─────────────────────────────────────────────────────


@pytest.mark.parametrize('wire', [GrpcClientWire(), WebsocketUnixClientWire()], ids=['grpc', 'websocket_unix'])
def test_a_wire_it_does_not_speak_is_refused_by_name(wire):
    with pytest.raises(ValueError, match=f'not {wire.NAME}'):
        probe.readiness_of(wire, '127.0.0.1', 8000, 1.0)


@pytest.mark.parametrize('answer', [Answer.cold, Answer.silent])
def test_an_answer_to_wait_on_leaves_the_server_coming_up(answer):
    assert probe.not_up_yet(answer) is True


@pytest.mark.parametrize('answer', [Answer.admitted, Answer.no_keepalive, Answer.refused, Answer.final])
def test_any_other_answer_settles_the_question(answer):
    """A refusal and a wrong server are verdicts, and waiting on either one only spends the deadline."""
    assert probe.not_up_yet(answer) is False


# ─── a roboarena server ──────────────────────────────────────────────────────


def _announcing(first: str | bytes):
    """A roboarena handler that sends `first` as its first frame, then waits for the client to go."""

    def handler(connection: ServerConnection) -> None:
        try:
            connection.send(first)
            connection.recv()
        except ConnectionClosed:
            pass

    return handler


def _refuses_every_handshake(_connection: ServerConnection, _request: Request) -> Response:
    return Response(HTTPStatus.UNAUTHORIZED, 'Unauthorized', Headers(), b'')


@pytest.fixture
def roboarena_on() -> Iterator[Callable[..., tuple[str, int]]]:
    """Serves a roboarena server on loopback until the test ends."""
    servers = []

    def start(handler, process_request=None) -> tuple[str, int]:
        server = serve(handler, '127.0.0.1', 0, process_request=process_request)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server.socket.getsockname()

    yield start
    for server in servers:
        server.shutdown()


def test_a_roboarena_server_that_announces_itself_serves_no_keepalive_call(roboarena_on):
    host, port = roboarena_on(_announcing(b'config'))
    assert probe.readiness_of(_ROBOARENA, host, port, 5.0) is Answer.no_keepalive
    assert probe.serving(_ROBOARENA, host, port, 5.0)


def test_a_roboarena_server_that_announces_a_failure_in_text_is_final(roboarena_on):
    host, port = roboarena_on(_announcing('the policy failed to load'))
    assert probe.readiness_of(_ROBOARENA, host, port, 5.0) is Answer.final


def test_a_roboarena_server_that_refuses_the_handshake_is_up(roboarena_on):
    host, port = roboarena_on(_announcing(b'config'), _refuses_every_handshake)
    assert probe.readiness_of(_ROBOARENA, host, port, 5.0) is Answer.refused
    assert probe.serving(_ROBOARENA, host, port, 5.0)
