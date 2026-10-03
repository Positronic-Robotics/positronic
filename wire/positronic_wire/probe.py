"""Whether a policy server serves.

A readiness call is one exchange on a raw socket, under one wall-clock deadline that covers the name
lookup, and with a cap on the bytes it reads. A server the caller did not write then cannot hold the
caller past the deadline or spend its memory.
"""

import functools
import http.client
import io
import json
import logging
import queue
import socket
import ssl
import threading
import time
from collections.abc import Callable, Mapping
from enum import Enum
from http import HTTPStatus
from typing import Any

from positronic_wire.roboarena import RoboarenaClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketTlsClientWire
from positronic_wire.wire import ALIVE_SECONDS, KEEPALIVE_PATH, ClientWire, HostPortAddress, bracket_ipv6, netloc

_logger = logging.getLogger(__name__)


class Answer(Enum):
    """What one readiness call on a server's wire came back with."""

    admitted = 'admitted'  # the keepalive call took these headers
    no_keepalive = 'no keepalive'  # a server answers, and serves no keepalive call
    refused = 'refused'  # an answer refused the credential
    cold = 'cold'  # an answer asking to retry: a 5xx, a 429
    final = 'final'  # an answer no retry changes: any other status, or a body that cannot be read
    silent = 'silent'  # no answer: no connection, no readable status line, or the deadline passed


# The answers that only a policy server gives to its readiness call.
POLICY_ANSWERS = frozenset({Answer.admitted, Answer.no_keepalive, Answer.refused})

# What one readiness call reads at most. A status line, its headers and a JSON object of one key fit in
# it many times over, and a server that floods instead reaches it first.
_MAX_ANSWER_BYTES = 16 * 1024

# A WebSocket handshake. The key is fixed: only the status line and the first frame byte are read.
_UPGRADE = {
    'Upgrade': 'websocket',
    'Connection': 'Upgrade',
    'Sec-WebSocket-Key': 'AAAAAAAAAAAAAAAAAAAAAA==',
    'Sec-WebSocket-Version': '13',
}

# The opcodes of a first frame that a roboarena server can send: its configuration, or a failure in text.
_BINARY_FRAME = 0x2
_TEXT_FRAME = 0x1


def _resolved(host: str, port: int, budget: Callable[[], float]) -> list[tuple[Any, ...]]:
    """The addresses `host` resolves to, inside the budget.

    `getaddrinfo` takes no timeout. A lookup that outlasts the budget finishes in its own thread, which
    glibc bounds at `timeout:5` x `attempts:2` per nameserver, and its answer goes unread.
    """
    found: queue.SimpleQueue[list[tuple[Any, ...]] | OSError] = queue.SimpleQueue()

    def resolve() -> None:
        try:
            found.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as failed:
            found.put(failed)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses = found.get(timeout=budget())
    except queue.Empty:
        raise TimeoutError(f'{host} did not resolve inside the budget') from None
    if isinstance(addresses, OSError):
        raise addresses
    return addresses


def _connect(host: str, port: int, budget: Callable[[], float]) -> socket.socket:
    """A connected socket, trying every address the name resolves to inside one budget.

    FOOTGUN: `socket.create_connection` arms its timeout per address, so a name answering with four
    records spends four budgets before it gives up.
    """
    failure: OSError = OSError(f'{host}:{port} resolved to no address')
    for family, kind, proto, _canonical, address in _resolved(host, port, budget):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(budget())
            sock.connect(address)
            return sock
        except OSError as exc:
            sock.close()
            failure = exc
    raise failure


def _request(method: str, path: str, host_header: str, headers: Mapping[str, str]) -> bytes:
    fields = (f'{name}: {value}' for name, value in headers.items())
    lines = [f'{method} {path} HTTP/1.1', f'Host: {host_header}', *fields]
    return ('\r\n'.join(lines) + '\r\n\r\n').encode('latin-1')


def _status_line(answer: bytes) -> int | None:
    """The status code out of `HTTP/1.x NNN …`, or None from anything that is not that.

    A status is three digits (RFC 9112). `int` refuses a digit string past 4300 digits.
    """
    fields = answer.split(b'\r\n', 1)[0].split(None, 2)
    if len(fields) < 2 or not fields[0].startswith(b'HTTP/'):
        return None
    if len(fields[1]) != 3 or not fields[1].isdigit():
        return None
    return int(fields[1])


def _answer_to(
    host: str, port: int, request: bytes, budget: Callable[[], float], *, tls: bool, whole: Callable[[bytes], bool]
) -> bytes:
    """What the server on `host:port` answers `request` with: the bytes read until `whole` holds, the
    server closes, or `_MAX_ANSWER_BYTES`. Raises `OSError` where the connection fails or the budget runs out.

    A redirect is returned as its status and never followed, so the caller's headers reach only this origin.
    """
    sock = _connect(host, port, budget)
    try:
        if tls:
            sock.settimeout(budget())
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        sock.settimeout(budget())
        sock.sendall(request)
        answer = b''
        while len(answer) < _MAX_ANSWER_BYTES and not whole(answer):
            sock.settimeout(budget())
            chunk = sock.recv(_MAX_ANSWER_BYTES - len(answer))
            if not chunk:
                break
            answer += chunk
        return answer
    finally:
        sock.close()


def _answer_of_status(status: int | None) -> Answer:
    if status is None:
        return Answer.silent
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return Answer.refused
    if status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return Answer.cold
    return Answer.final


class _Received:
    """Bytes a server sent, in the shape `http.client.HTTPResponse` reads an answer from."""

    def __init__(self, answer: bytes):
        self._answer = answer

    def makefile(self, _mode: str) -> io.BytesIO:
        return io.BytesIO(self._answer)


def _carries_alive_seconds(answer: bytes) -> bool:
    """Whether the body of the HTTP answer in `answer` is the keepalive call's JSON object."""
    response = http.client.HTTPResponse(_Received(answer))  # pyright: ignore[reportArgumentType]
    try:
        response.begin()
        alive = json.loads(response.read())[ALIVE_SECONDS]
    except (http.client.HTTPException, ValueError, LookupError, TypeError):
        return False
    return alive is None or isinstance(alive, int)


def _whole_only_at_close(_answer: bytes) -> bool:
    """False: the answer to a request that asks to close is whole where the server closes."""
    return False


def _head_whole(answer: bytes) -> bool:
    return b'\r\n\r\n' in answer


def _positronic_answer(
    host: str, port: int, headers: Mapping[str, str], budget: Callable[[], float], *, tls: bool, default_port: int
) -> Answer:
    """The keepalive call, and the upgrade on the root where the server serves no keepalive call."""
    host_header = netloc(HostPortAddress(host, port, '', ''), default_port)
    asked = {**headers, 'Content-Length': '0', 'Connection': 'close'}
    keepalive = _request('POST', KEEPALIVE_PATH, host_header, asked)
    answer = _answer_to(host, port, keepalive, budget, tls=tls, whole=_whole_only_at_close)
    status = _status_line(answer)
    if status == HTTPStatus.OK:
        return Answer.admitted if _carries_alive_seconds(answer) else Answer.final
    if status != HTTPStatus.NOT_FOUND:
        return _answer_of_status(status)
    # A server without the keepalive call answers it 404, and refuses an upgrade on its root with 403.
    upgrade = _request('GET', '/', host_header, {**headers, **_UPGRADE})
    status = _status_line(_answer_to(host, port, upgrade, budget, tls=tls, whole=_head_whole))
    if status in (HTTPStatus.FORBIDDEN, HTTPStatus.SWITCHING_PROTOCOLS):
        return Answer.no_keepalive
    return _answer_of_status(status)


def _announcement_whole(answer: bytes) -> bool:
    """Whether `answer` holds the head of an answer to an upgrade, and the first frame byte where it upgraded."""
    head, found, frames = answer.partition(b'\r\n\r\n')
    return bool(found) and (_status_line(head) != HTTPStatus.SWITCHING_PROTOCOLS or len(frames) > 0)


def _roboarena_answer(host: str, port: int, headers: Mapping[str, str], budget: Callable[[], float]) -> Answer:
    """The upgrade on the root, and the first frame, which a roboarena server announces itself with."""
    upgrade = _request('GET', '/', f'{bracket_ipv6(host)}:{port}', {**headers, **_UPGRADE})
    answer = _answer_to(host, port, upgrade, budget, tls=False, whole=_announcement_whole)
    status = _status_line(answer)
    if status != HTTPStatus.SWITCHING_PROTOCOLS:
        return _answer_of_status(status)
    frames = answer.partition(b'\r\n\r\n')[2]
    if not frames:
        return Answer.silent
    opcode = frames[0] & 0x0F
    return {_BINARY_FRAME: Answer.no_keepalive, _TEXT_FRAME: Answer.final}.get(opcode, Answer.silent)


# The readiness call of each wire it speaks. A gRPC call needs HTTP/2, which this module does not speak.
_READINESS_CALLS: dict[str, Callable[[str, int, Mapping[str, str], Callable[[], float]], Answer]] = {
    WebsocketClientWire.NAME: functools.partial(
        _positronic_answer, tls=False, default_port=WebsocketClientWire.DEFAULT_PORT
    ),
    WebsocketTlsClientWire.NAME: functools.partial(
        _positronic_answer, tls=True, default_port=WebsocketTlsClientWire.DEFAULT_PORT
    ),
    RoboarenaClientWire.NAME: _roboarena_answer,
}
READINESS_WIRES = frozenset(_READINESS_CALLS)


def _budget_until(deadline_s: float) -> Callable[[], float]:
    """What is left of a `deadline_s` clock started now. Never zero, which would set a socket non-blocking."""
    deadline = time.monotonic() + deadline_s

    def budget() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f'the {deadline_s:g} s budget is spent')
        return left

    return budget


def readiness_of(
    wire: ClientWire[Any], host: str, port: int, deadline_s: float, headers: Mapping[str, str] | None = None
) -> Answer:
    """What the policy on `host:port` answers its readiness call on `wire` with, inside `deadline_s`.

    `wire` is one of `READINESS_WIRES`. On a positronic server the call is the keepalive call, and a call
    the server admits resets its idle timer, as a session does.
    """
    call = _READINESS_CALLS.get(wire.NAME)
    if call is None:
        raise ValueError(f'a readiness call speaks {", ".join(sorted(READINESS_WIRES))}, not {wire.NAME}')
    if deadline_s <= 0:
        return Answer.silent
    try:
        return call(host, port, headers or {}, _budget_until(deadline_s))
    except OSError as failed:
        _logger.debug('%s:%d did not answer its readiness call: %s', host, port, failed)
        return Answer.silent


def not_up_yet(answer: Answer) -> bool:
    """Whether this answer leaves the server still coming up: an answer asking to retry, or no answer.

    Any other answer settles it. A refusal is a verdict, and waiting on it only spends the deadline.
    """
    return answer in (Answer.cold, Answer.silent)


def serving(wire: ClientWire[Any], host: str, port: int, deadline_s: float) -> bool:
    """Whether the server on `host:port` is up: it answers its readiness call with no answer to wait on.

    A refusal counts, because the server sent it.
    """
    answer = readiness_of(wire, host, port, deadline_s)
    if not_up_yet(answer):
        _logger.debug('%s:%d answered its readiness call %s: not up yet', host, port, answer.value)
        return False
    _logger.info('%s:%d answered its readiness call %s: the server is up', host, port, answer.value)
    return True
