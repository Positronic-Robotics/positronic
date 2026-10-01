"""Whether a policy server serves, and whether its bearer-token gate holds.

The readiness call is the server's own wire: its keepalive call, and its probe where it serves none.
The server can be an image the caller did not write, so each readiness call runs in a forked child
process that is killed at a wall-clock deadline and holds a memory limit. That needs Linux: on macOS a
forked child can abort in a system library, and the limit reads `/proc`.
"""

import logging
import os
import resource
import secrets
import select
import signal
import socket
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import Enum
from http import HTTPStatus
from typing import Any

from positronic_wire.roboarena import RoboarenaAddress, RoboarenaClientWire, TextAnswer
from positronic_wire.websocket import WebsocketClientWire
from positronic_wire.wire import (
    AUTH_HEADER,
    SESSION_PATH,
    ClientWire,
    ConnectRefused,
    HostPortAddress,
    KeepaliveUnsupported,
    Refusal,
    SessionAddress,
    bearer,
    bracket_ipv6,
)

_logger = logging.getLogger(__name__)

# The session under a model id: a separate handler on a positronic server built before `4fec81f`, which
# runs the gate before it resolves the id. `gate` proves it too, so such a server cannot serve it open.
MODEL_SESSION_PATH = f'{SESSION_PATH}/0'

# A roboarena session opens on the server's root.
ROBOARENA_SESSION_PATH = '/'

# The paths a session opens on, by the wire `gate` proves. The server's own session opens on the first.
SESSION_PATHS_OF_WIRE: dict[str, tuple[str, ...]] = {
    WebsocketClientWire.NAME: (SESSION_PATH, MODEL_SESSION_PATH),
    RoboarenaClientWire.NAME: (ROBOARENA_SESSION_PATH,),
}


class Answer(Enum):
    """What one readiness call through a server's wire came back with."""

    admitted = 'admitted'  # the keepalive call took these headers
    no_keepalive = 'no keepalive'  # a server answers, and serves no keepalive call
    refused = 'refused'  # an answer refused the credential
    cold = 'cold'  # an answer asking to retry: a gateway, a 5xx, a 429
    final = 'final'  # an answer no retry changes: any other status, or a body the wire cannot read
    silent = 'silent'  # no answer: the connection failed, the probe found no server, or the deadline passed


# What a policy server itself answers its readiness call with, whether or not the caller holds its token.
POLICY_ANSWERS = frozenset({Answer.admitted, Answer.no_keepalive, Answer.refused})

# What a readiness call's child may map beyond what it inherits: room for the wire's thread stack and a
# TLS context. A flooding answer ends the child when it reaches this.
CHILD_HEADROOM_BYTES = 64 * 1024 * 1024

# What a child reports back, reason included, in one pipe write.
_REPORT_BYTES = 512

# What a child reports where the call raised something no answer explains.
_RAISED = 'raised'


def _answer_of_refusal(refusal: Refusal, *, answered: bool) -> Answer:
    """What a refusal says, where `answered` says whether a server's answer produced it."""
    if not answered:
        return Answer.silent
    return {Refusal.COLD: Answer.cold, Refusal.FORBIDDEN: Answer.refused, Refusal.FINAL: Answer.final}[refusal]


def answer_of(
    wire: ClientWire[Any], address: SessionAddress, headers: Mapping[str, str] | None, timeout: float
) -> Answer:
    """What the server at `address` answers one readiness call on `wire` with, in this process.

    The keepalive call, and the wire's probe where the server serves none. `timeout` bounds each phase
    as the wire times it, not the whole call.

    FOOTGUN: `ConnectRefused` does not say whether a server answered, so this reads its cause. A wire
    gives the library error as the cause, and a refusal read off a status has none.
    """
    started = time.monotonic()
    try:
        wire.keepalive(address, headers, timeout)
    except KeepaliveUnsupported:
        refusal = wire.probe(address, headers, max(0.0, timeout - (time.monotonic() - started)))
        if refusal is None:
            return Answer.no_keepalive
        # A cold probe found no server yet; the others read a status off one.
        return _answer_of_refusal(refusal, answered=refusal is not Refusal.COLD)
    except ConnectRefused as refused:
        return _answer_of_refusal(refused.refusal, answered=refused.__cause__ is None)
    return Answer.admitted


def _limit_memory() -> None:
    """Hold this process to what it maps now, plus `CHILD_HEADROOM_BYTES`."""
    with open('/proc/self/statm') as statm:
        mapped = int(statm.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = mapped + CHILD_HEADROOM_BYTES
    resource.setrlimit(resource.RLIMIT_AS, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))


def _report_of(ask: Callable[[], Answer]) -> str:
    """The line a child writes back: the answer's value, then why, where the call returned none."""
    try:
        _limit_memory()
        return ask().value
    except (ValueError, LookupError, TypeError, MemoryError, TextAnswer) as unreadable:
        # A body the wire cannot decode, a text greeting, or a flood: the server answered, and no retry changes it.
        return f'{Answer.final.value}\n{unreadable!r}'
    except Exception as failed:  # noqa: BLE001 — a raise would die with the child; the parent logs this report
        return f'{_RAISED}\n{failed!r}'


def _ask_in_child(ask: Callable[[], Answer], deadline_s: float, where: str) -> Answer:
    """`ask()` in a child process, killed at `deadline_s` and held to its memory limit.

    A killed child reads nothing more, so a trickling or a flooding server spends one deadline at most,
    and none of this process's memory.
    """
    if sys.platform != 'linux':
        raise NotImplementedError(f'a readiness call runs in a forked child, which needs Linux, not {sys.platform}')
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_end)
            os.write(write_end, _report_of(ask).encode()[:_REPORT_BYTES])
        finally:
            os._exit(0)
    os.close(write_end)
    reported = select.poll()
    reported.register(read_end, select.POLLIN)
    try:
        said = os.read(read_end, _REPORT_BYTES) if reported.poll(max(0, int(deadline_s * 1000))) else b''
    finally:
        os.close(read_end)
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        with suppress(ChildProcessError):
            os.waitpid(pid, 0)
    kind, _, reason = said.decode(errors='replace').partition('\n')
    if kind == _RAISED:
        # rules-allow: swallowed-error — a server the caller probes must not raise into the caller's loop.
        _logger.error('the readiness call on %s raised %s; read as no answer', where, reason)
        return Answer.silent
    if not kind:
        _logger.debug('%s did not answer inside %.0fs', where, deadline_s)
        return Answer.silent
    if reason:
        _logger.debug('%s answered as no policy server does: %s', where, reason)
    return Answer(kind)


def address_on(wire: ClientWire[Any], host: str, port: int) -> SessionAddress:
    """Where a session on `wire` opens on `host:port`.

    A positronic server serves its session on `SESSION_PATH`. A roboarena server takes no route, and
    that wire's address takes the host and the port alone.
    """
    if wire.ADDRESS is RoboarenaAddress:
        return RoboarenaAddress(host, port)
    if wire.ADDRESS is HostPortAddress:
        return HostPortAddress(host, port, SESSION_PATH, '')
    raise ValueError(f'{wire.NAME} dials no host and port')


def readiness_of(
    wire: ClientWire[Any], host: str, port: int, deadline_s: float, headers: Mapping[str, str] | None = None
) -> Answer:
    """What the policy on `host:port` answers its readiness call on `wire` with, inside `deadline_s`.

    On a positronic server the call is the keepalive call. A call the server admits resets its idle
    timer, as a session does: one that carries the token, or any call to a server with no token.

    FOOTGUN: the child is forked. In a caller with other threads it can block on a lock one of them held
    at the fork, and then reads as `Answer.silent` at the deadline.
    """
    address = address_on(wire, host, port)
    return _ask_in_child(lambda: answer_of(wire, address, headers, deadline_s), deadline_s, f'{host}:{port}')


def warming(answer: Answer) -> bool:
    """Whether this answer leaves the server still coming up.

    True for an answer asking to retry and for no answer. Any other answer settles it: a refused token
    is a verdict, and waiting on it only spends the deadline.
    """
    return answer in (Answer.cold, Answer.silent)


def serving(wire: ClientWire[Any], host: str, port: int, deadline_s: float) -> bool:
    """Whether the server on `host:port` is up: it answers its readiness call with no answer to wait on.

    The call carries no token, so a refusal counts. Whether the server admits its own token is `gate`'s
    to prove.
    """
    answer = readiness_of(wire, host, port, deadline_s)
    if warming(answer):
        _logger.debug('%s:%d answered its readiness call %s: not up yet', host, port, answer.value)
        return False
    _logger.info('%s:%d answered its readiness call %s: the server is up', host, port, answer.value)
    return True


# The status line is all `status_of` reads. A server that floods instead of answering reaches this first
# and reads as no answer, so nothing unbounded accumulates.
_MAX_HEAD_BYTES = 8192


def _request(path: str, host_header: str, headers: Mapping[str, str]) -> bytes:
    """The GET `status_of` sends. `Connection: close` because the status line is the whole answer, unless
    the caller's headers make it an upgrade."""
    sent = {'Connection': 'close', **headers}
    lines = [f'GET {path} HTTP/1.1', f'Host: {host_header}', *(f'{name}: {value}' for name, value in sent.items())]
    return ('\r\n'.join(lines) + '\r\n\r\n').encode()


def _status_line(head: bytes) -> int | None:
    """The status code out of `HTTP/1.x NNN …`, or None from anything that is not that.

    A status is three digits (RFC 9112). `int` refuses a digit string past 4300 digits.
    """
    fields = head.split(b'\r\n', 1)[0].split(None, 2)
    if len(fields) < 2 or not fields[0].startswith(b'HTTP/'):
        return None
    if len(fields[1]) != 3 or not fields[1].isdigit():
        return None
    return int(fields[1])


def _connect(host: str, port: int, budget: Callable[[], float]) -> socket.socket:
    """A connected socket, trying every address the name resolves to inside one budget.

    FOOTGUN: `socket.create_connection` arms its timeout per address, so a name answering with four
    records spends four budgets before it gives up.
    """
    failure: Exception = OSError(f'{host}:{port} resolved to no address')
    for family, kind, proto, _canonical, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        left = budget()
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(left)
            sock.connect(address)
            return sock
        except OSError as exc:
            sock.close()
            failure = exc
    raise failure


def _budget_until(deadline_s: float) -> Callable[[], float]:
    """What is left of a `deadline_s` clock started now. Never zero, which would set a socket non-blocking."""
    deadline = time.monotonic() + deadline_s

    def budget() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f'the {deadline_s:.0f}s budget is spent')
        return left

    return budget


def status_of(host: str, port: int, path: str, headers: Mapping[str, str], deadline_s: float) -> int | None:
    """The status the server on `host:port` answers a GET of `path` with, or None where nothing answered
    inside `deadline_s`.

    The budget is wall-clock over connect, send and every read, so a server that trickles one byte at a
    time cannot outlast it. A redirect is returned as its status and never followed, so `AUTH_HEADER`
    reaches only the origin it was aimed at.

    FOOTGUN: name resolution runs before the clock and takes no timeout. glibc bounds it at `timeout:5` x
    `attempts:2` per nameserver, so a call on a name can overrun `deadline_s` by seconds.
    """
    budget = _budget_until(deadline_s)
    head = b''
    sock: socket.socket | None = None
    try:
        sock = _connect(host, port, budget)
        sock.settimeout(budget())
        sock.sendall(_request(path, f'{bracket_ipv6(host)}:{port}', headers))
        while b'\r\n' not in head and len(head) < _MAX_HEAD_BYTES:
            sock.settimeout(budget())
            chunk = sock.recv(_MAX_HEAD_BYTES - len(head))
            if not chunk:
                break
            head += chunk
    except (OSError, ValueError) as exc:
        _logger.debug('%s:%d%s does not answer yet: %s', host, port, path, exc)
        return None
    finally:
        if sock is not None:
            sock.close()
    return _status_line(head)


def _wrong_token(token: str) -> str:
    """A fresh, well-formed bearer that is not `token`: a fixed value could be refused by shape or by name
    while every other wrong token is served."""
    while (candidate := secrets.token_urlsafe(32)) == token:
        pass
    return candidate


# A WebSocket handshake; the key is fixed because only the status line is read.
_UPGRADE = {
    'Upgrade': 'websocket',
    'Connection': 'Upgrade',
    'Sec-WebSocket-Key': 'AAAAAAAAAAAAAAAAAAAAAA==',
    'Sec-WebSocket-Version': '13',
}


class Gate(Enum):
    """What the token gate does to a caller, read from the routes an inference runs over."""

    holds = 'holds'  # every route refuses a caller with no valid token and admits the server's own
    open = 'open'  # a route serves a caller that holds no token, or a wrong one
    token_rejected = 'token rejected'  # a route does not serve the server's own token


def gate(
    wire: ClientWire[Any], host: str, port: int, token: str, deadline_s: float, *, prove_own_token: bool = True
) -> Gate:
    """Prove the gate on the readiness call and on every session route of `wire`.

    A tokenless and a wrong-token caller must be refused on each. `token` must pass the readiness call
    and open the first session route, the one a client opens. `deadline_s` bounds each call.

    FOOTGUN: the own-token session probe upgrades, so it takes a session. A server that serves one
    session at a time refuses its client while the probe holds it, so pass `prove_own_token=False` once
    the gate has held; the refusals are then the whole check.
    """
    sessions = SESSION_PATHS_OF_WIRE.get(wire.NAME)
    if sessions is None:
        raise ValueError(f'the gate proves the wires {", ".join(SESSION_PATHS_OF_WIRE)}, not {wire.NAME}')
    wrong = {AUTH_HEADER: bearer(_wrong_token(token))}
    own = {AUTH_HEADER: bearer(token)}

    def answered(headers: Mapping[str, str]) -> Answer:
        return readiness_of(wire, host, port, deadline_s, headers)

    def upgraded(headers: Mapping[str, str], path: str) -> bool:
        return status_of(host, port, path, {**_UPGRADE, **headers}, deadline_s) == HTTPStatus.SWITCHING_PROTOCOLS

    if answered({}) is Answer.admitted or answered(wrong) is Answer.admitted:
        _logger.warning('%s:%d admits a readiness call with no valid token', host, port)
        return Gate.open
    # A server with no keepalive call proves the token on the session route alone.
    if prove_own_token and answered(own) not in (Answer.admitted, Answer.no_keepalive):
        return Gate.token_rejected
    for path in sessions:
        if upgraded({}, path) or upgraded(wrong, path):
            _logger.warning('%s:%d%s upgrades a caller with no valid token', host, port, path)
            return Gate.open
    if prove_own_token and not upgraded(own, sessions[0]):
        return Gate.token_rejected
    return Gate.holds
