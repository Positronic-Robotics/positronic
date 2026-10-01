"""Whether a policy server serves.

The readiness call is the server's own wire: its keepalive call, and its probe where it serves none.
Each readiness call runs in a forked Linux child that holds a memory limit and is killed at a
wall-clock deadline, so a server the caller did not write cannot hold the caller.
"""

import logging
import os
import resource
import select
import signal
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import Enum
from typing import Any

from positronic_wire.roboarena import RoboarenaAddress, RoboarenaClientWire, TextAnswer
from positronic_wire.websocket import WebsocketClientWire, WebsocketTlsClientWire
from positronic_wire.wire import (
    SESSION_PATH,
    ClientWire,
    ConnectRefused,
    HostPortAddress,
    KeepaliveUnsupported,
    Refusal,
    SessionAddress,
)

_logger = logging.getLogger(__name__)


class Answer(Enum):
    """What one readiness call through a server's wire came back with."""

    admitted = 'admitted'  # the keepalive call took these headers
    no_keepalive = 'no keepalive'  # a server answers, and serves no keepalive call
    refused = 'refused'  # an answer refused the credential
    cold = 'cold'  # an answer asking to retry: a gateway, a 5xx, a 429
    final = 'final'  # an answer no retry changes: any other status, or a body the wire cannot read
    silent = 'silent'  # no answer: the connection failed, the probe found no server, or the deadline passed


# The answers that only a policy server gives to its readiness call.
POLICY_ANSWERS = frozenset({Answer.admitted, Answer.no_keepalive, Answer.refused})

# The wires a readiness call reads. Each refuses with a library error as the cause only where no server
# answered. The gRPC wires do not: every refusal they raise chains the status the server sent.
READINESS_WIRES = frozenset({WebsocketClientWire.NAME, WebsocketTlsClientWire.NAME, RoboarenaClientWire.NAME})

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


def _require_readiness_wire(wire: ClientWire[Any]) -> None:
    if wire.NAME not in READINESS_WIRES:
        raise ValueError(f'a readiness call reads the wires {", ".join(sorted(READINESS_WIRES))}, not {wire.NAME}')


def answer_of(
    wire: ClientWire[Any], address: SessionAddress, headers: Mapping[str, str] | None, timeout: float
) -> Answer:
    """What the server at `address` answers one readiness call on `wire` with, in this process.

    The keepalive call, and the wire's probe where the server serves none. `timeout` bounds each phase
    as the wire times it, not the whole call. `wire` is one of `READINESS_WIRES`.

    FOOTGUN: `ConnectRefused` does not say whether a server answered, so this reads its cause. A wire
    in `READINESS_WIRES` gives the library error as the cause, and a refusal read off a status has none.
    """
    _require_readiness_wire(wire)
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
    # rules-allow: swallowed-error — a raise dies with the child; the parent logs this report at ERROR.
    except Exception as failed:  # noqa: BLE001
        return f'{_RAISED}\n{failed!r}'


def _ask_in_child(ask: Callable[[], Answer], deadline_s: float, where: str) -> Answer:
    """`ask()` in a child process, killed at `deadline_s` and held to its memory limit.

    A killed child reads nothing more, so a trickling or a flooding server spends one deadline at most,
    and none of this process's memory.
    """
    # The memory limit reads `/proc`, and on macOS a forked child can abort in a system library.
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

    On a positronic server the call is the keepalive call, and a call the server admits resets its idle
    timer, as a session does.

    FOOTGUN: the child is forked. In a caller with other threads it can block on a lock one of them held
    at the fork, and then reads as `Answer.silent` at the deadline.
    """
    _require_readiness_wire(wire)
    address = address_on(wire, host, port)
    return _ask_in_child(lambda: answer_of(wire, address, headers, deadline_s), deadline_s, f'{host}:{port}')


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
