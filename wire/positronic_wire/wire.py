"""The client side of the transports a session runs over, and the facts both ends of a wire share.

A wire carries a protocol's frames as opaque bytes and reads none of them, and encodes the keepalive
call itself. Nothing here reads a URL:
a caller names the wire it wants by ``ClientWire.NAME`` (``registry.CLIENT_WIRES``), and the wire alone
spells whatever its library takes.
"""

import abc
import dataclasses
import json
from collections.abc import Iterator, Mapping
from enum import Enum
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'
# The keepalive route beside the session, and the keys of its JSON answer (`KeepaliveAnswer`).
KEEPALIVE_PATH = f'{API_PATH}/keepalive'
ALIVE_SECONDS = 'alive_seconds'
DECLARED_PARAMS = 'declared_params'


@dataclasses.dataclass(frozen=True)
class KeepaliveAnswer:
    """The answer to a keepalive call.

    ``alive_seconds`` is the seconds the server stays alive after the call, or ``None`` for a server with no idle
    timeout. ``declared_params`` names the session parameters the server takes, or is ``None`` for a server that
    does not list them.
    """

    alive_seconds: int | None
    declared_params: tuple[str, ...] | None = None

    def encode(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()

    @classmethod
    def decode(cls, body: bytes) -> Self:
        """The answer in ``body``. Raises ``ValueError`` where ``body`` is not a keepalive answer."""
        try:
            answer = json.loads(body)
            alive, declared = answer[ALIVE_SECONDS], answer.get(DECLARED_PARAMS)
        except (ValueError, LookupError, TypeError, RecursionError) as e:
            raise ValueError('not a keepalive answer') from e
        if not (alive is None or (isinstance(alive, int) and not isinstance(alive, bool))):
            raise ValueError(f'{ALIVE_SECONDS} {alive!r}')
        if declared is None:
            return cls(alive)
        if not (isinstance(declared, list) and all(isinstance(name, str) for name in declared)):
            raise ValueError(f'{DECLARED_PARAMS} {declared!r}')
        return cls(alive, tuple(declared))


def bracket_ipv6(host: str) -> str:
    """``host`` in the brackets an IPv6 literal needs before a port. Every other host is unchanged."""
    return f'[{host}]' if ':' in host else host


class SessionAddress(abc.ABC):
    """Where one session opens. Each wire declares the address it dials, and takes no other.

    ``path`` is the session route, ``SESSION_PATH``, and ``query`` carries the session params as written: the server
    reads each value as a JSON literal, and only whoever wrote the query knows whether ``true`` means the
    bool or the string. Every wire carries both; how a wire names the server is its own.
    """

    path: str
    query: str

    @abc.abstractmethod
    def at_root(self) -> 'Self':
        """The same server, with no route and no params: what a probe asks for."""


@dataclasses.dataclass(frozen=True)
class HostPortAddress(SessionAddress):
    """A session on a server reached over the network. ``host`` is raw: each wire spells it itself."""

    host: str
    port: int
    path: str
    query: str

    def at_root(self) -> 'Self':
        return dataclasses.replace(self, path='', query='')


@dataclasses.dataclass(frozen=True)
class UnixSocketAddress(SessionAddress):
    """A session on a server on this machine, opened on the socket it bound.

    A socket is same-machine by construction, so the address names no host and no port.
    """

    uds: Path
    path: str
    query: str

    def __post_init__(self) -> None:
        # A relative path is resolved against the directory each process was started from, so it names a
        # different socket to each caller.
        if not self.uds.is_absolute():
            raise ValueError(f'{self.uds!r} is a relative socket path; name an absolute one')

    def at_root(self) -> 'Self':
        return dataclasses.replace(self, path='', query='')


AddressT = TypeVar('AddressT', bound=SessionAddress)


# The largest frame a wire may carry. An observation is a stack of camera frames, and the gRPC default
# of 4 MiB refuses one.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class PeerDisconnected(Exception):
    """The peer ended the session."""


class KeepaliveUnsupported(Exception):
    """The server answers sessions but not the keepalive call."""


class ProbeUnsupported(Exception):
    """The wire carries no probe. A caller asks ``keepalive`` whether the server is up."""


class Refusal(Enum):
    """What a refused connect says about the server."""

    COLD = 'cold'  # a backend still starting; retry to the deadline
    FORBIDDEN = 'forbidden'  # a cold backend, or a refused credential; a few attempts, then surface
    FINAL = 'final'  # a permanent refusal; surface at once
    # Nothing answered: no connection, or none before the timeout or the close. Retry to the deadline.
    SILENT = 'silent'
    # The resolver had no answer for the host name yet. Retry for a limited time, then surface.
    UNRESOLVED = 'unresolved'


def retry_pauses() -> Iterator[float]:
    """The seconds to pause before each retry of a refused call: 1, then double the last, up to 30."""
    pause = 1.0
    while True:
        yield pause
        pause = min(pause * 2, 30.0)


class ConnectRefused(Exception):
    """A wire could not open a session. The library error that refused it is the cause."""

    def __init__(self, refusal: Refusal, message: str):
        super().__init__(message)
        self.refusal = refusal


def netloc(address: HostPortAddress, default_port: int) -> str:
    """``host:port``, less the port the wire defaults to."""
    host = bracket_ipv6(address.host)
    return host if address.port == default_port else f'{host}:{address.port}'


class ClientWire(abc.ABC, Generic[AddressT]):
    """The client side of one wire: what it is called, the address it dials, and how it dials one.

    A wire over TLS is a member of its own, not a flag on the plain one.
    """

    # The name a caller selects this wire by.
    NAME: ClassVar[str]
    # The address this wire dials. A caller that built another wire's address is refused by it.
    ADDRESS: ClassVar[type[SessionAddress]]

    @abc.abstractmethod
    def session_url(self, address: AddressT) -> str:
        """``address`` as this wire names one session, for a log and for an error.

        Each wire dials its own way: a member whose library takes this spelling dials it, and one
        that takes a target or a socket dials that instead.
        """

    @abc.abstractmethod
    def dial(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> 'ClientConnection':
        """A client's end of one session on ``address``. Raises ``ConnectRefused`` when it does not open."""

    @abc.abstractmethod
    def keepalive(self, address: AddressT, headers: Mapping[str, str] | None, timeout: float) -> KeepaliveAnswer:
        """Reset the idle timer of the server on ``address``, outside any session, and return its answer.

        ``timeout`` bounds the whole call. Raises ``KeepaliveUnsupported`` where the server serves
        sessions but not the call, and ``ConnectRefused`` otherwise, in the terms ``probe`` uses.
        """

    @abc.abstractmethod
    def probe(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> Refusal | None:
        """Whether a server answers at ``address``, without opening a session.

        ``headers`` are the ones ``dial`` sends: an edge that authenticates on them lets the probe through to
        the server behind it, so the probe wakes what a session would reach. ``None`` when a server answers.
        A ``Refusal`` says why none did, in the terms ``dial`` uses: ``COLD`` for a backend still starting,
        ``SILENT`` for a port nothing answers on, ``FORBIDDEN`` for a credential the edge refused, ``FINAL`` for
        a refusal no retry reaches. Raises ``ProbeUnsupported`` on a wire that carries no probe.
        """


class ClientConnection(abc.ABC):
    """A client's end of one open session."""

    @abc.abstractmethod
    def send(self, message: bytes, timeout: float | None = None) -> None:
        """Send one message, and return once the wire has written it.

        Raises ``TimeoutError`` when the wire cannot write it within ``timeout`` seconds, and the session is then
        unusable. Raises ``PeerDisconnected`` once the session has ended.
        """

    @abc.abstractmethod
    def recv(self, timeout: float | None = None) -> bytes:
        """The next message.

        Raises ``TimeoutError`` when none arrives in time, ``PeerDisconnected`` once the server ends the
        session, and ``ConnectRefused`` when the server refuses the session before its first message.
        """

    @abc.abstractmethod
    def close(self, timeout: float | None = None) -> str:
        """Close this end, and report what the wire saw, for the log.

        ``timeout`` shortens the wait for the peer to answer the close. It never lengthens the wire's own bound,
        and ``None`` keeps that bound.
        The report says whether the peer answered the close, in the wire's own terms. A server that still
        holds a session strands the next session's handshake, and only the wire can see that.
        """
