"""The client side of the transports a session runs over, and the facts both ends of a wire share.

A wire carries a protocol's frames as opaque bytes and reads none of them, and encodes a control call
itself. Nothing here reads a URL:
a caller names the wire it wants by ``ClientWire.NAME`` (``registry.CLIENT_WIRES``), and the wire alone
spells whatever its library takes.
"""

import abc
import dataclasses
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Generic, NamedTuple, Self, TypeVar

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'


class ControlCall(NamedTuple):
    """A call beside the session, and how each wire spells it.

    ``ready`` asks what the server can do now, and ``warm`` asks it to warm, naming the run's task under
    ``positronic.keys.TASK``. Both answer one ``positronic.offboard.protocol.Readiness`` record, and
    neither opens a session.
    """

    name: str
    http_method: str
    grpc_method: str

    @property
    def http_path(self) -> str:
        return f'{API_PATH}/{self.name}'


READY = ControlCall('ready', 'GET', 'Ready')
WARM = ControlCall('warm', 'POST', 'Warm')
CONTROL_CALLS = (READY, WARM)


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


class ControlCallUnsupported(Exception):
    """The server answers sessions but not this control call."""


class Refusal(Enum):
    """What a refused connect says about the server."""

    COLD = 'cold'  # a backend still starting; retry to the deadline
    FORBIDDEN = 'forbidden'  # a cold backend, or a refused credential; a few attempts, then surface
    FINAL = 'final'  # a permanent refusal; surface at once


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
    def call(
        self,
        address: AddressT,
        control_call: ControlCall,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        """What the server on ``address`` answers ``control_call`` with, outside any session.

        ``payload`` and the answer are plain data, and each wire encodes them its own way. Each wire says
        what ``timeout`` bounds: the whole call, or each phase its transport times. Raises
        ``ControlCallUnsupported`` where the server serves sessions but not this control call, and
        ``ConnectRefused`` where it answers nothing.
        """

    @abc.abstractmethod
    def probe(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> Refusal | None:
        """Whether a server answers at ``address``, without opening a session.

        ``headers`` are the ones ``dial`` sends: an edge that authenticates on them lets the probe through to
        the server behind it, so the probe wakes what a session would reach. ``None`` when a server answers.
        A ``Refusal`` says why none did, in the terms ``dial`` uses: ``COLD`` for a backend still starting or
        a port nothing answers on, ``FORBIDDEN`` for a credential the edge refused, ``FINAL`` for a refusal
        no retry reaches.
        """


class ClientConnection(abc.ABC):
    """A client's end of one open session."""

    @abc.abstractmethod
    def send(self, message: bytes) -> None:
        """Send one message, and return once the wire has written it.

        The caller starts the answer's timeout when this returns, so a send that returns early bills its
        own upload to that timeout. Raises ``PeerDisconnected`` once the session has ended.
        """

    @abc.abstractmethod
    def recv(self, timeout: float | None = None) -> bytes:
        """The next message.

        Raises ``TimeoutError`` when none arrives in time, ``PeerDisconnected`` once the server ends the
        session, and ``ConnectRefused`` when the server refuses the session before its first message.
        """

    @abc.abstractmethod
    def close(self) -> str:
        """Close this end, and report what the wire saw, for the log.

        The report says whether the peer answered the close, in the wire's own terms. A server that still
        holds a session strands the next session's handshake, and only the wire can see that.
        """
