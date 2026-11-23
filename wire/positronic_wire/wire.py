"""The client side of the transports a session runs over, and the facts both ends of a wire share.

A wire carries a protocol's frames as opaque bytes and reads none of them, and encodes the keepalive
call itself. Nothing here reads a URL:
a caller names the wire it wants by ``ClientWire.NAME`` (``registry.CLIENT_WIRES``), and the wire alone
spells whatever its library takes.
"""

import abc
import dataclasses
import ipaddress
import queue
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'
# The keepalive route beside the session, and the key its JSON answer carries the seconds under.
KEEPALIVE_PATH = f'{API_PATH}/keepalive'
ALIVE_SECONDS = 'alive_seconds'


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


class NotPublic(Exception):
    """The address names a host off the public internet, which ``ClientWire.probe_public`` does not dial."""


class Refusal(Enum):
    """What a refused connect says about the server."""

    COLD = 'cold'  # a backend still starting; retry to the deadline
    FORBIDDEN = 'forbidden'  # a cold backend, or a refused credential; a few attempts, then surface
    FINAL = 'final'  # a permanent refusal; surface at once
    # Nothing answered: no connection, or none before the timeout or the close. Retry to the deadline.
    SILENT = 'silent'
    # The resolver had no answer for the host name yet. Retry for a limited time, then surface.
    UNRESOLVED = 'unresolved'


# The resolver's authoritative answers that the host has no address. A resolver that timed out answers
# ``EAI_AGAIN``, which a later lookup can clear: a name can start resolving, where a misspelt one never does.
_NO_SUCH_HOST_ERRNOS = (socket.EAI_NONAME, socket.EAI_NODATA)


def lookup_refusal(failed: OSError) -> Refusal:
    """What a name lookup that raised ``failed`` says about the server."""
    if isinstance(failed, socket.gaierror) and failed.errno in _NO_SUCH_HOST_ERRNOS:
        return Refusal.FINAL
    if isinstance(failed, socket.gaierror) and failed.errno == socket.EAI_AGAIN:
        return Refusal.UNRESOLVED
    return Refusal.SILENT


def seconds_until(deadline: float) -> float:
    """Raises ``TimeoutError`` once ``deadline`` has passed, as a socket that timed out does."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('the call spent its timeout')
    return left


def addresses(host: str, port: int, deadline: float) -> list[tuple[Any, ...]]:
    """The addresses ``host`` resolves to, before ``deadline``, as ``socket.getaddrinfo`` answers them.

    ``getaddrinfo`` takes no timeout. A lookup past the deadline ends in its own thread, which the resolver
    configuration bounds (``timeout`` x ``attempts`` per nameserver), and nothing reads its answer.
    """
    found: queue.SimpleQueue[list[tuple[Any, ...]] | OSError] = queue.SimpleQueue()

    def look_up() -> None:
        try:
            found.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as failed:
            found.put(failed)

    threading.Thread(target=look_up, name='wire-lookup', daemon=True).start()
    try:
        resolved = found.get(timeout=seconds_until(deadline))
    except queue.Empty:
        raise TimeoutError(f'{host} did not resolve in time') from None
    if isinstance(resolved, OSError):
        raise resolved
    return resolved


# The domain a tailnet's own names sit under. Its resolver answers them with tailnet addresses.
TAILNET_DOMAIN = 'ts.net'
# The well-known NAT64 prefix. A translator delivers each address under it to the IPv4 address in its last 32 bits.
_NAT64 = ipaddress.IPv6Network('64:ff9b::/96')


def is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``ip`` reaches one host on the public internet.

    An IPv6 address that carries an IPv4 one is judged by the IPv4 one, which a translator delivers it to.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        nat64 = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF) if ip in _NAT64 else None
        carried = ip.ipv4_mapped or ip.sixtofour or nat64
        if carried is not None:
            return is_public(carried)
    return ip.is_global and not ip.is_multicast


def public_addresses(host: str, port: int, deadline: float) -> list[tuple[Any, ...]]:
    """The addresses ``host`` resolves to, before ``deadline``, where each one is public (``is_public``).

    A dial to these addresses makes no second lookup, which can answer another address. Raises ``NotPublic`` for a
    name under ``TAILNET_DOMAIN``, before the lookup, and for a name that resolves to any address that is not public.
    A failed lookup raises as ``addresses`` does.
    """
    name = host.rstrip('.').lower()
    if name == TAILNET_DOMAIN or name.endswith(f'.{TAILNET_DOMAIN}'):
        raise NotPublic(f'{host} is a tailnet name')
    resolved = addresses(host, port, deadline)
    private = [str(sockaddr[0]) for *_, sockaddr in resolved if not is_public(ipaddress.ip_address(sockaddr[0]))]
    if private:
        raise NotPublic(f'{host} resolves to {", ".join(private)}, which is off the public internet')
    return resolved


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
    def keepalive(self, address: AddressT, headers: Mapping[str, str] | None, timeout: float) -> int | None:
        """Reset the idle timer of the server on ``address``, outside any session.

        Returns the seconds the server stays alive after the call, or ``None`` for a server with no idle
        timeout. ``timeout`` bounds the whole call. Raises ``KeepaliveUnsupported`` where the server serves
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

    @abc.abstractmethod
    def probe_public(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> Refusal | None:
        """``probe``, of a server on the public internet alone.

        The host resolves once (``public_addresses``), and the probe dials the addresses that lookup answered, under
        the host's own name. Raises ``NotPublic`` before any dial where the address is not public, and
        ``ProbeUnsupported`` where ``probe`` does.
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
    def close(self, timeout: float | None = None) -> str:
        """Close this end, and report what the wire saw, for the log.

        ``timeout`` shortens the wait for the peer to answer the close. It never lengthens the wire's own bound,
        and ``None`` keeps that bound.
        The report says whether the peer answered the close, in the wire's own terms. A server that still
        holds a session strands the next session's handshake, and only the wire can see that.
        """
