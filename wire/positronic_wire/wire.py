"""The client side of the transports a session runs over, and the facts both ends of a wire share.

A wire carries a protocol's frames as opaque bytes and reads none of them. A caller names the wire it
wants by ``ClientWire.NAME`` (``registry.CLIENT_WIRES``), and nothing here reads a wire off a URL scheme.
The wire alone reads a URL into its address, and spells whatever its library takes.
"""

import abc
import dataclasses
import re
import urllib.parse
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Generic, Self, TypeVar

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'
# The model catalogue, served under the HTTP API: the route it answers on, and the key it answers under.
MODELS_ROUTE = 'models'
MODELS_PATH = f'{API_PATH}/{MODELS_ROUTE}'
MODELS_KEY = 'models'


def session_path(model: str = '') -> str:
    """The route a session on ``model`` opens on; the model the server pinned when ``model`` is empty.

    The id is percent-encoded as a path, so an id that is itself a path (a HuggingFace repo) keeps its
    slashes as separators and the server decodes the rest.
    """
    return f'{SESSION_PATH}/{urllib.parse.quote(model, safe="/")}' if model else SESSION_PATH


# A scheme and its delimiter, which only the start of a URL carries. A query may hold another URL.
_LEADING_SCHEME = re.compile(r'[A-Za-z][A-Za-z0-9+.-]*://')
# The session route in a socket URL's path. The socket path ends where it starts.
_SESSION_ROUTE = re.compile(rf'{re.escape(SESSION_PATH)}(?=/|$)')


def split_url(url: str) -> urllib.parse.SplitResult:
    """``url`` in its parts. A URL with no leading ``<scheme>://`` is ``host[:port]...``, and its scheme is empty."""
    stripped = url.strip()
    return urllib.parse.urlsplit(stripped if _LEADING_SCHEME.match(stripped) else f'//{stripped}')


def _session_route(path: str, url: str) -> str:
    """The session route ``path`` names: ``SESSION_PATH`` where it names none."""
    if path.rstrip('/') in ('', SESSION_PATH):
        return SESSION_PATH
    if not path.startswith(f'{SESSION_PATH}/'):
        raise ValueError(f'unexpected path {path!r} in {url!r}; expected {SESSION_PATH}[/<model>]')
    return path


def bracket_ipv6(host: str) -> str:
    """``host`` in the brackets an IPv6 literal needs before a port. Every other host is unchanged."""
    return f'[{host}]' if ':' in host else host


class SessionAddress(abc.ABC):
    """Where one session opens. Each wire declares the address it dials, and takes no other.

    ``path`` is ``session_path(model)``, and ``query`` carries the session params as written: the server
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

    @classmethod
    def from_url(cls, url: str, default_port: int) -> 'HostPortAddress':
        """The session ``[scheme://]host[:port][/api/v1/session[/<model>]][?query]`` names.

        The scheme is the caller's to read. Raises ``ValueError`` where ``url`` names no host, or a path that
        is not a session route.
        """
        split = split_url(url)
        if not split.hostname:
            raise ValueError(f'no host in {url!r}')
        port = default_port if split.port is None else split.port
        return cls(split.hostname, port, _session_route(split.path, url), split.query)

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
        if _SESSION_ROUTE.search(str(self.uds)):
            raise ValueError(f'{self.uds!r} holds the session route {SESSION_PATH!r}, so no URL can name it')

    @classmethod
    def from_url(cls, url: str) -> 'UnixSocketAddress':
        """The socket and the session ``scheme:///<socket>[/api/v1/session[/<model>]][?query]`` names.

        The socket path runs to the session route. Raises ``ValueError`` where ``url`` names a host, which a
        socket cannot reach, or a path that ends in a slash, which names a directory.
        """
        split = split_url(url)
        if split.netloc:
            raise ValueError(f'a socket URL names no host, and this one names {split.netloc!r}: {url!r}')
        route = _SESSION_ROUTE.search(split.path)
        socket, path = (split.path, '') if route is None else (split.path[: route.start()], split.path[route.start() :])
        if not PurePosixPath(socket).name or socket.endswith('/'):
            raise ValueError(f'a socket URL names a socket file, and this one names none: {url!r}')
        return cls(Path(urllib.parse.unquote(socket)), _session_route(path, url), split.query)

    def at_root(self) -> 'Self':
        return dataclasses.replace(self, path='', query='')


AddressT = TypeVar('AddressT', bound=SessionAddress)


# The largest frame a wire may carry. An observation is a stack of camera frames, and the gRPC default
# of 4 MiB refuses one.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class PeerDisconnected(Exception):
    """The peer ended the session."""


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
    # Whether a caller hands this wire the headers its own edge authenticates on. A server that another
    # party runs sits behind no edge of the caller's, and does not get them.
    TAKES_EDGE_HEADERS: ClassVar[bool]

    @abc.abstractmethod
    def address_of(self, url: str) -> AddressT:
        """The address ``url`` names on this wire. Raises ``ValueError`` where this wire dials none.

        The caller selects the wire; this reads the rest of ``url`` and ignores its scheme.
        """

    @abc.abstractmethod
    def session_url(self, address: AddressT) -> str:
        """``address`` as this wire names one session, for a log and for an error.

        Each wire dials its own way: a member whose library takes this spelling dials it, and one
        that takes a target or a socket dials that instead.
        """

    @abc.abstractmethod
    def list_models(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> list[str]:
        """The models the server at ``address`` serves, read over this wire's own transport.

        ``headers`` are the ones ``dial`` sends, so an edge that authenticates on them lets the read
        through. Raises ``ConnectRefused`` when the catalogue does not answer, in the terms ``dial``
        uses, and ``ValueError`` on a wire whose transport carries sessions alone.
        """

    @abc.abstractmethod
    def dial(self, address: AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> 'ClientConnection':
        """A client's end of one session on ``address``. Raises ``ConnectRefused`` when it does not open."""

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
