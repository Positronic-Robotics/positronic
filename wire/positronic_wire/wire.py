"""The client side of the transports a session runs over, and the facts both ends of a wire share.

A wire carries the ``protocol`` frames as opaque bytes and reads none of them.
"""

import abc
from collections.abc import Mapping
from enum import Enum
from typing import ClassVar, NamedTuple

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'
# The model catalogue, served under the HTTP API.
MODELS_ROUTE = 'models'
MODELS_PATH = f'{API_PATH}/{MODELS_ROUTE}'


def default_port(secure: bool) -> int:
    """The port a URL naming none opens on."""
    return 443 if secure else 80


def bracket_ipv6(host: str) -> str:
    """``host`` in the brackets an IPv6 literal needs before a port. Every other host is unchanged."""
    return f'[{host}]' if ':' in host else host


class SessionAddress(NamedTuple):
    """Where one session opens. ``host`` is raw: each wire spells it for its own syntax."""

    host: str
    port: int
    path: str
    query: str
    secure: bool

    @property
    def netloc(self) -> str:
        """``host:port``, less the port a URL at this TLS setting defaults to."""
        host = bracket_ipv6(self.host)
        return host if self.port == default_port(self.secure) else f'{host}:{self.port}'

    def url(self, scheme: str) -> str:
        """This session as a URL on ``scheme``."""
        query = f'?{self.query}' if self.query else ''
        return f'{scheme}://{self.netloc}{self.path}{query}'


# The largest frame a session may carry, on either wire. An observation is a stack of camera frames, and
# the gRPC default of 4 MiB refuses one.
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


class Endpoint(NamedTuple):
    """Where a wire serves."""

    host: str
    port: int


class Scheme(NamedTuple):
    """A URL scheme that selects a wire, and whether it names TLS."""

    text: str
    secure: bool


class ClientWire(abc.ABC):
    """The client side of one wire: the schemes that select it, how it spells a session, and how it dials one."""

    # The URL scheme that selects this wire, and the one that selects it over TLS.
    SCHEME: ClassVar[str]
    SECURE_SCHEME: ClassVar[str]
    # Other schemes that select this wire. Each one names whether it carries TLS.
    ALIASES: ClassVar[tuple[Scheme, ...]] = ()

    def schemes(self) -> tuple[Scheme, ...]:
        """Every URL scheme that selects this wire."""
        return (Scheme(self.SCHEME, secure=False), Scheme(self.SECURE_SCHEME, secure=True), *self.ALIASES)

    def session_url(self, address: SessionAddress) -> str:
        """``address`` as this wire spells it."""
        return address.url(self.SECURE_SCHEME if address.secure else self.SCHEME)

    @abc.abstractmethod
    def api_url(self, address: SessionAddress) -> str | None:
        """The server's HTTP API beside this wire, or ``None`` where the wire's port carries sessions alone."""

    @abc.abstractmethod
    def dial(
        self, address: SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> 'ClientConnection':
        """A client's end of one session on ``address``. Raises ``ConnectRefused`` when it does not open."""

    @abc.abstractmethod
    def probe(self, address: SessionAddress, open_timeout: float) -> Refusal | None:
        """Whether a server answers at ``address``, without opening a session.

        ``None`` when one does. A ``Refusal`` says why none did, in the terms ``dial`` uses: ``COLD`` for a
        backend still starting or a port nothing answers on, ``FINAL`` for one no retry reaches.
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
