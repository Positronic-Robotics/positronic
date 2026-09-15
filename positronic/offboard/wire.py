"""The transports a session runs over, and the two ends of one open session.

A wire carries a session's ``protocol`` frames as opaque bytes and reads none of them. It encodes a
unary call itself, because the two wires spell one differently.
"""

import abc
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import Any, ClassVar, NamedTuple

from starlette.datastructures import QueryParams

# The server's HTTP API, and the route a session opens on under it.
API_PATH = '/api/v1'
SESSION_PATH = f'{API_PATH}/session'
# The model catalogue, served under the HTTP API.
MODELS_ROUTE = 'models'
MODELS_PATH = f'{API_PATH}/{MODELS_ROUTE}'


class Verb(NamedTuple):
    """One unary call beside the session, and the spelling each wire gives it.

    ``ready`` asks what the server can do now, and ``warm`` asks it to pay the first inference, naming
    the run's task under ``positronic.keys.TASK``. Both answer one ``protocol.Readiness`` record and
    neither opens a session, so a caller reads the answer as often as it needs it.
    """

    name: str
    http_method: str
    grpc_method: str

    @property
    def path(self) -> str:
        """The HTTP route this verb answers on."""
        return f'{API_PATH}/{self.name}'


READY = Verb('ready', 'GET', 'Ready')
WARM = Verb('warm', 'POST', 'Warm')
VERBS = (READY, WARM)


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


class VerbUnsupported(Exception):
    """The server answers sessions but not this verb."""


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
    def call(
        self,
        address: SessionAddress,
        verb: Verb,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        """What the server on ``address`` answers ``verb`` with, outside any session.

        Each wire encodes the payload and the answer its own way, so both hold plain data alone. Raises
        ``VerbUnsupported`` where the server serves sessions but not this verb, and ``ConnectRefused``
        where it answers nothing.
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


class ServerConnection(abc.ABC):
    """A server's end of one open session."""

    @property
    @abc.abstractmethod
    def peer(self) -> str:
        """Whom this session serves, for the log."""

    @property
    @abc.abstractmethod
    def endpoint(self) -> Endpoint:
        """Where the wire that accepted this session serves."""

    @property
    @abc.abstractmethod
    def query_params(self) -> QueryParams:
        """The session params the client asked for."""

    @abc.abstractmethod
    async def send(self, message: bytes) -> None:
        """Raises ``PeerDisconnected`` once the client ends the session."""

    @abc.abstractmethod
    async def receive(self) -> bytes:
        """The next message. Raises ``PeerDisconnected`` once the client ends the session."""

    @abc.abstractmethod
    async def refuse(self, reason: str) -> None:
        """End a session the server cannot serve, and tell the client why."""


# What a wire hands the server for each session it accepts: the connection, and the model the route
# names, or ``None`` for the model the server pinned.
SessionHandler = Callable[[ServerConnection, str | None], Awaitable[None]]

# What a wire hands the server for one unary call: the verb, and the payload the caller sent.
VerbHandler = Callable[[Verb, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]

# Whether the session headers carry a credential the server accepts. Header names are lower case.
Authorized = Callable[[Mapping[str, str]], bool]


class Wire(abc.ABC):
    """One transport that sessions and unary calls arrive on.

    A wire reads its own route for the model a session names, and refuses an unauthorized peer before
    the session opens.
    """

    @property
    @abc.abstractmethod
    def endpoint(self) -> Endpoint:
        """Where this wire serves. The port is known once ``start`` returns."""

    @abc.abstractmethod
    async def start(self, session: SessionHandler, verbs: VerbHandler, authorized: Authorized) -> None:
        """Bind, and give every accepted session to ``session`` and every unary call to ``verbs``.

        Raises when the port is not free.
        """

    @abc.abstractmethod
    async def serve(self) -> None:
        """Carry sessions until ``stop``, or until the wire ends for its own reason."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """End the wire, and every session on it."""
