"""The transports a session runs over, and the two ends of one open session.

A wire carries the ``protocol`` frames as opaque bytes and reads none of them. ``websocket_wire`` and
``grpc_wire`` hold the two wires.
"""

import abc
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import NamedTuple, Protocol

from starlette.datastructures import QueryParams

# The route a session opens on: in the URL on the websocket wire, in the session metadata on the gRPC wire.
SESSION_PATH = '/api/v1/session'

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


class ClientConnection(Protocol):
    """A client's end of one open session."""

    def send(self, message: bytes) -> None: ...

    def recv(self, timeout: float | None = None) -> bytes:
        """The next message.

        Raises ``TimeoutError`` when none arrives in time, ``PeerDisconnected`` once the server ends the
        session, and ``ConnectRefused`` when the server refuses the session before its first message.
        """
        ...

    def close(self) -> str:
        """Close this end, and report what the wire saw, for the log.

        The report says whether the peer answered the close, in the wire's own terms. A server that still
        holds a session strands the next session's handshake, and only the wire can see that.
        """
        ...


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
    async def send(self, message: bytes) -> None: ...

    @abc.abstractmethod
    async def receive(self) -> bytes:
        """The next message. Raises ``PeerDisconnected`` once the client ends the session."""

    @abc.abstractmethod
    async def refuse(self, reason: str) -> None:
        """End a session the server cannot serve, and tell the client why."""


# What a wire hands the server for each session it accepts: the connection, and the model the route
# names, or ``None`` for the model the server pinned.
SessionHandler = Callable[[ServerConnection, str | None], Awaitable[None]]

# Whether the session headers carry a credential the server accepts. Header names are lower case.
Authorized = Callable[[Mapping[str, str]], bool]


class Wire(abc.ABC):
    """One transport that sessions arrive on.

    A wire reads its own route for the model a session names, and refuses an unauthorized peer before
    the session opens.
    """

    @property
    @abc.abstractmethod
    def endpoint(self) -> Endpoint:
        """Where this wire serves. The port is known once ``start`` returns."""

    @abc.abstractmethod
    async def start(self, session: SessionHandler, authorized: Authorized) -> None:
        """Bind, and give every accepted session to ``session``. Raises when the port is not free."""

    @abc.abstractmethod
    async def serve(self) -> None:
        """Carry sessions until ``stop``, or until the wire ends for its own reason."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """End the wire, and every session on it."""
