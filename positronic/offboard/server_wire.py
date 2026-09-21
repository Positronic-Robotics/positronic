"""The server side of the transports a session arrives on: one wire per transport, one connection per session.

The client side, and the facts both ends share, are ``positronic_wire``.
"""

import abc
import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from starlette.datastructures import QueryParams

from . import keys


class ServedAddress(abc.ABC):
    """Where a wire serves, and how a session served there names that in its own metadata.

    A wire that binds a host and a port and one that binds a socket answer with different keys, so each
    writes its own; nothing reads one kind of address and decides.
    """

    @property
    @abc.abstractmethod
    def meta(self) -> dict[str, Any]:
        """The metadata keys this address writes into every session served on it."""


@dataclasses.dataclass(frozen=True)
class ServedHostPort(ServedAddress):
    """A wire serving on a host and a port. The port is known once the wire has bound."""

    host: str
    port: int

    @property
    def meta(self) -> dict[str, Any]:
        return {keys.HOST: self.host, keys.PORT: self.port}


class ServerConnection(abc.ABC):
    """A server's end of one open session."""

    @property
    @abc.abstractmethod
    def peer(self) -> str:
        """Whom this session serves, for the log."""

    @property
    @abc.abstractmethod
    def served_address(self) -> ServedAddress:
        """Where the wire that accepted this session serves."""

    @property
    @abc.abstractmethod
    def query_params(self) -> QueryParams:
        """The session params the client asked for."""

    @abc.abstractmethod
    async def send(self, message: bytes) -> None:
        """Raises ``positronic_wire.wire.PeerDisconnected`` once the client ends the session."""

    @abc.abstractmethod
    async def receive(self) -> bytes:
        """The next message. Raises ``positronic_wire.wire.PeerDisconnected`` once the client ends the session."""

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
    def served_address(self) -> ServedAddress:
        """Where this wire serves. A bound port is known once ``start`` returns."""

    @abc.abstractmethod
    async def start(self, session: SessionHandler, authorized: Authorized) -> None:
        """Bind, and give every accepted session to ``session``. Raises when the port is not free."""

    @abc.abstractmethod
    async def serve(self) -> None:
        """Carry sessions until ``stop``, or until the wire ends for its own reason."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """End the wire, and every session on it."""
