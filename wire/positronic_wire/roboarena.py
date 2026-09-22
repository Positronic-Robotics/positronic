"""The client side of the roboarena wire.

Roboarena is a cross-vendor protocol a partner serves: msgpack frames on a websocket at the bare root of a
port the partner names. The server announces its configuration as the first frame of every connection, it
serves one model, and it carries no HTTP API beside the frames. This wire carries the frames as bytes, and
the caller's codec reads them.
"""

import dataclasses
from collections.abc import Mapping
from typing import Self

from positronic_wire import wire
from positronic_wire.websocket import refusal_of
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import connect
from websockets.sync.connection import Connection

# How often the client pings an idle connection, and how long it waits for the pong. A server holds one
# connection open across a whole run and sends nothing between inferences, so a shorter pong deadline drops
# a connection that is merely quiet. FOOTGUN: this bounds a missing pong. A server that answers pings while
# its policy wedges holds the caller until the caller's own `recv` deadline passes.
PING_INTERVAL_S = 60.0
PING_TIMEOUT_S = 600.0


@dataclasses.dataclass(frozen=True)
class RoboarenaAddress(wire.SessionAddress):
    """A session on a roboarena server, named by the host and the port the partner published.

    The port has no default. A roboarena server publishes none, so a guess dials a machine nobody named.
    """

    host: str
    port: int
    # Bare assignments, so neither is a field: the protocol routes on a key inside each frame, and the
    # server closes any path but the root with `1008 Unsupported WebSocket path`.
    path = ''
    query = ''

    def at_root(self) -> Self:
        return self


class RoboarenaClientConnection(wire.ClientConnection):
    """A client's end of one roboarena connection."""

    def __init__(self, connection: Connection):
        self._connection = connection

    def send(self, message: bytes) -> None:
        try:
            self._connection.send(message)
        except ConnectionClosed as e:
            raise wire.PeerDisconnected(str(e)) from e

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            message = self._connection.recv(timeout=timeout)
        except ConnectionClosed as e:
            raise wire.PeerDisconnected(str(e)) from e
        if isinstance(message, str):
            # The server reports a failure as a text frame, and serves nothing more on that connection.
            raise wire.PeerDisconnected(f'the server answered this error text: {message}')
        return message

    def close(self) -> str:
        state_before_close = self._connection.state.name
        self._connection.close()
        # A close that times out still reaches CLOSED locally; only the close code says the server answered.
        return f'state {state_before_close} -> {self._connection.state.name}, close code {self._connection.close_code}'


class RoboarenaClientWire(wire.ClientWire[RoboarenaAddress]):
    """The client side of the roboarena wire, whose root carries frames alone.

    The protocol names no URL scheme and a partner publishes a plain port, so no TLS member sits beside
    this one.
    """

    NAME = 'roboarena'
    ADDRESS = RoboarenaAddress

    def session_url(self, address: RoboarenaAddress) -> str:
        """The root this wire dials. A roboarena session names no route, so the server is the whole of it."""
        return f'ws://{wire.bracket_ipv6(address.host)}:{address.port}'

    def list_models(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> list[str]:
        """Raises: a roboarena server serves one model and no catalogue route to read it on."""
        raise ValueError(
            f'{self.NAME} serves one model and no catalogue; the model a partner serves is the endpoint '
            f'itself, and the server announces its configuration on connect'
        )

    def _open(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> RoboarenaClientConnection:
        """One opened connection on ``address``. Raises ``wire.ConnectRefused`` when it does not open."""
        url = self.session_url(address)
        try:
            connection = connect(
                url,
                open_timeout=open_timeout,
                additional_headers=headers,
                compression=None,
                ping_interval=PING_INTERVAL_S,
                ping_timeout=PING_TIMEOUT_S,
                max_size=wire.MAX_MESSAGE_BYTES,
            )
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            raise wire.ConnectRefused(refusal_of(e), f'{e} (connecting to {url})') from e
        return RoboarenaClientConnection(connection)

    def dial(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> RoboarenaClientConnection:
        """A client's end of one session on ``address``.

        The server announces its configuration as the first frame, so the caller's first ``recv`` reads it.
        """
        return self._open(address, headers, open_timeout)

    def probe(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """Whether a server announces itself at ``address``.

        The protocol serves no route a probe can ask for, so the announcement is the whole of what says a
        server is up. A port that accepts a connection and announces nothing is a backend still starting.
        """
        try:
            connection = self._open(address, headers, open_timeout)
        except wire.ConnectRefused as e:
            return e.refusal
        try:
            connection.recv(timeout=open_timeout)
        except (TimeoutError, wire.PeerDisconnected):
            return wire.Refusal.COLD
        finally:
            connection.close()
        return None
