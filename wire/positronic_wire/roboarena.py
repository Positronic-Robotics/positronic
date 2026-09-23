"""The client side of the roboarena wire: each frame carried as bytes, on a websocket at the root."""

import dataclasses
from collections.abc import Mapping
from typing import Self

from positronic_wire import wire
from positronic_wire.websocket import WebsocketClientConnection, refusal_of
from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import connect


@dataclasses.dataclass(frozen=True)
class RoboarenaAddress(wire.SessionAddress):
    """A session on a roboarena server, named by the host and the port the partner published."""

    host: str
    port: int
    # Bare assignments, so neither is a field: the protocol routes on a key inside each frame, and the
    # server closes any path but the root with `1008 Unsupported WebSocket path`.
    path = ''
    query = ''

    @classmethod
    def from_url(cls, url: str) -> 'RoboarenaAddress':
        """The host and the port ``scheme://<host>:<port>`` names.

        Raises ``ValueError`` where ``url`` names no port, because a server publishes no default one. Raises
        where it names a path or a query too, because the wire dials the root alone.
        """
        split = wire.split_url(url)
        port = split.port
        if not split.hostname or port is None:
            raise ValueError(f'roboarena address {url!r} names no host and port; write <scheme>://<host>:<port>')
        dropped = [
            name
            for name, present in (('a path', split.path not in ('', '/')), ('a query', bool(split.query)))
            if present
        ]
        if dropped:
            raise ValueError(
                f'roboarena address {url!r} names {", ".join(dropped)}, and this wire dials the root alone; '
                'write <scheme>://<host>:<port>'
            )
        return cls(split.hostname, port)

    def at_root(self) -> Self:
        return self


class TextAnswer(Exception):
    """The server answered in text. It serves nothing more on that connection, and a retry does not change the text."""

    def __init__(self, text: str):
        super().__init__(f'the server answered this text: {text}')
        self.text = text


class RoboarenaClientConnection(WebsocketClientConnection):
    """A client's end of one roboarena connection: a websocket session whose peer may answer in text."""

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            message = self._websocket.recv(timeout=timeout)
        except ConnectionClosed as e:
            raise wire.PeerDisconnected(str(e)) from e
        if isinstance(message, str):
            raise TextAnswer(message)
        return message


class RoboarenaClientWire(wire.ClientWire[RoboarenaAddress]):
    """The client side of the roboarena wire, whose root carries frames alone."""

    NAME = 'roboarena'
    ADDRESS = RoboarenaAddress
    TAKES_EDGE_HEADERS = False
    # A server holds one connection open across a run and sends nothing between inferences, so a shorter
    # pong deadline drops a quiet connection.
    PING_INTERVAL_S = 60.0
    PING_TIMEOUT_S = 600.0

    def session_url(self, address: RoboarenaAddress) -> str:
        """The root this wire dials; a roboarena session names no route under it."""
        return f'ws://{wire.bracket_ipv6(address.host)}:{address.port}'

    def address_of(self, url: str) -> RoboarenaAddress:
        return RoboarenaAddress.from_url(url)

    def list_models(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> list[str]:
        """Raises: this wire has no catalogue route to read."""
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
                ping_interval=self.PING_INTERVAL_S,
                ping_timeout=self.PING_TIMEOUT_S,
                max_size=wire.MAX_MESSAGE_BYTES,
            )
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            raise wire.ConnectRefused(refusal_of(e), f'{e} (connecting to {url})') from e
        return RoboarenaClientConnection(connection)

    def dial(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> RoboarenaClientConnection:
        """A client's end of one session on ``address``.

        The server announces its configuration as the first frame, and ``dial`` leaves it unread.
        """
        return self._open(address, headers, open_timeout)

    def probe(
        self, address: RoboarenaAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """Whether a server announces itself at ``address``. Raises ``TextAnswer`` when it answers in text."""
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
