"""The client side of the websocket wire."""

import socket
import ssl
from collections.abc import Mapping
from http import HTTPStatus

from positronic_wire import wire
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection


class WebsocketClientConnection(wire.ClientConnection):
    """A client's end of one websocket session."""

    def __init__(self, websocket: Connection):
        self._websocket = websocket

    def send(self, message: bytes) -> None:
        try:
            self._websocket.send(message)
        except ConnectionClosed as e:
            raise wire.PeerDisconnected(str(e)) from e

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            message = self._websocket.recv(timeout=timeout)
        except ConnectionClosed as e:
            raise wire.PeerDisconnected(str(e)) from e
        assert isinstance(message, bytes), f'A frame is bytes, and this one is {type(message).__name__}'
        return message

    def close(self) -> str:
        state_before_close = self._websocket.state.name
        self._websocket.close()
        # A close that times out still reaches CLOSED locally; only the close code says the server answered.
        return f'state {state_before_close} -> {self._websocket.state.name}, close code {self._websocket.close_code}'


def _status_refusal(status_code: int) -> wire.Refusal:
    """What a non-101 answer to the upgrade says about the server."""
    if status_code == HTTPStatus.FORBIDDEN:
        return wire.Refusal.FORBIDDEN
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR or status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return wire.Refusal.COLD
    return wire.Refusal.FINAL


# The resolver's authoritative answers that the host has no address. A resolver that timed out answers
# ``EAI_AGAIN`` and stays cold: a name can start resolving, where a misspelt one never does.
_NO_SUCH_HOST_ERRNOS = (socket.EAI_NONAME, socket.EAI_NODATA)


def _refusal_of(raised: OSError | InvalidHandshake | ConnectionClosed) -> wire.Refusal:
    """What a handshake that did not open says about the server."""
    if isinstance(raised, InvalidStatus):
        return _status_refusal(raised.response.status_code)
    if isinstance(raised, ssl.SSLCertVerificationError):
        return wire.Refusal.FINAL
    if isinstance(raised, socket.gaierror) and raised.errno in _NO_SUCH_HOST_ERRNOS:
        return wire.Refusal.FINAL
    # A refused connect, a timed-out one, a reset TLS handshake, a refused upgrade, a dropped handshake: a
    # backend that is not ready.
    return wire.Refusal.COLD


class WebsocketClientWire(wire.ClientWire):
    """The client side of the websocket wire, which the server's HTTP port carries beside its API."""

    SCHEME = 'ws'
    SECURE_SCHEME = 'wss'
    # A bare host names this wire, and so does an http(s) URL: the session upgrades from HTTP.
    ALIASES = (wire.Scheme('', secure=False), wire.Scheme('http', secure=False), wire.Scheme('https', secure=True))

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'{"https" if address.secure else "http"}://{address.netloc}{wire.API_PATH}'

    def dial(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> WebsocketClientConnection:
        """A client's end of one session on ``address``. Raises ``wire.ConnectRefused`` when it does not open."""
        url = self.session_url(address)
        try:
            # A proxy closes a connection it has read nothing from, often after 60 s, and one inference sends
            # nothing until it answers. The pings keep it open.
            websocket = connect(
                url,
                open_timeout=open_timeout,
                additional_headers=headers,
                ping_interval=20.0,
                max_size=wire.MAX_MESSAGE_BYTES,
            )
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            raise wire.ConnectRefused(_refusal_of(e), f'{e} (connecting to {url})') from e
        return WebsocketClientConnection(websocket)

    def probe(self, address: wire.SessionAddress, open_timeout: float) -> wire.Refusal | None:
        """A handshake on the host's root, which no server upgrades: a status of any kind is an answer."""
        root = address._replace(path='', query='')
        try:
            connect(self.session_url(root), open_timeout=open_timeout).close()
        except InvalidStatus as e:
            # Only a gateway status names an edge with nothing behind it; every other one is the server.
            refusal = _status_refusal(e.response.status_code)
            return refusal if refusal is wire.Refusal.COLD else None
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            return _refusal_of(e)
        return None
