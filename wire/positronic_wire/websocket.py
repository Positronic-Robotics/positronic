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

    NAME = 'websocket'
    DEFAULT_PORT = 80
    # The URL schemes this wire writes: the session upgrades from HTTP, and the API answers on it.
    SCHEME = 'ws'
    API_SCHEME = 'http'

    def session_url(self, address: wire.SessionAddress) -> str:
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}://{self.netloc(address)}{address.path}{query}'

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'{self.API_SCHEME}://{self.netloc(address)}{wire.API_PATH}'

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

    def probe(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """A handshake on the host's root, which no server upgrades.

        The server refuses that upgrade with 403, and nothing else answers 403 there: an edge that refuses a
        credential answers 401. So 403 is the server, and every other status reads as ``dial`` reads it.
        """
        root = f'{self.SCHEME}://{self.netloc(address)}'
        try:
            connect(root, open_timeout=open_timeout, additional_headers=headers).close()
        except InvalidStatus as e:
            if e.response.status_code == HTTPStatus.FORBIDDEN:
                return None
            return _status_refusal(e.response.status_code)
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            return _refusal_of(e)
        return None


class WebsocketTlsClientWire(WebsocketClientWire):
    """The websocket wire over TLS: the same session behind an edge that terminates it."""

    NAME = 'websocket_tls'
    DEFAULT_PORT = 443
    SCHEME = 'wss'
    API_SCHEME = 'https'
