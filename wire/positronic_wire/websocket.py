"""The client side of the websocket wire."""

import os
import socket
import ssl
import stat
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path

from positronic_wire import wire
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect, unix_connect
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

    def _refusal(
        self, raised: OSError | InvalidHandshake | ConnectionClosed, address: wire.SessionAddress
    ) -> wire.Refusal:
        """What a handshake that did not open says about the server, in this wire's terms."""
        return _refusal_of(raised)

    def handshake_url(self, address: wire.SessionAddress) -> str:
        """The URL the upgrade asks for. It is what this wire dials, unless the wire dials a socket."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}://{self.netloc(address)}{address.path}{query}'

    def session_url(self, address: wire.SessionAddress) -> str:
        return self.handshake_url(address)

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
            websocket = self._connect(
                address,
                open_timeout=open_timeout,
                additional_headers=headers,
                ping_interval=20.0,
                max_size=wire.MAX_MESSAGE_BYTES,
            )
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            raise wire.ConnectRefused(self._refusal(e, address), f'{e} (connecting to {url})') from e
        return WebsocketClientConnection(websocket)

    def _connect(self, address: wire.SessionAddress, **settings) -> Connection:
        """One opened websocket on ``address``, however this wire reaches it."""
        return connect(self.handshake_url(address), **settings)

    def probe(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """A handshake on the host's root, which no server upgrades.

        The server refuses that upgrade with 403, and nothing else answers 403 there: an edge that refuses a
        credential answers 401. So 403 is the server, and every other status reads as ``dial`` reads it.
        """
        root = address._replace(path='', query='')
        try:
            self._connect(root, open_timeout=open_timeout, additional_headers=headers).close()
        except InvalidStatus as e:
            if e.response.status_code == HTTPStatus.FORBIDDEN:
                return None
            return _status_refusal(e.response.status_code)
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            return self._refusal(e, root)
        return None


class WebsocketTlsClientWire(WebsocketClientWire):
    """The websocket wire over TLS: the same session behind an edge that terminates it."""

    NAME = 'websocket_tls'
    DEFAULT_PORT = 443
    SCHEME = 'wss'
    API_SCHEME = 'https'


def _socket_may_still_appear(uds: Path, raised: OSError) -> bool:
    """Whether a failed dial is a co-located server that has not bound its socket yet.

    Only an absent path and a refusal can mean that; every other ``OSError`` is settled, and waiting for
    it spends the whole deadline on an answer that will not change. A refusal then reads the path, which
    tells a restarting server from a path naming something that is not a socket.
    """
    if not isinstance(raised, FileNotFoundError | ConnectionRefusedError):
        return False
    try:
        return stat.S_ISSOCK(os.stat(uds).st_mode)
    except FileNotFoundError:
        return True
    except OSError:
        return False


class WebsocketUnixClientWire(WebsocketClientWire):
    """The websocket wire over a Unix socket: the same session, reached on a path instead of a port.

    ``SessionAddress.uds`` is the socket, and ``host`` stands for the server in the handshake this wire
    sends over it. A socket is same-machine by construction, so there is no TLS member beside this one.
    """

    NAME = 'websocket_unix'

    def netloc(self, address: wire.SessionAddress) -> str:
        """The host alone: a socket has no port, so the handshake must not claim one."""
        return wire.bracket_ipv6(address.host)

    def session_url(self, address: wire.SessionAddress) -> str:
        """The socket and the route on it, for the log. Nothing reads this back."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}+unix://{address.uds}{address.path}{query}'

    def _refusal(
        self, raised: OSError | InvalidHandshake | ConnectionClosed, address: wire.SessionAddress
    ) -> wire.Refusal:
        """A socket a co-located server has not bound yet is cold; every other ``OSError`` is settled.

        The base wire reads any ``OSError`` as cold, which is right for a port a backend will answer on and
        wrong for a path: a misspelt one, a path that is not a socket and a refused permission never change,
        so retrying one spends the whole connect deadline on an answer that is already final.
        """
        if isinstance(raised, OSError) and not isinstance(raised, InvalidHandshake | ConnectionClosed):
            return wire.Refusal.COLD if _socket_may_still_appear(self._socket(address), raised) else wire.Refusal.FINAL
        return super()._refusal(raised, address)

    @classmethod
    def _socket(cls, address: wire.SessionAddress) -> Path:
        if address.uds is None:
            raise ValueError(f'{cls.NAME} dials a Unix socket; SessionAddress.uds names none')
        return address.uds

    def _connect(self, address: wire.SessionAddress, **settings) -> Connection:
        # The handshake asks for the route and the query under a host that stands in for the socket.
        return unix_connect(str(self._socket(address)), uri=self.handshake_url(address), **settings)
