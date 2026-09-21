"""The client side of the websocket wire."""

import abc
import json
import os
import socket
import ssl
import stat
from collections.abc import Mapping
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from typing import ClassVar, Generic

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


class _WebsocketWire(wire.ClientWire[wire.AddressT], Generic[wire.AddressT]):
    """What every websocket member shares: one session per connection, and how a refused one reads."""

    # The URL scheme this wire writes for a session.
    SCHEME: ClassVar[str]

    def _refusal(self, raised: OSError | InvalidHandshake | ConnectionClosed, address: wire.AddressT) -> wire.Refusal:
        """What a handshake that did not open says about the server, in this wire's terms."""
        return _refusal_of(raised)

    @abc.abstractmethod
    def _connect(self, address: wire.AddressT, **settings) -> Connection:
        """One opened websocket on ``address``, however this wire reaches it."""

    @abc.abstractmethod
    def _api_connection(self, address: wire.AddressT, open_timeout: float) -> HTTPConnection:
        """An unopened connection to the server's HTTP API, however this wire reaches it."""

    def list_models(self, address: wire.AddressT, headers: Mapping[str, str] | None, open_timeout: float) -> list[str]:
        """The catalogue, read on the transport that carries this wire's sessions."""
        where = self.session_url(address)
        connection = self._api_connection(address, open_timeout)
        try:
            connection.request('GET', wire.MODELS_PATH, headers=dict(headers or {}))
            answer = connection.getresponse()
            status, body = answer.status, answer.read()
        except HTTPException as e:
            # The connection opened and the exchange did not finish: a backend that is not ready.
            raise wire.ConnectRefused(wire.Refusal.COLD, f'{e} (listing the models on {where})') from e
        except OSError as e:
            raise wire.ConnectRefused(self._refusal(e, address), f'{e} (listing the models on {where})') from e
        finally:
            connection.close()
        if status != HTTPStatus.OK:
            raise wire.ConnectRefused(_status_refusal(status), f'the catalogue on {where} answered {status}')
        return json.loads(body)[wire.MODELS_KEY]

    def dial(
        self, address: wire.AddressT, headers: Mapping[str, str] | None, open_timeout: float
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

    def probe(
        self, address: wire.AddressT, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """A handshake on the server's root, which no server upgrades.

        The server refuses that upgrade with 403, and nothing else answers 403 there: an edge that refuses a
        credential answers 401. So 403 is the server, and every other status reads as ``dial`` reads it.
        """
        root = address.at_root()
        try:
            self._connect(root, open_timeout=open_timeout, additional_headers=headers).close()
        except InvalidStatus as e:
            if e.response.status_code == HTTPStatus.FORBIDDEN:
                return None
            return _status_refusal(e.response.status_code)
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            return self._refusal(e, root)
        return None


class WebsocketClientWire(_WebsocketWire[wire.HostPortAddress]):
    """The client side of the websocket wire, which the server's HTTP port carries beside its API."""

    NAME = 'websocket'
    ADDRESS = wire.HostPortAddress
    DEFAULT_PORT = 80
    # The URL scheme this wire writes for a session; ``_api_connection`` says how the API is reached.
    SCHEME = 'ws'

    def netloc(self, address: wire.HostPortAddress) -> str:
        """``host:port``, less the port this wire defaults to."""
        return wire.netloc(address, self.DEFAULT_PORT)

    def handshake_url(self, address: wire.HostPortAddress) -> str:
        """The URL the upgrade asks for, which this wire also dials."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}://{self.netloc(address)}{address.path}{query}'

    def session_url(self, address: wire.HostPortAddress) -> str:
        return self.handshake_url(address)

    def _connect(self, address: wire.HostPortAddress, **settings) -> Connection:
        return connect(self.handshake_url(address), **settings)

    def _api_connection(self, address: wire.HostPortAddress, open_timeout: float) -> HTTPConnection:
        return HTTPConnection(address.host, address.port, timeout=open_timeout)


class WebsocketTlsClientWire(WebsocketClientWire):
    """The websocket wire over TLS: the same session behind an edge that terminates it."""

    NAME = 'websocket_tls'
    DEFAULT_PORT = 443
    SCHEME = 'wss'

    def _api_connection(self, address: wire.HostPortAddress, open_timeout: float) -> HTTPConnection:
        # No context named: the connection verifies the edge against the system's own roots.
        return HTTPSConnection(address.host, address.port, timeout=open_timeout)


class _UnixHTTPConnection(HTTPConnection):
    """An HTTP connection opened on a Unix socket rather than dialled on a host and a port.

    ``HTTPConnection`` takes a host to write the ``Host`` header with; it resolves and dials nothing
    here, because ``connect`` opens the socket itself.
    """

    def __init__(self, uds: Path, host: str, timeout: float):
        super().__init__(host, timeout=timeout)
        self._uds = uds

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(str(self._uds))
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class WebsocketUnixClientWire(_WebsocketWire[wire.UnixSocketAddress]):
    """The websocket wire over a Unix socket: the same session, reached on a path instead of a port.

    A socket is same-machine by construction, so there is no TLS member beside this one.
    """

    NAME = 'websocket_unix'
    ADDRESS = wire.UnixSocketAddress
    SCHEME = 'ws'
    # A socket names no authority, so the handshake and the API carry this in place of one. The server
    # reads the route and ignores it, and no name is resolved: the connection is already open.
    STANDS_FOR_THE_SERVER = 'localhost'

    def handshake_url(self, address: wire.UnixSocketAddress) -> str:
        """The URL the upgrade asks for, under the name that stands in for the socket."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}://{self.STANDS_FOR_THE_SERVER}{address.path}{query}'

    def session_url(self, address: wire.UnixSocketAddress) -> str:
        """The socket and the route on it, as this wire names one session."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}+unix://{address.uds}{address.path}{query}'

    def _api_connection(self, address: wire.UnixSocketAddress, open_timeout: float) -> HTTPConnection:
        """The catalogue answers on the session's own socket, beside the sessions."""
        return _UnixHTTPConnection(address.uds, self.STANDS_FOR_THE_SERVER, timeout=open_timeout)

    @staticmethod
    def _a_retry_can_reach_it(uds: Path, raised: OSError) -> bool:
        """Whether a failed dial can still come good, or is settled.

        A connection-level failure means the dial reached a live socket and the server behind it was
        not ready: it accepted and closed, reset the connection, broke the pipe, or never finished the
        handshake. A co-located server that is starting or restarting raises each of them in turn.

        A refusal is the exception, and reads the path: a path holding something that is not a socket
        is refused in the same words as a socket nobody listens on. An absent path can still become
        one. Every other ``OSError`` is this process's own — a descriptor limit, a refused permission —
        and no retry reaches it.
        """
        if isinstance(raised, ConnectionRefusedError | FileNotFoundError):
            try:
                return stat.S_ISSOCK(os.stat(uds).st_mode)
            except FileNotFoundError:
                return True
            except OSError:
                return False
        return isinstance(raised, ConnectionError | TimeoutError)

    def _refusal(
        self, raised: OSError | InvalidHandshake | ConnectionClosed, address: wire.UnixSocketAddress
    ) -> wire.Refusal:
        """A dial a retry can still reach is cold; one it cannot is final.

        The base wire reads any ``OSError`` as cold, which is right for a port a backend will answer on
        and wrong for a path: a path holding something that is not a socket, a refused permission and a
        spent descriptor limit never change, so retrying one spends the whole connect deadline on an
        answer that is already final.
        """
        if isinstance(raised, OSError) and not isinstance(raised, InvalidHandshake | ConnectionClosed):
            cold = self._a_retry_can_reach_it(address.uds, raised)
            return wire.Refusal.COLD if cold else wire.Refusal.FINAL
        return super()._refusal(raised, address)

    def _connect(self, address: wire.UnixSocketAddress, **settings) -> Connection:
        # The handshake asks for the route and the query under a name that stands in for the socket.
        return unix_connect(str(address.uds), uri=self.handshake_url(address), **settings)
