"""The client side of the websocket wire."""

import abc
import errno
import json
import os
import queue
import socket
import ssl
import stat
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any, ClassVar, Generic

from positronic_wire import wire
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidMessage, InvalidStatus
from websockets.proxy import get_proxy
from websockets.sync.client import connect, unix_connect
from websockets.sync.connection import Connection
from websockets.uri import parse_uri


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
# ``EAI_AGAIN``, which reads as no answer: a name can start resolving, where a misspelt one never does.
_NO_SUCH_HOST_ERRNOS = (socket.EAI_NONAME, socket.EAI_NODATA)


def refusal_of(raised: OSError | InvalidHandshake | ConnectionClosed) -> wire.Refusal:
    """What a handshake that did not open says about the server."""
    if isinstance(raised, InvalidStatus):
        return _status_refusal(raised.response.status_code)
    if isinstance(raised, ssl.SSLCertVerificationError):
        return wire.Refusal.FINAL
    if isinstance(raised, socket.gaierror) and raised.errno in _NO_SUCH_HOST_ERRNOS:
        return wire.Refusal.FINAL
    if isinstance(raised, InvalidHandshake) and not isinstance(raised, InvalidMessage):
        # An answer that is not a valid upgrade: a backend that is not ready.
        return wire.Refusal.COLD
    # A refused connect, a timed-out one, a reset TLS handshake, a connection that closed before an answer.
    return wire.Refusal.SILENT


def _seconds_until(deadline: float) -> float:
    """Raises ``TimeoutError`` once ``deadline`` has passed, as a socket that timed out does."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('the call spent its timeout')
    return left


class _WebsocketWire(wire.ClientWire[wire.AddressT], Generic[wire.AddressT]):
    """What every websocket member shares: one session per connection, and how a refused one reads."""

    # The URL scheme this wire writes for a session.
    SCHEME: ClassVar[str]

    def _refusal(self, raised: OSError | InvalidHandshake | ConnectionClosed, address: wire.AddressT) -> wire.Refusal:
        """What a handshake that did not open says about the server, in this wire's terms."""
        return refusal_of(raised)

    @abc.abstractmethod
    def _connect(self, address: wire.AddressT, **settings) -> Connection:
        """One opened websocket on ``address``, however this wire reaches it, or on the ``sock`` a caller opened."""

    @abc.abstractmethod
    def _open_socket(self, address: wire.AddressT, timeout: float) -> socket.socket:
        """A socket connected to the server on ``address``, before any TLS. ``timeout`` covers every step."""

    @abc.abstractmethod
    def _api_connection(self, address: wire.AddressT, deadline: float) -> HTTPConnection:
        """An unopened connection to the server's HTTP API, which opens its socket before ``deadline``."""

    def _reached_through_proxy(self, address: wire.AddressT) -> bool:
        """Whether ``websockets`` dials ``address`` through a proxy the environment names."""
        return False

    # The most of the keepalive answer's body the call reads: a JSON object of one key fits in it many times.
    _MAX_KEEPALIVE_BODY_BYTES = 16 * 1024

    @classmethod
    def _post_keepalive(
        cls, connection: HTTPConnection, headers: Mapping[str, str] | None, deadline: float
    ) -> tuple[int, bytes]:
        """The status of the keepalive call on ``connection`` before ``deadline``, and the body of a 200.

        A socket timeout bounds each read, so a peer that sends one byte at a time can hold a read past any
        deadline. A timer shuts the socket down at the deadline instead.
        """
        connection.connect()
        sock = connection.sock

        def shut_down() -> None:
            try:
                # The plain socket's call: `ssl.SSLSocket.shutdown` also drops its TLS state, which a read uses.
                socket.socket.shutdown(sock, socket.SHUT_RDWR)
            except OSError as e:
                # ENOTCONN: the peer ended the connection first, and with it the read.
                if e.errno != errno.ENOTCONN:
                    raise

        timer = threading.Timer(max(0.0, deadline - time.monotonic()), shut_down)
        timer.daemon = True
        timer.start()
        try:
            connection.request('POST', wire.KEEPALIVE_PATH, headers=dict(headers or {}))
            # Closed here: the answer holds the socket open past the connection's own close.
            with connection.getresponse() as answer:
                body = answer.read(cls._MAX_KEEPALIVE_BODY_BYTES) if answer.status == HTTPStatus.OK else b''
                return answer.status, body
        finally:
            timer.cancel()
            # `join` waits out a shutdown already running, so none runs after this returns.
            timer.join()

    @staticmethod
    def _alive_seconds(body: bytes, where: str) -> int | None:
        """The seconds the keepalive answer in ``body`` carries.

        Raises ``wire.ConnectRefused`` with ``FINAL`` where ``body`` is not that answer: something other than a
        server answered 200.
        """
        try:
            alive = json.loads(body)[wire.ALIVE_SECONDS]
        except (ValueError, LookupError, TypeError, RecursionError) as e:
            raise wire.ConnectRefused(wire.Refusal.FINAL, f'{where} answers 200 without the keepalive answer') from e
        if alive is None or (isinstance(alive, int) and not isinstance(alive, bool)):
            return alive
        raise wire.ConnectRefused(wire.Refusal.FINAL, f'{where} answers 200 with {wire.ALIVE_SECONDS} {alive!r}')

    def keepalive(self, address: wire.AddressT, headers: Mapping[str, str] | None, timeout: float) -> int | None:
        """``POST`` to ``wire.KEEPALIVE_PATH`` on the HTTP API beside the session route.

        ``timeout`` bounds the whole call: the name lookup, every address and every read. ``http.client`` caps
        the head of the answer, and the call reads at most ``_MAX_KEEPALIVE_BODY_BYTES`` of its body.
        """
        deadline = time.monotonic() + timeout
        where = f'{wire.KEEPALIVE_PATH} on {self.session_url(address)}'
        connection = self._api_connection(address, deadline)
        try:
            status, body = self._post_keepalive(connection, headers, deadline)
        except HTTPException as e:
            # The connection opened, and no answer the call can read came before the close or the deadline.
            raise wire.ConnectRefused(wire.Refusal.SILENT, f'{e} (calling {where})') from e
        except OSError as e:
            raise wire.ConnectRefused(self._refusal(e, address), f'{e} (calling {where})') from e
        finally:
            connection.close()
        if status == HTTPStatus.OK:
            return self._alive_seconds(body, where)
        if status == HTTPStatus.NOT_FOUND:
            # A server without the call answers 404, and so does an address that serves something else.
            refusal = self.probe(address, headers, max(0.0, deadline - time.monotonic()))
            if refusal is None:
                raise wire.KeepaliveUnsupported(f'{where} answers 404; this server serves sessions but not keepalive')
            raise wire.ConnectRefused(refusal, f'{where} answers 404, and no session server answers there')
        # An HTTP route refuses a credential with 401, where the upgrade beside it refuses with 403.
        refusal = wire.Refusal.FORBIDDEN if status == HTTPStatus.UNAUTHORIZED else _status_refusal(status)
        raise wire.ConnectRefused(refusal, f'{where} answers {status}')

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
                # Deflate costs ~100 ms of sender CPU on three raw 640x400 frames; compress_images makes them small.
                compression=None,
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
        ``open_timeout`` bounds the whole call, the name lookup included.
        """
        root = address.at_root()
        deadline = time.monotonic() + open_timeout
        try:
            # A proxy the environment names takes the connect, under the time left. websockets closes `sock` where
            # the handshake fails, and caps the answer it reads. The probe does not wait for the server's close.
            sock = None if self._reached_through_proxy(root) else self._open_socket(root, open_timeout)
            websocket = self._connect(
                root,
                sock=sock,
                open_timeout=max(0.0, deadline - time.monotonic()),
                additional_headers=headers,
                close_timeout=0,
            )
        except InvalidStatus as e:
            if e.response.status_code == HTTPStatus.FORBIDDEN:
                return None
            return _status_refusal(e.response.status_code)
        except (OSError, InvalidHandshake, ConnectionClosed) as e:
            return self._refusal(e, root)
        websocket.close()
        return None


def _addresses(host: str, port: int, deadline: float) -> list[tuple[Any, ...]]:
    """The addresses ``host`` resolves to, before ``deadline``.

    ``getaddrinfo`` takes no timeout. A lookup past the deadline ends in its own thread, which the resolver
    configuration bounds (``timeout`` x ``attempts`` per nameserver), and nothing reads its answer.
    """
    found: queue.SimpleQueue[list[tuple[Any, ...]] | OSError] = queue.SimpleQueue()

    def look_up() -> None:
        try:
            found.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as failed:
            found.put(failed)

    threading.Thread(target=look_up, name='wire-lookup', daemon=True).start()
    try:
        addresses = found.get(timeout=_seconds_until(deadline))
    except queue.Empty:
        raise TimeoutError(f'{host} did not resolve in time') from None
    if isinstance(addresses, OSError):
        raise addresses
    return addresses


def connected_socket(host: str, port: int, timeout: float) -> socket.socket:
    """A TCP socket connected to ``host:port``. ``timeout`` covers the name lookup and every address it tries.

    FOOTGUN: ``socket.create_connection`` arms its timeout once per address, and gives the lookup none.
    """
    deadline = time.monotonic() + timeout
    failure = OSError(f'{host} resolves to no address')
    for family, kind, proto, _name, address in _addresses(host, port, deadline):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(_seconds_until(deadline))
            sock.connect(address)
        except OSError as e:
            sock.close()
            failure = e
        else:
            return sock
    raise failure


class _HTTPConnectionOn(HTTPConnection):
    """An HTTP connection on the socket ``open_socket`` opens with the seconds left before ``deadline``.

    ``netloc`` names the server in the ``Host`` header, and nothing resolves it.
    """

    def __init__(self, netloc: str, open_socket: Callable[[float], socket.socket], deadline: float):
        super().__init__(netloc)
        self._open_socket = open_socket
        self._deadline = deadline

    def connect(self) -> None:
        self.sock = self._open_socket(_seconds_until(self._deadline))


class WebsocketClientWire(_WebsocketWire[wire.HostPortAddress]):
    """The client side of the websocket wire, which the server's HTTP port carries beside its API."""

    NAME = 'websocket'
    ADDRESS = wire.HostPortAddress
    DEFAULT_PORT = 80
    # The URL scheme this wire writes for a session; ``_api_connection`` says how the API is reached.
    SCHEME = 'ws'

    def handshake_url(self, address: wire.HostPortAddress) -> str:
        """The URL the upgrade asks for, which this wire also dials."""
        query = f'?{address.query}' if address.query else ''
        return f'{self.SCHEME}://{wire.netloc(address, self.DEFAULT_PORT)}{address.path}{query}'

    def session_url(self, address: wire.HostPortAddress) -> str:
        return self.handshake_url(address)

    def _connect(self, address: wire.HostPortAddress, **settings) -> Connection:
        return connect(self.handshake_url(address), **settings)

    def _open_socket(self, address: wire.HostPortAddress, timeout: float) -> socket.socket:
        return connected_socket(address.host, address.port, timeout)

    def _reached_through_proxy(self, address: wire.HostPortAddress) -> bool:
        return get_proxy(parse_uri(self.handshake_url(address))) is not None

    def _api_connection(self, address: wire.HostPortAddress, deadline: float) -> HTTPConnection:
        return _HTTPConnectionOn(wire.netloc(address, self.DEFAULT_PORT), partial(self._open_socket, address), deadline)


class WebsocketTlsClientWire(WebsocketClientWire):
    """The websocket wire over TLS: the same session behind an edge that terminates it."""

    NAME = 'websocket_tls'
    DEFAULT_PORT = 443
    SCHEME = 'wss'

    def _api_connection(self, address: wire.HostPortAddress, deadline: float) -> HTTPConnection:
        return _HTTPConnectionOn(wire.netloc(address, self.DEFAULT_PORT), partial(self._open_tls, address), deadline)

    def _open_tls(self, address: wire.HostPortAddress, timeout: float) -> socket.socket:
        """A TLS socket to the edge on ``address``. ``timeout`` covers the connect and the TLS handshake."""
        deadline = time.monotonic() + timeout
        sock = self._open_socket(address, timeout)
        try:
            sock.settimeout(_seconds_until(deadline))
            # No context named: the socket verifies the edge against the system's own roots.
            return ssl.create_default_context().wrap_socket(sock, server_hostname=address.host)
        except BaseException:
            sock.close()
            raise


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

    def _open_socket(self, address: wire.UnixSocketAddress, timeout: float) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            sock.connect(str(address.uds))
        except BaseException:
            sock.close()
            raise
        return sock

    def _api_connection(self, address: wire.UnixSocketAddress, deadline: float) -> HTTPConnection:
        """The HTTP API answers on the session's own socket, beside the sessions."""
        return _HTTPConnectionOn(self.STANDS_FOR_THE_SERVER, partial(self._open_socket, address), deadline)

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
        """A dial a retry can still reach is silent; one it cannot is final.

        The base wire reads any ``OSError`` as silent, which is right for a port a backend will answer on
        and wrong for a path: a path holding something that is not a socket, a refused permission and a
        spent descriptor limit never change, so retrying one spends the whole connect deadline on an
        answer that is already final.
        """
        if isinstance(raised, OSError) and not isinstance(raised, InvalidHandshake | ConnectionClosed):
            reachable = self._a_retry_can_reach_it(address.uds, raised)
            return wire.Refusal.SILENT if reachable else wire.Refusal.FINAL
        return super()._refusal(raised, address)

    def _connect(self, address: wire.UnixSocketAddress, **settings) -> Connection:
        # The handshake asks for the route and the query under a name that stands in for the socket. A socket the
        # caller opened takes the place of the path.
        path = str(address.uds) if settings.get('sock') is None else None
        return unix_connect(path, uri=self.handshake_url(address), **settings)
