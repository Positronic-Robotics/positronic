"""The websocket wire, and the two ends of a websocket session."""

import errno
import os
import socket
import ssl
import stat
from collections.abc import Mapping
from http import HTTPStatus
from urllib.parse import quote

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from starlette.datastructures import QueryParams
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect, unix_connect
from websockets.sync.connection import Connection

from . import wire


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


def _socket_may_still_appear(uds: str, e: OSError) -> bool:
    """Whether a failed dial is a co-located server that has not bound its socket yet.

    Only an absent path and a refusal can mean that; every other ``OSError`` is settled, and waiting for
    it spends the whole deadline on an answer that will not change. A refusal then reads the path, which
    tells a restarting server from a path naming something that is not a socket.
    """
    if not isinstance(e, (FileNotFoundError, ConnectionRefusedError)):
        return False
    try:
        return stat.S_ISSOCK(os.stat(uds).st_mode)
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _status_refusal(status_code: int) -> wire.Refusal:
    """What a non-101 answer to the upgrade says about the server."""
    if status_code == HTTPStatus.FORBIDDEN:
        return wire.Refusal.FORBIDDEN
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR or status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return wire.Refusal.COLD
    return wire.Refusal.FINAL


class WebsocketClientWire(wire.ClientWire):
    """The client side of the websocket wire, which the server's HTTP port carries beside its API."""

    SCHEME = 'ws'
    SECURE_SCHEME = 'wss'
    # A bare host names this wire, and so does an http(s) URL: the session upgrades from HTTP.
    ALIASES = (
        wire.Scheme('', secure=False),
        wire.Scheme('http', secure=False),
        wire.Scheme('https', secure=True),
        # A Unix socket path in place of a host. The session upgrades from HTTP over that socket.
        wire.Scheme(wire.UNIX_SCHEME, secure=False),
    )

    def session_url(self, address: wire.SessionAddress) -> str:
        if address.uds is None:
            return super().session_url(address)
        # ``dial`` reads ``uds``, so this reads it too: a URL that named a socket must name one back.
        # Spell it as the URL wrote it, since a decoded ``?`` or ``#`` reads as a delimiter and names
        # another socket; an address built in code carries no spelling, so the dialled path is it.
        query = f'?{address.query}' if address.query else ''
        return f'{wire.UNIX_SCHEME}://{address.uds_as_written or quote(address.uds)}{address.path}{query}'

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'{"https" if address.secure else "http"}://{address.netloc}{wire.API_PATH}'

    def dial(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> WebsocketClientConnection:
        """A client's end of one session on ``address``. Raises ``wire.ConnectRefused`` when it does not open."""
        # A proxy closes a connection it has read nothing from, often after 60 s, and one inference sends
        # nothing until it answers. The pings keep it open.
        settings = {
            'open_timeout': open_timeout,
            'additional_headers': headers,
            'ping_interval': 20.0,
            'max_size': wire.MAX_MESSAGE_BYTES,
        }
        try:
            if address.uds is None:
                websocket = connect(self.session_url(address), **settings)
            else:
                # The handshake asks for the path and the query under a host that stands in for the socket.
                websocket = unix_connect(address.uds, uri=address.url(self.SCHEME), **settings)
        except InvalidStatus as e:
            raise wire.ConnectRefused(_status_refusal(e.response.status_code), str(e)) from e
        except ssl.SSLCertVerificationError as e:
            raise wire.ConnectRefused(wire.Refusal.FINAL, str(e)) from e
        # A timed-out connect, a reset TLS handshake, a refused upgrade, a dropped handshake: a backend that is
        # not ready.
        except (TimeoutError, ssl.SSLError, ConnectionClosed, InvalidHandshake) as e:
            raise wire.ConnectRefused(wire.Refusal.COLD, str(e)) from e
        except OSError as e:
            # A socket a co-located server has not bound yet is a backend that is not ready.
            if address.uds is not None and _socket_may_still_appear(address.uds, e):
                raise wire.ConnectRefused(wire.Refusal.COLD, str(e)) from e
            raise
        return WebsocketClientConnection(websocket)


class WebsocketServerConnection(wire.ServerConnection):
    """A server's end of one websocket session, over an accepted ``WebSocket``."""

    def __init__(self, websocket: WebSocket, endpoint: wire.Endpoint):
        self._websocket = websocket
        self._endpoint = endpoint

    @property
    def peer(self) -> str:
        return str(self._websocket.client)

    @property
    def endpoint(self) -> wire.Endpoint:
        return self._endpoint

    @property
    def query_params(self) -> QueryParams:
        return self._websocket.query_params

    async def send(self, message: bytes) -> None:
        try:
            await self._websocket.send_bytes(message)
        except WebSocketDisconnect as e:
            raise wire.PeerDisconnected(str(e)) from e

    async def receive(self) -> bytes:
        try:
            return await self._websocket.receive_bytes()
        except WebSocketDisconnect as e:
            raise wire.PeerDisconnected(str(e)) from e

    async def refuse(self, reason: str) -> None:
        await self._websocket.close(code=1008, reason=reason[:100])


def _listening_sockets(host: str, port: int) -> list[socket.socket]:
    """A listening socket for every address ``host`` resolves to, all on one port.

    A ``port`` of 0 takes the port the first socket bound, so the whole set still answers on one port.
    """
    sockets: list[socket.socket] = []
    bound_port = port
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
        # A name can resolve to one address more than once, and a second bind on it fails the whole set.
        for family, kind, proto, address in dict.fromkeys((f, k, pr, a) for f, k, pr, _c, a in resolved):
            sock = socket.socket(family, kind, proto)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family is socket.AF_INET6:
                # Each family gets its own socket here, so the IPv6 one must leave the IPv4 address alone.
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((address[0], bound_port, *address[2:]) if bound_port else address)
            # The port answers from the moment ``start`` returns: the kernel queues a connect that arrives
            # before the serving loop runs.
            sock.listen()
            bound_port = sock.getsockname()[1]
    except OSError:
        for sock in sockets:
            sock.close()
        raise
    return sockets


# The probe bounds its wait, and reads a wait that runs out as a live server: a server whose backlog is
# full holds a connect open, and an unbounded one would stall startup.
LIVE_SOCKET_PROBE_SEC = 1.0


def _is_stale_socket(path: str) -> bool:
    """Whether ``path`` is a socket no server answers on, so replacing it takes nothing from anybody.

    A live socket, a probe that runs out of time against a full backlog, and a path that holds something
    other than a socket are none of them stale.
    """
    try:
        if not stat.S_ISSOCK(os.stat(path).st_mode):
            return False
    except FileNotFoundError:
        return False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(LIVE_SOCKET_PROBE_SEC)
        try:
            probe.connect(path)
        except ConnectionRefusedError:
            return True
        except OSError:
            return False
    return False


def claim_socket_path(path: str) -> socket.socket:
    """Bind and listen on ``path``, and return the socket, or refuse a path something already holds.

    The bind is the claim, so two servers starting together cannot both take one path: the loser's bind
    fails. A probe follows it only to tell a stale file from a live server. Serve the returned socket by
    its descriptor: a server handed the path instead binds again, and unlinks this claim.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            sock.bind(path)
        except OSError as taken:
            if taken.errno != errno.EADDRINUSE:
                raise
            if not _is_stale_socket(path):
                raise OSError(errno.EADDRINUSE, f'{path!r} is already in use') from None
            os.unlink(path)
            sock.bind(path)
        # The mode is the deployment's, through its umask: widening it here would open the socket to
        # every local account that can reach the directory.
        sock.listen()
    except BaseException:
        sock.close()
        raise
    return sock


# uvicorn's default ('websockets') reassembles an 846 KiB observation in 58 ms, against 29 ms here
# (measured by positronic/offboard/serving_cost.py).
WS_IMPL = 'websockets-sansio'


class WebsocketWire(wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and ``api`` answers on the same port.

    ``uds`` binds a Unix socket path in place of ``host:port``, which serves a client on the same machine
    over no network. A client reaches it with a ``unix://`` URL. The socket file stays after ``stop``: a
    successor reads it as stale, where an unlink here could take a path that successor has claimed.
    """

    # How long ``stop`` lets an open session finish before it cuts the connection. The uvicorn default
    # waits for ever, and a session mid-inference holds the whole server open.
    STOP_GRACE_SEC = 2

    def __init__(self, host: str, port: int, api: APIRouter, uds: str | None = None):
        self._host = host
        self._port = port
        self._uds = uds
        self._api = api
        self._sockets: list[socket.socket] = []
        self._server: uvicorn.Server | None = None
        self._endpoint: wire.Endpoint | None = None
        self._served = False

    @property
    def endpoint(self) -> wire.Endpoint:
        assert self._endpoint is not None, 'The websocket wire has not started'
        return self._endpoint

    async def start(self, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        self._served = False
        if self._uds is not None:
            self._sockets = [claim_socket_path(self._uds)]
            self._endpoint = wire.Endpoint(self._host, 0, uds=self._uds)
        else:
            self._sockets = _listening_sockets(self._host, self._port)
            self._endpoint = wire.Endpoint(self._host, self._sockets[0].getsockname()[1])
        app = FastAPI()
        app.include_router(self._api)
        self._route_sessions(app, session, authorized)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._endpoint.port,
            log_level='info',
            ws=WS_IMPL,
            ws_max_size=wire.MAX_MESSAGE_BYTES,
            timeout_graceful_shutdown=self.STOP_GRACE_SEC,
        )
        self._server = uvicorn.Server(config)

    def _route_sessions(self, app: FastAPI, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        async def require_auth(websocket: WebSocket) -> None:
            """Refuse before ``accept()``. An unauthorized peer never reaches the session handshake."""
            if not authorized(websocket.headers):
                raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

        async def serve_pinned_model(websocket: WebSocket) -> None:
            """Serve the model the server pinned. The path names a model; every query param is a pipeline override."""
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.endpoint), None)

        async def serve_named_model(websocket: WebSocket, model_id: str) -> None:
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.endpoint), model_id)

        auth = [Depends(require_auth)]
        app.websocket(wire.SESSION_PATH, dependencies=auth)(serve_pinned_model)
        # ``:path``: a model id can itself be a path (a HuggingFace repo), and opens under the name the
        # catalogue advertises.
        app.websocket(f'{wire.SESSION_PATH}/{{model_id:path}}', dependencies=auth)(serve_named_model)

    async def serve(self) -> None:
        assert self._server is not None and self._sockets, 'The websocket wire has not started'
        self._served = True
        await self._server.serve(sockets=self._sockets)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        # uvicorn releases the sockets when it shuts down. A wire that bound but never served has no
        # uvicorn to release them.
        if not self._served:
            for sock in self._sockets:
                sock.close()
