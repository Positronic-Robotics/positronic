"""The server side of the websocket wire."""

import dataclasses
import errno
import os
import socket
import stat
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from positronic_wire import wire
from starlette.datastructures import QueryParams
from starlette.websockets import WebSocketState

from . import keys, server_wire


@dataclasses.dataclass(frozen=True)
class ServedUnixSocket(server_wire.ServedAddress):
    """A wire serving on a Unix socket. It names no host and no port, because a socket has neither."""

    uds: Path

    @property
    def meta(self) -> dict[str, Any]:
        return {keys.UDS: str(self.uds)}


class WebsocketServerConnection(server_wire.ServerConnection):
    """A server's end of one websocket session, over an accepted ``WebSocket``."""

    def __init__(self, websocket: WebSocket, served_address: server_wire.ServedAddress):
        self._websocket = websocket
        self._served_address = served_address

    @property
    def peer(self) -> str:
        return str(self._websocket.client)

    @property
    def served_address(self) -> server_wire.ServedAddress:
        return self._served_address

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


def _is_stale_socket(path: Path) -> bool:
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
            probe.connect(str(path))
        except ConnectionRefusedError:
            return True
        except OSError:
            return False
    return False


def claim_socket_path(path: Path) -> socket.socket:
    """Bind and listen on ``path``, and return the socket, or refuse a path a live server holds.

    A live path is refused: the bind precedes any probe, so the loser fails on ``EADDRINUSE`` and the
    probe reads the holder as live. An absent or a stale path holds no such claim. A socket is bound
    before it listens, and a probe in that window reads the binder as stale. Two starters on one path
    can then both unlink and rebind, and the first serves a socket nothing links to. Give each path
    one starter. Serve the returned socket by its descriptor: a server handed the path instead binds
    again, and unlinks this claim.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            sock.bind(str(path))
        except OSError as taken:
            if taken.errno != errno.EADDRINUSE:
                raise
            if not _is_stale_socket(path):
                raise OSError(errno.EADDRINUSE, f'{path!r} is already in use') from None
            os.unlink(path)
            sock.bind(str(path))
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


class WebsocketWire(server_wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and the API answers beside it.

    ``served_address`` is what this binds: a host and a port, or a Unix socket path. A socket file
    stays after ``stop``, where an unlink could take a path a successor has claimed.
    """

    # How long ``stop`` lets an open session finish before it cuts the connection. The uvicorn default
    # waits for ever, and a session mid-inference holds the whole server open.
    STOP_GRACE_SEC = 2

    def __init__(self, served_address: server_wire.ServedAddress):
        if isinstance(served_address, ServedUnixSocket) and not served_address.uds.is_absolute():
            # A relative path is resolved against whatever directory the server was started from, so the
            # path an operator wrote and the path a client dials would part company on the next start.
            raise ValueError(f'{served_address.uds!r} is a relative socket path; bind an absolute one')
        self._binds = served_address
        self._sockets: list[socket.socket] = []
        self._server: uvicorn.Server | None = None
        self._served_address: server_wire.ServedAddress | None = None
        self._served = False

    @property
    def served_address(self) -> server_wire.ServedAddress:
        assert self._served_address is not None, 'The websocket wire has not started'
        return self._served_address

    async def start(
        self, session: server_wire.SessionHandler, authorized: server_wire.Authorized, api: APIRouter
    ) -> None:
        self._served = False
        binds = self._binds
        if isinstance(binds, ServedUnixSocket):
            self._sockets = [claim_socket_path(binds.uds)]
            self._served_address = binds
            bound_port, host = 0, ''
        else:
            assert isinstance(binds, server_wire.ServedHostPort), f'{type(binds).__name__} names no address to bind'
            host = binds.host
            self._sockets = _listening_sockets(host, binds.port)
            # A wire asked for port 0 binds any free one, so what it serves on is known only now.
            bound_port = self._sockets[0].getsockname()[1]
            self._served_address = server_wire.ServedHostPort(host, bound_port)
        app = FastAPI()
        app.include_router(api)
        self._route_sessions(app, session, authorized)
        config = uvicorn.Config(
            app,
            host=host,
            port=bound_port,
            log_level='info',
            ws=WS_IMPL,
            ws_max_size=wire.MAX_MESSAGE_BYTES,
            timeout_graceful_shutdown=self.STOP_GRACE_SEC,
        )
        self._server = uvicorn.Server(config)

    def _route_sessions(
        self, app: FastAPI, session: server_wire.SessionHandler, authorized: server_wire.Authorized
    ) -> None:
        async def require_auth(websocket: WebSocket) -> None:
            """Refuse before ``accept()``. An unauthorized peer never reaches the session handshake."""
            if not authorized(websocket.headers):
                raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

        async def serve_pinned_model(websocket: WebSocket) -> None:
            """Serve the model the server pinned. The path names a model; every query param is a pipeline override."""
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.served_address), None)
            if (
                websocket.application_state is WebSocketState.CONNECTED
                and websocket.client_state is WebSocketState.CONNECTED
            ):
                await websocket.close()

        async def serve_named_model(websocket: WebSocket, model_id: str) -> None:
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.served_address), model_id)
            if (
                websocket.application_state is WebSocketState.CONNECTED
                and websocket.client_state is WebSocketState.CONNECTED
            ):
                await websocket.close()

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
