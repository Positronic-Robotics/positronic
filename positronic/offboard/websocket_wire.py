"""The server side of the websocket wire."""

import socket

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from positronic_wire import wire
from starlette.datastructures import QueryParams

from . import server_wire


class WebsocketServerConnection(server_wire.ServerConnection):
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


# uvicorn's default ('websockets') reassembles an 846 KiB observation in 58 ms, against 29 ms here
# (measured by positronic/offboard/serving_cost.py).
WS_IMPL = 'websockets-sansio'


class WebsocketWire(server_wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and ``api`` answers on the same port."""

    # How long ``stop`` lets an open session finish before it cuts the connection. The uvicorn default
    # waits for ever, and a session mid-inference holds the whole server open.
    STOP_GRACE_SEC = 2

    def __init__(self, host: str, port: int, api: APIRouter):
        self._host = host
        self._port = port
        self._api = api
        self._sockets: list[socket.socket] = []
        self._server: uvicorn.Server | None = None
        self._endpoint: wire.Endpoint | None = None
        self._served = False

    @property
    def endpoint(self) -> wire.Endpoint:
        assert self._endpoint is not None, 'The websocket wire has not started'
        return self._endpoint

    async def start(self, session: server_wire.SessionHandler, authorized: server_wire.Authorized) -> None:
        self._served = False
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
