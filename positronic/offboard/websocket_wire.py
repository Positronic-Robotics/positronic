"""The websocket wire, and the two ends of a websocket session."""

import socket

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from starlette.datastructures import QueryParams
from websockets.sync.connection import Connection

from . import wire


class WebsocketClientConnection:
    """A client's end of one websocket session."""

    def __init__(self, websocket: Connection):
        self._websocket = websocket

    def send(self, message: bytes) -> None:
        self._websocket.send(message)

    def recv(self, timeout: float | None = None) -> bytes:
        message = self._websocket.recv(timeout=timeout)
        assert isinstance(message, bytes), f'A frame is bytes, and this one is {type(message).__name__}'
        return message

    def close(self) -> str:
        state_before_close = self._websocket.state.name
        self._websocket.close()
        # A close that times out still reaches CLOSED locally; only the close code says the server answered.
        return f'state {state_before_close} -> {self._websocket.state.name}, close code {self._websocket.close_code}'


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
        await self._websocket.send_bytes(message)

    async def receive(self) -> bytes:
        try:
            return await self._websocket.receive_bytes()
        except WebSocketDisconnect as e:
            raise wire.PeerDisconnected(str(e)) from e

    async def refuse(self, reason: str) -> None:
        await self._websocket.close(code=1008, reason=reason[:100])


def _listening_socket(host: str, port: int) -> socket.socket:
    """A socket bound on ``host``, where a ``port`` of 0 takes any free one.

    The family comes from ``host`` itself, so an IPv6 host binds an IPv6 socket. Binding here rather
    than inside uvicorn names the port before the wire serves, and holds it from then on.
    """
    family, kind, proto, _canonical, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.socket(family, kind, proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(address)
    # Listening here rather than at the first accept makes the port answer from the moment ``start``
    # returns: the kernel queues a connect that beats the serving loop to it.
    sock.listen()
    return sock


class WebsocketWire(wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and ``api`` answers on the same port.

    One uvicorn serves both, so the routes a client reads a model catalogue from sit on the endpoint it
    opens sessions on.
    """

    # How long ``stop`` lets an open session finish before it cuts the connection. Left to itself
    # uvicorn waits for ever, so a session mid-inference would hold the whole server open.
    STOP_GRACE_SEC = 2

    def __init__(self, host: str, port: int, api: APIRouter):
        self._host = host
        self._port = port
        self._api = api
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._endpoint: wire.Endpoint | None = None
        self._served = False

    @property
    def endpoint(self) -> wire.Endpoint:
        assert self._endpoint is not None, 'The websocket wire has not started'
        return self._endpoint

    async def start(self, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        self._socket = _listening_socket(self._host, self._port)
        self._endpoint = wire.Endpoint(self._host, self._socket.getsockname()[1])
        app = FastAPI()
        app.include_router(self._api)
        self._route_sessions(app, session, authorized)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._endpoint.port,
            log_level='info',
            ws_max_size=wire.MAX_MESSAGE_BYTES,
            timeout_graceful_shutdown=self.STOP_GRACE_SEC,
        )
        self._server = uvicorn.Server(config)

    def _route_sessions(self, app: FastAPI, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        async def require_auth(websocket: WebSocket) -> None:
            """Refuses before ``accept()``, so an unauthorized peer never reaches the session handshake."""
            if not authorized(websocket.headers):
                raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

        async def serve_pinned_model(websocket: WebSocket) -> None:
            """Serves the model the server pinned. Naming a model is the path's job, so every query param
            here is a pipeline override."""
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.endpoint), None)

        async def serve_named_model(websocket: WebSocket, model_id: str) -> None:
            await websocket.accept()
            await session(WebsocketServerConnection(websocket, self.endpoint), model_id)

        auth = [Depends(require_auth)]
        app.websocket(wire.SESSION_PATH, dependencies=auth)(serve_pinned_model)
        # ``:path`` so an id that is itself a path (a HuggingFace repo, say) opens under the name the
        # model catalogue advertises.
        app.websocket(f'{wire.SESSION_PATH}/{{model_id:path}}', dependencies=auth)(serve_named_model)

    async def serve(self) -> None:
        assert self._server is not None and self._socket is not None, 'The websocket wire has not started'
        self._served = True
        await self._server.serve(sockets=[self._socket])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        # uvicorn releases the socket as it shuts down. A startup that rolls back stops a wire that
        # bound but never served, so uvicorn never runs; close the socket here to free the port.
        if self._socket is not None and not self._served:
            self._socket.close()
