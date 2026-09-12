"""The websocket wire, and the two ends of a websocket session."""

import socket
import ssl
from collections.abc import Mapping
from http import HTTPStatus

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from starlette.datastructures import QueryParams
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from . import wire


class WebsocketClientConnection:
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


def dial(url: str, headers: Mapping[str, str] | None, open_timeout: float) -> WebsocketClientConnection:
    """A client's end of one session on ``url``. Raises ``wire.ConnectRefused`` when the upgrade does not open."""
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
    except InvalidStatus as e:
        raise wire.ConnectRefused(_status_refusal(e.response.status_code), str(e)) from e
    except ssl.SSLCertVerificationError as e:
        raise wire.ConnectRefused(wire.Refusal.FINAL, str(e)) from e
    # A timed-out connect, a reset TLS handshake, a refused upgrade, a dropped handshake: a backend that is
    # not ready.
    except (TimeoutError, ssl.SSLError, ConnectionClosed, InvalidHandshake) as e:
        raise wire.ConnectRefused(wire.Refusal.COLD, str(e)) from e
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
        await self._websocket.send_bytes(message)

    async def receive(self) -> bytes:
        try:
            return await self._websocket.receive_bytes()
        except WebSocketDisconnect as e:
            raise wire.PeerDisconnected(str(e)) from e

    async def refuse(self, reason: str) -> None:
        await self._websocket.close(code=1008, reason=reason[:100])


def _listening_socket(host: str, port: int) -> socket.socket:
    """A listening socket bound on ``host``, where a ``port`` of 0 takes any free one."""
    family, kind, proto, _canonical, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.socket(family, kind, proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(address)
    # The port answers from the moment ``start`` returns: the kernel queues a connect that arrives before
    # the serving loop runs.
    sock.listen()
    return sock


class WebsocketWire(wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and ``api`` answers on the same port."""

    # How long ``stop`` lets an open session finish before it cuts the connection. The uvicorn default
    # waits for ever, and a session mid-inference holds the whole server open.
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
        assert self._server is not None and self._socket is not None, 'The websocket wire has not started'
        self._served = True
        await self._server.serve(sockets=[self._socket])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        # uvicorn releases the socket when it shuts down. A wire that bound but never served has no
        # uvicorn to release it.
        if self._socket is not None and not self._served:
            self._socket.close()
