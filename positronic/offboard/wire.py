"""The transports a session runs over, the two ends of an open one, and the server's end of a wire.

A wire carries the ``protocol`` frames as opaque bytes and reads none of them, so the handshake and
the inference loop read the same over every wire. ``grpc_wire`` holds the gRPC one.
"""

import abc
import socket
from collections.abc import Awaitable, Callable, Mapping
from typing import NamedTuple, Protocol

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status
from starlette.datastructures import QueryParams
from websockets.sync.connection import Connection

# The route a session opens on. The websocket wire puts it in the URL; the gRPC wire names it in the
# session metadata, so both wires address a model the same way.
SESSION_PATH = '/api/v1/session'

# The largest frame a session may carry, on either wire. An observation is a stack of camera frames,
# so the gRPC default of 4 MiB refuses one; uvicorn's own default happens to be this, and passing it
# explicitly is what keeps the two wires equal when that default moves.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# How long ``WebsocketWire.stop`` lets an open session finish before it cuts the connection. Left to
# itself uvicorn waits for ever, so a session mid-inference would hold the whole server open.
STOP_GRACE_SEC = 2


class PeerDisconnected(Exception):
    """The peer ended the session."""


class Endpoint(NamedTuple):
    """Where a wire serves."""

    host: str
    port: int


class ClientConnection(Protocol):
    """A client's end of one open session."""

    def send(self, message: bytes) -> None: ...

    def recv(self, timeout: float | None = None) -> bytes:
        """The next message. Raises ``TimeoutError`` when none arrives in time."""
        ...

    def close(self) -> str:
        """Close this end, and report what the wire saw, for the log.

        A peer that answered the close leaves a different trace from one that had already gone while the
        server still held the session, and the second is what strands the next session's handshake. Only
        the wire can tell the two apart, and each says it in its own terms.
        """
        ...


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


class ServerConnection(abc.ABC):
    """A server's end of one open session."""

    @property
    @abc.abstractmethod
    def peer(self) -> str:
        """Whom this session serves, for the log."""

    @property
    @abc.abstractmethod
    def endpoint(self) -> Endpoint:
        """Where the wire that accepted this session serves."""

    @property
    @abc.abstractmethod
    def query_params(self) -> QueryParams:
        """The session params the client asked for."""

    @abc.abstractmethod
    async def send(self, message: bytes) -> None: ...

    @abc.abstractmethod
    async def receive(self) -> bytes:
        """The next message. Raises ``PeerDisconnected`` once the client ends the session."""

    @abc.abstractmethod
    async def refuse(self, reason: str) -> None:
        """End a session the server cannot serve, telling the client why."""


class WebsocketServerConnection(ServerConnection):
    """A server's end of one websocket session, over an accepted ``WebSocket``."""

    def __init__(self, websocket: WebSocket, endpoint: Endpoint):
        self._websocket = websocket
        self._endpoint = endpoint

    @property
    def peer(self) -> str:
        return str(self._websocket.client)

    @property
    def endpoint(self) -> Endpoint:
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
            raise PeerDisconnected(str(e)) from e

    async def refuse(self, reason: str) -> None:
        await self._websocket.close(code=1008, reason=reason[:100])


# What a wire hands the server for each session it accepts: the connection, and the model the route
# names, which is ``None`` where the route names the model the server pinned.
SessionHandler = Callable[[ServerConnection, str | None], Awaitable[None]]

# Whether the session headers carry a credential the server accepts. Header names are lower case.
Authorized = Callable[[Mapping[str, str]], bool]


class Wire(abc.ABC):
    """One transport that sessions arrive on.

    A wire reads its own route for the model a session names, and refuses an unauthorized peer before
    the session opens. So a server hands every wire one ``SessionHandler`` and serves them all alike.
    """

    @property
    @abc.abstractmethod
    def endpoint(self) -> Endpoint:
        """Where this wire serves. The port is bound, and so known, once ``start`` returns."""

    @abc.abstractmethod
    async def start(self, session: SessionHandler, authorized: Authorized) -> None:
        """Bind, and give every accepted session to ``session``. Raises when the port is not free."""

    @abc.abstractmethod
    async def serve(self) -> None:
        """Carry sessions until ``stop``, or until the wire ends for its own reason."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """End the wire, and every session on it."""


def _listening_socket(host: str, port: int) -> socket.socket:
    """A socket bound on ``host``, where a ``port`` of 0 takes any free one.

    The family comes from ``host`` itself, so an IPv6 host binds an IPv6 socket. Binding here rather
    than inside uvicorn is what names the port before the wire serves, and holds it from then on.
    """
    family, kind, proto, _canonical, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.socket(family, kind, proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(address)
    # Listening here rather than at the first accept is what makes the port answer from the moment
    # ``start`` returns: the kernel queues a connect that beats the serving loop to it.
    sock.listen()
    return sock


class WebsocketWire(Wire):
    """The websocket wire: a session upgrades on ``SESSION_PATH``, and ``api`` answers on the same port.

    One uvicorn serves both, so the routes a client reads a model catalogue from sit on the endpoint it
    opens sessions on.
    """

    def __init__(self, host: str, port: int, api: APIRouter):
        self._host = host
        self._port = port
        self._api = api
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._endpoint: Endpoint | None = None

    @property
    def endpoint(self) -> Endpoint:
        assert self._endpoint is not None, 'The websocket wire has not started'
        return self._endpoint

    async def start(self, session: SessionHandler, authorized: Authorized) -> None:
        self._socket = _listening_socket(self._host, self._port)
        self._endpoint = Endpoint(self._host, self._socket.getsockname()[1])
        app = FastAPI()
        app.include_router(self._api)
        self._route_sessions(app, session, authorized)
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._endpoint.port,
            log_level='info',
            ws_max_size=MAX_MESSAGE_BYTES,
            timeout_graceful_shutdown=STOP_GRACE_SEC,
        )
        self._server = uvicorn.Server(config)

    def _route_sessions(self, app: FastAPI, session: SessionHandler, authorized: Authorized) -> None:
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
        app.websocket(SESSION_PATH, dependencies=auth)(serve_pinned_model)
        # ``:path`` so an id that is itself a path (a HuggingFace repo, say) opens under the name the
        # model catalogue advertises.
        app.websocket(f'{SESSION_PATH}/{{model_id:path}}', dependencies=auth)(serve_named_model)

    async def serve(self) -> None:
        assert self._server is not None and self._socket is not None, 'The websocket wire has not started'
        await self._server.serve(sockets=[self._socket])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
