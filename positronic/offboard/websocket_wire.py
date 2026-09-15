"""The websocket wire, and the two ends of a websocket session."""

import socket
import ssl
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import httpx
import uvicorn
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from starlette.datastructures import QueryParams
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
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


def _status_refusal(status_code: int) -> wire.Refusal:
    """What a non-101 answer to the upgrade says about the server."""
    if status_code == HTTPStatus.FORBIDDEN:
        return wire.Refusal.FORBIDDEN
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR or status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return wire.Refusal.COLD
    return wire.Refusal.FINAL


def _answers_model_catalogue(api_url: str, headers: Mapping[str, str] | None, timeout: float) -> bool:
    """Whether the model catalogue answers on ``api_url``.

    A server too old for a verb answers 404 for it, and so does an address serving something else. The
    catalogue is on every positronic server and tells the two apart.
    """
    try:
        answer = httpx.get(f'{api_url}/{wire.MODELS_ROUTE}', headers=dict(headers or {}), timeout=timeout)
    except httpx.HTTPError:
        return False
    return answer.status_code != HTTPStatus.NOT_FOUND


class WebsocketClientWire(wire.ClientWire):
    """The client side of the websocket wire, which the server's HTTP port carries beside its API."""

    SCHEME = 'ws'
    SECURE_SCHEME = 'wss'
    # A bare host names this wire, and so does an http(s) URL: the session upgrades from HTTP.
    ALIASES = (wire.Scheme('', secure=False), wire.Scheme('http', secure=False), wire.Scheme('https', secure=True))

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'{"https" if address.secure else "http"}://{address.netloc}{wire.API_PATH}'

    def dial(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> WebsocketClientConnection:
        """A client's end of one session on ``address``. Raises ``wire.ConnectRefused`` when it does not open."""
        try:
            # A proxy closes a connection it has read nothing from, often after 60 s, and one inference sends
            # nothing until it answers. The pings keep it open.
            websocket = connect(
                self.session_url(address),
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

    def call(
        self,
        address: wire.SessionAddress,
        verb: wire.Verb,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        """What the server answers ``verb`` with, over the HTTP API beside the session route.

        The call carries JSON, which the catalogue route beside it carries too, so a person reads the
        answer with ``curl``.
        """
        api_url = self.api_url(address)
        url = f'{api_url}/{verb.name}'
        try:
            answer = httpx.request(
                verb.http_method,
                url,
                json=dict(payload) if payload else None,
                headers=dict(headers or {}),
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise wire.ConnectRefused(wire.Refusal.COLD, f'{e} (calling {url})') from e
        if answer.status_code == HTTPStatus.NOT_FOUND:
            if _answers_model_catalogue(api_url, headers, timeout):
                raise wire.VerbUnsupported(f'{url} answers 404; this server serves sessions but not {verb.name}')
            raise wire.ConnectRefused(wire.Refusal.FINAL, f'{url} answers 404, and so does the model catalogue')
        if answer.status_code != HTTPStatus.OK:
            # An HTTP route refuses a credential with 401, where the upgrade beside it refuses with 403.
            unauthorized = answer.status_code == HTTPStatus.UNAUTHORIZED
            refusal = wire.Refusal.FORBIDDEN if unauthorized else _status_refusal(answer.status_code)
            raise wire.ConnectRefused(refusal, f'{url} answers {answer.status_code}')
        return answer.json()


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


# uvicorn's default ('websockets') reassembles an 846 KiB observation in 58 ms, against 29 ms here
# (measured by positronic/offboard/serving_cost.py).
WS_IMPL = 'websockets-sansio'


class WebsocketWire(wire.Wire):
    """The websocket wire: a session upgrades on ``wire.SESSION_PATH``, and the HTTP routes answer beside it."""

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

    async def start(self, session: wire.SessionHandler, verbs: wire.VerbHandler, authorized: wire.Authorized) -> None:
        self._served = False
        self._sockets = _listening_sockets(self._host, self._port)
        self._endpoint = wire.Endpoint(self._host, self._sockets[0].getsockname()[1])
        app = FastAPI()
        app.include_router(self._api)
        self._route_sessions(app, session, authorized)
        self._route_verbs(app, verbs, authorized)
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

    @staticmethod
    def _route_verbs(app: FastAPI, verbs: wire.VerbHandler, authorized: wire.Authorized) -> None:
        """Answer every unary verb on the HTTP API, under the same credential the session route takes."""

        async def require_auth(request: Request) -> None:
            if not authorized(request.headers):
                raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail='Invalid or missing bearer token')

        def answer(verb: wire.Verb):
            async def route(request: Request) -> Mapping[str, Any]:
                return await verbs(verb, await request.json() if await request.body() else {})

            return route

        for verb in wire.VERBS:
            app.add_api_route(verb.path, answer(verb), methods=[verb.http_method], dependencies=[Depends(require_auth)])

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
