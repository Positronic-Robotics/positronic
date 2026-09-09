"""The transports one session runs over, and the two ends of an open one.

A wire carries the ``protocol`` frames as opaque bytes and reads none of them, so the handshake and
the inference loop read the same over every wire. ``grpc_wire`` holds the gRPC one.
"""

import abc
from typing import Protocol

from fastapi import WebSocket, WebSocketDisconnect
from starlette.datastructures import QueryParams
from websockets.sync.connection import Connection

# The route a session opens on. The websocket wire puts it in the URL; the gRPC wire names it in the
# session metadata, so both wires address a model the same way.
SESSION_PATH = '/api/v1/session'


class PeerDisconnected(Exception):
    """The peer ended the session."""


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

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    @property
    def peer(self) -> str:
        return str(self._websocket.client)

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
