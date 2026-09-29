"""The server side of the gRPC wire: one bidirectional stream per session, which carries the ``protocol`` frames,
and the unary keepalive call beside it."""

import json
import logging
from collections.abc import AsyncIterator, Mapping

import grpc
import grpc.aio
from positronic_wire import wire
from positronic_wire.grpc import (
    KEEPALIVE_METHOD,
    MESSAGE_SIZE_OPTIONS,
    METHOD,
    PING_EVERY_MS,
    SERVICE,
    SESSION_PATH_HEADER,
    SESSION_QUERY_HEADER,
    target,
)
from starlette.datastructures import QueryParams

from . import server_wire

logger = logging.getLogger(__name__)


class GrpcServerConnection(server_wire.ServerConnection):
    """A server's end of one gRPC session."""

    def __init__(
        self,
        requests: AsyncIterator[bytes],
        context: grpc.aio.ServicerContext,
        headers: Mapping[str, str],
        served_address: server_wire.ServedHostPort,
    ):
        self._requests = requests
        self._context = context
        self._headers = headers
        self._served_address = served_address

    @property
    def peer(self) -> str:
        return self._context.peer()

    @property
    def served_address(self) -> server_wire.ServedHostPort:
        return self._served_address

    @property
    def session_path(self) -> str:
        return self._headers.get(SESSION_PATH_HEADER, wire.SESSION_PATH)

    @property
    def query_params(self) -> QueryParams:
        return QueryParams(self._headers.get(SESSION_QUERY_HEADER, ''))

    async def send(self, message: bytes) -> None:
        try:
            await self._context.write(message)
        except grpc.RpcError as e:
            raise wire.PeerDisconnected(f'{self.peer} ended the session: {e}') from e

    async def receive(self) -> bytes:
        try:
            return await anext(self._requests)
        except StopAsyncIteration:
            raise wire.PeerDisconnected(f'{self.peer} ended the session') from None

    async def refuse(self, reason: str) -> None:
        self._context.set_code(grpc.StatusCode.ABORTED)
        self._context.set_details(reason)


def _headers(context: grpc.aio.ServicerContext) -> dict[str, str]:
    """The session metadata, as the header names both wires share. A ``-bin`` key carries no header."""
    return {key: value for key, value in (context.invocation_metadata() or ()) if isinstance(value, str)}


# The shortest ping interval the server answers without a strike: half of `PING_EVERY_MS`, so a client
# ping that arrives early is never one.
_PING_TOLERATED_EVERY_MS = PING_EVERY_MS // 2


def _server_options() -> list[tuple[str, int]]:
    return [
        *MESSAGE_SIZE_OPTIONS,
        # gRPC's own defaults, a five-minute floor and two strikes, answer a 20s ping with GOAWAY.
        ('grpc.http2.min_ping_interval_without_data_ms', _PING_TOLERATED_EVERY_MS),
        ('grpc.http2.max_ping_strikes', 0),
    ]


class GrpcWire(server_wire.Wire):
    """The gRPC wire: sessions on a port of their own, one bidirectional stream each, and the keepalive call.

    ``served_address`` is the host and the port it binds; a port of 0 binds any free one. The port is
    plaintext, and a TLS edge in front of it serves an authenticated endpoint.
    """

    def __init__(self, served_address: server_wire.ServedHostPort):
        self._binds = served_address
        self._server: grpc.aio.Server | None = None
        self._served_address: server_wire.ServedHostPort | None = None

    @property
    def served_address(self) -> server_wire.ServedHostPort:
        assert self._served_address is not None, 'The gRPC wire has not started'
        return self._served_address

    async def start(
        self,
        session: server_wire.SessionHandler,
        keepalive: server_wire.KeepaliveHandler,
        authorized: server_wire.Authorized,
    ) -> None:
        """Bind the gRPC port, which carries sessions and the keepalive call."""

        async def serve_one(requests: AsyncIterator[bytes], context: grpc.aio.ServicerContext) -> None:
            headers = _headers(context)
            if not authorized(headers):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, 'Invalid or missing bearer token')
            conn = GrpcServerConnection(requests, context, headers, self.served_address)
            if conn.session_path != wire.SESSION_PATH:
                refusal = f'No session on {conn.session_path!r}; sessions open on {wire.SESSION_PATH}'
                await context.abort(grpc.StatusCode.NOT_FOUND, refusal)
            try:
                await session(conn)
            except Exception as e:
                # The session reports its own errors over the stream. One that reaches here reaches the client
                # as the status alone.
                logger.error(f'Failed gRPC session: {e}', exc_info=True)
                await context.abort(grpc.StatusCode.INTERNAL, str(e))

        async def answer_keepalive(_request: bytes, context: grpc.aio.ServicerContext) -> bytes:
            if not authorized(_headers(context)):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, 'Invalid or missing bearer token')
            return json.dumps({wire.ALIVE_SECONDS: keepalive()}).encode()

        methods = {
            METHOD: grpc.stream_stream_rpc_method_handler(
                serve_one, request_deserializer=None, response_serializer=None
            ),
            KEEPALIVE_METHOD: grpc.unary_unary_rpc_method_handler(
                answer_keepalive, request_deserializer=None, response_serializer=None
            ),
        }
        server = grpc.aio.server(options=_server_options())
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, methods),))
        bound = server.add_insecure_port(target(self._binds.host, self._binds.port))
        if bound == 0:
            # gRPC reports a refused bind as port 0, and a server started on it accepts nothing and says nothing.
            raise OSError(f'gRPC could not bind {target(self._binds.host, self._binds.port)}')
        self._server = server
        self._served_address = server_wire.ServedHostPort(self._binds.host, bound)
        await server.start()
        logger.info(f'gRPC sessions on {self._binds.host}:{bound}')

    async def serve(self) -> None:
        assert self._server is not None, 'The gRPC wire has not started'
        await self._server.wait_for_termination()

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop(grace=None)
