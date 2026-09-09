"""The gRPC wire: one bidirectional stream per session, carrying the same ``protocol`` frames.

The stream is untyped bytes on both sides, so there is no protobuf schema and no generated code: a
generic handler with no serialiser hands each frame over as it arrived.
"""

import logging
import queue
import threading
import urllib.parse
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

import grpc
import grpc.aio
from starlette.datastructures import QueryParams

from . import wire

logger = logging.getLogger(__name__)

# The one method every session runs on. gRPC routes by this path alone.
SERVICE = 'positronic.offboard.v1.Inference'
METHOD = 'Session'
METHOD_PATH = f'/{SERVICE}/{METHOD}'

# What the websocket wire says in the URL, said here in the session metadata.
SESSION_PATH_HEADER = 'positronic-session-path'
SESSION_QUERY_HEADER = 'positronic-session-query'

_MESSAGE_SIZE_OPTIONS = [
    ('grpc.max_receive_message_length', wire.MAX_MESSAGE_BYTES),
    ('grpc.max_send_message_length', wire.MAX_MESSAGE_BYTES),
]

# How long ``close`` waits for the server to end the stream, so its own session cleanup runs.
_CLOSE_TIMEOUT_SEC = 5.0


class GrpcClientConnection:
    """A client's end of one gRPC session.

    A reader thread drains the response stream into a queue, because the stream itself has no
    per-message timeout and ``recv`` needs one.
    """

    def __init__(
        self,
        target: str,
        session_path: str,
        query: str,
        headers: Mapping[str, str] | None = None,
        open_timeout: float = 10.0,
    ):
        self._target = target
        self._channel = grpc.insecure_channel(target, options=_MESSAGE_SIZE_OPTIONS)
        try:
            grpc.channel_ready_future(self._channel).result(timeout=open_timeout)
        except grpc.FutureTimeoutError:
            self._channel.close()
            raise TimeoutError(f'gRPC channel to {target} is not ready within {open_timeout}s') from None
        # gRPC metadata keys are lower case, and they are the same header names the websocket wire sends.
        metadata = tuple((key.lower(), value) for key, value in (headers or {}).items()) + (
            (SESSION_PATH_HEADER, session_path),
            (SESSION_QUERY_HEADER, query),
        )
        self._outbox: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._inbox: queue.SimpleQueue[bytes | BaseException] = queue.SimpleQueue()
        self._closed = False
        call = self._channel.stream_stream(METHOD_PATH, request_serializer=None, response_deserializer=None)
        self._responses = call(self._requests(), metadata=metadata)
        self._reader = threading.Thread(target=self._read, name='grpc-session-reader', daemon=True)
        self._reader.start()

    def _requests(self):
        """The outbound frames. ``None`` ends the stream, which half-closes the session."""
        while (message := self._outbox.get()) is not None:
            yield message

    def _read(self) -> None:
        """Drain the response stream into the inbox, ending it with what stopped it."""
        try:
            for message in self._responses:
                self._inbox.put(message)
            self._inbox.put(wire.PeerDisconnected(f'{self._target} ended the session'))
        except Exception as e:
            self._inbox.put(e)
        finally:
            self._responses.cancel()

    def send(self, message: bytes) -> None:
        self._outbox.put(message)

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            answer = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f'No message from {self._target} within {timeout}s') from None
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def close(self) -> str:
        if self._closed:
            return 'already closed'
        self._closed = True
        self._outbox.put(None)
        # The half-close ends the server's session, and the server then ends the stream. Waiting for
        # that lets the server release its model slot; closing the channel now would cut it short.
        self._reader.join(timeout=_CLOSE_TIMEOUT_SEC)
        server_ended_stream = not self._reader.is_alive()
        self._channel.close()
        # The websocket wire reads the same two facts off a close code. A stream the server never ended
        # means it still holds this session, so the next one's handshake waits on a slot nobody released.
        return f'peer had ended the stream {self._ended}, server ended it within {_CLOSE_TIMEOUT_SEC}s {server_ended_stream}'


def model_id_of(session_path: str) -> str | None:
    """The model a session path names, or ``None`` where it names the model the server pinned."""
    prefix = f'{wire.SESSION_PATH}/'
    if session_path == wire.SESSION_PATH:
        return None
    if not session_path.startswith(prefix):
        raise ValueError(f'Unexpected session path {session_path!r}; expected {wire.SESSION_PATH}[/<model_id>]')
    return urllib.parse.unquote(session_path[len(prefix) :])


class GrpcServerConnection(wire.ServerConnection):
    """A server's end of one gRPC session."""

    def __init__(self, requests: AsyncIterator[bytes], context: grpc.aio.ServicerContext, headers: Mapping[str, str]):
        self._requests = requests
        self._context = context
        self._headers = headers

    @property
    def peer(self) -> str:
        return self._context.peer()

    @property
    def session_path(self) -> str:
        return self._headers.get(SESSION_PATH_HEADER, wire.SESSION_PATH)

    @property
    def query_params(self) -> QueryParams:
        return QueryParams(self._headers.get(SESSION_QUERY_HEADER, ''))

    async def send(self, message: bytes) -> None:
        await self._context.write(message)

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


def _bind_target(host: str, port: int) -> str:
    """The address to bind, with an IPv6 literal in the brackets gRPC's target syntax requires."""
    return f'[{host}]:{port}' if ':' in host else f'{host}:{port}'


async def serve(
    serve_session: Callable[[GrpcServerConnection], Awaitable[None]],
    authorized: Callable[[Mapping[str, str]], bool],
    host: str,
    port: int,
) -> grpc.aio.Server:
    """Start a gRPC server that gives every accepted session to ``serve_session``.

    ``authorized`` reads the session headers and refuses before the session opens, as the websocket
    wire refuses the upgrade.
    """

    async def _serve_one(requests: AsyncIterator[bytes], context: grpc.aio.ServicerContext) -> None:
        headers = _headers(context)
        if not authorized(headers):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, 'Invalid or missing bearer token')
        try:
            await serve_session(GrpcServerConnection(requests, context, headers))
        except Exception as e:
            # The session itself reports what it can over the stream; anything reaching here happened
            # before or beyond that, so the client learns of it from the status alone.
            logger.error(f'Failed gRPC session: {e}', exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    handler = grpc.stream_stream_rpc_method_handler(_serve_one, request_deserializer=None, response_serializer=None)
    server = grpc.aio.server(options=_MESSAGE_SIZE_OPTIONS)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, {METHOD: handler}),))
    bound = server.add_insecure_port(_bind_target(host, port))
    if bound == 0:
        # gRPC reports a refused bind by returning port 0, so a server left to start here would
        # accept nothing and say nothing.
        raise OSError(f'gRPC could not bind {_bind_target(host, port)}')
    await server.start()
    logger.info(f'gRPC sessions on {host}:{bound}')
    return server
