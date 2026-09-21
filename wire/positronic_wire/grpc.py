"""The client side of the gRPC wire, and the call both ends of it agree on.

The stream is untyped bytes on both sides. There is no protobuf schema and no generated code.
"""

import queue
import threading
import time
from collections.abc import Mapping

import grpc
from positronic_wire import wire

# The one method every session runs on. gRPC routes by this path alone.
SERVICE = 'positronic.offboard.v1.Inference'
METHOD = 'Session'
METHOD_PATH = f'/{SERVICE}/{METHOD}'

# A path no handler serves: a probe of it opens no session on a server that is up.
PROBE_PATH = f'/{SERVICE}/ChannelProbe'

# The session path and the query cross as metadata: a gRPC call carries no URL path of its own.
SESSION_PATH_HEADER = 'positronic-session-path'
SESSION_QUERY_HEADER = 'positronic-session-query'

MESSAGE_SIZE_OPTIONS = [
    ('grpc.max_receive_message_length', wire.MAX_MESSAGE_BYTES),
    ('grpc.max_send_message_length', wire.MAX_MESSAGE_BYTES),
]

# How often the client pings an idle connection. A front drops a connection it reads nothing from.
PING_EVERY_MS = 20_000
_PING_ANSWER_TIMEOUT_MS = 10_000

# How long ``close`` waits for the server to end the stream and release the session.
_CLOSE_TIMEOUT_SEC = 5.0

# Status details naming a TLS failure that no retry clears: the edge's own configuration, or a certificate
# that covers an address this client did not dial.
_UNUSABLE_EDGE_DETAILS = (
    'CERTIFICATE_VERIFY_FAILED',
    'missing selected ALPN property',
    'Hostname Verification Check failed',
)

# Status details carrying an authoritative answer that the host has no address. A resolver that timed out
# says something else, and stays cold: a name can start resolving, where a misspelt one never does.
_NO_SUCH_HOST_DETAILS = ('Domain name not found', 'DNS server returned answer with no data')

# Status details naming a size limit: the metadata a session opens with, or a frame either end refuses.
# A retry sends the same oversized request, so no wait fixes it. Capacity exhaustion answers
# RESOURCE_EXHAUSTED too, says something else, and stays cold.
_HARD_LIMIT_DETAILS = ('exceeds hard limit', 'message larger than max')

_COLD_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.DEADLINE_EXCEEDED)


def _refusal(status: grpc.RpcError) -> wire.Refusal:
    """What a status that ended a call before it opened says about the server.

    ``PERMISSION_DENIED`` reads as 403, ``UNAVAILABLE`` as 503, ``RESOURCE_EXHAUSTED`` as 429. Refusals that
    no wait fixes wear a retryable code too — an unusable edge, a host with no address, a breached size
    limit — and their details tell them from a cold backend.
    """
    details = status.details() or ''
    if any(marker in details for marker in _UNUSABLE_EDGE_DETAILS + _NO_SUCH_HOST_DETAILS + _HARD_LIMIT_DETAILS):
        return wire.Refusal.FINAL
    code = status.code()
    if code is grpc.StatusCode.PERMISSION_DENIED:
        return wire.Refusal.FORBIDDEN
    return wire.Refusal.COLD if code in _COLD_CODES else wire.Refusal.FINAL


class GrpcClientConnection(wire.ClientConnection):
    """A client's end of one gRPC session, over a ready ``channel``.

    A reader thread drains the response stream into a queue: the stream has no per-message timeout, and
    ``recv`` needs one.
    """

    def __init__(self, channel: grpc.Channel, target: str, metadata: tuple[tuple[str, str], ...]):
        self._target = target
        self._channel = channel
        self._outbox: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._inbox: queue.SimpleQueue[bytes | BaseException] = queue.SimpleQueue()
        # One receipt per frame gRPC writes, and a last ``False`` once the call can write no more.
        self._written: queue.SimpleQueue[bool] = queue.SimpleQueue()
        self._closed = False
        self._ended = False
        self._received = False
        self._stopped_by: BaseException | None = None
        call = self._channel.stream_stream(METHOD_PATH, request_serializer=None, response_deserializer=None)
        self._responses = call(self._requests(), metadata=metadata)
        self._reader = threading.Thread(target=self._read, name='grpc-session-reader', daemon=True)
        self._reader.start()

    def _requests(self):
        """The outbound frames. ``None`` ends the stream, which half-closes the session."""
        while (message := self._outbox.get()) is not None:
            yield message
            # gRPC asks for the next frame only once it has written this one, so the resume is the receipt.
            self._written.put(True)

    def _read(self) -> None:
        """Drain the response stream into the inbox, and end the inbox with what stopped the stream."""
        try:
            for message in self._responses:
                self._inbox.put(message)
            self._stopped_by = wire.PeerDisconnected(f'{self._target} ended the session')
            self._inbox.put(self._stopped_by)
        except Exception as e:
            self._stopped_by = e
            self._inbox.put(e)
        finally:
            self._responses.cancel()
            # The call is over, so no further frame leaves the iterator. Release a send waiting on one.
            self._written.put(False)

    def send(self, message: bytes) -> None:
        if self._closed or self._ended:
            raise wire.PeerDisconnected(f'The session on {self._target} has ended')
        self._outbox.put(message)
        # The call's end releases this wait, so a dead connection raises here instead of blocking. The end is
        # recorded here too: the receipt queue says it once, and a later send must not wait for it again.
        if not self._written.get():
            self._ended = True
            raise wire.PeerDisconnected(f'{self._target} ended the session: {self._stopped_by}') from self._stopped_by

    def recv(self, timeout: float | None = None) -> bytes:
        # A reply that arrived during ``close`` sits in the inbox, and would pair one observation's actions
        # with the next observation.
        if self._closed:
            raise wire.PeerDisconnected(f'The session on {self._target} is closed')
        try:
            answer = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f'No message from {self._target} within {timeout}s') from None
        if isinstance(answer, BaseException):
            # What ended the stream is queued once. The caller reads it before ``send`` refuses a write.
            self._ended = True
            if isinstance(answer, grpc.RpcError):
                # A status before any message crossed is the server refusing the call. A status after one is a
                # lost peer, which the connect retry reads as cold.
                if not self._received:
                    raise wire.ConnectRefused(_refusal(answer), str(answer)) from answer
                raise wire.PeerDisconnected(f'{self._target} ended the session: {answer}') from answer
            raise answer
        self._received = True
        return answer

    def close(self) -> str:
        if self._closed:
            return 'already closed'
        self._closed = True
        self._outbox.put(None)
        # The half-close ends the server's session, and the server then ends the stream. A channel closed
        # before that cuts the server's cleanup short.
        self._reader.join(timeout=_CLOSE_TIMEOUT_SEC)
        server_ended_stream = not self._reader.is_alive()
        self._channel.close()
        # A stream the server never ended means the server still holds this session, and the next session's
        # handshake waits on its slot.
        return (
            f'peer had ended the stream {self._ended}, '
            f'server ended it within {_CLOSE_TIMEOUT_SEC}s {server_ended_stream}'
        )


# The largest share of one connect attempt's budget the refusal probe may spend. Both waits fit inside
# the caller's ``open_timeout``: a target that drops every connect answers neither.
_REFUSAL_PROBE_SEC = 1.0


def _client_options() -> list[tuple[str, int]]:
    return [
        *MESSAGE_SIZE_OPTIONS,
        ('grpc.keepalive_time_ms', PING_EVERY_MS),
        ('grpc.keepalive_timeout_ms', _PING_ANSWER_TIMEOUT_MS),
        # The gRPC default sends two pings without data, five minutes apart.
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', PING_EVERY_MS),
    ]


def target(host: str, port: int) -> str:
    """``host:port`` in gRPC's target syntax, which needs an IPv6 literal in brackets."""
    return f'{wire.bracket_ipv6(host)}:{port}'


def _channel(target: str, secure: bool) -> grpc.Channel:
    options = _client_options()
    if secure:
        # No roots named: the channel verifies the edge against the system's own roots.
        return grpc.secure_channel(target, grpc.ssl_channel_credentials(), options=options)
    return grpc.insecure_channel(target, options=options)


def _probe_share(open_timeout: float) -> float:
    """The share of one connect attempt the refusal probe gets; the readiness wait gets the rest.

    Half at most: an ``open_timeout`` under ``_REFUSAL_PROBE_SEC`` still waits for a healthy server.
    """
    return min(_REFUSAL_PROBE_SEC, open_timeout / 2)


def _connect_refusal(channel: grpc.Channel, timeout: float) -> grpc.RpcError | None:
    """What gRPC says stopped the channel. The readiness future says only that the channel is not ready."""
    probe = channel.stream_stream(PROBE_PATH, request_serializer=None, response_deserializer=None)
    try:
        next(probe(iter(()), timeout=timeout))
    except grpc.RpcError as e:
        return e
    except StopIteration:
        return None
    return None


def _ready_channel(target: str, secure: bool, open_timeout: float) -> grpc.Channel:
    """A channel to ``target`` that is ready. Raises ``wire.ConnectRefused`` when it is not within ``open_timeout``."""
    channel = _channel(target, secure)
    deadline = time.monotonic() + open_timeout
    try:
        grpc.channel_ready_future(channel).result(timeout=open_timeout - _probe_share(open_timeout))
    except grpc.FutureTimeoutError as not_ready:
        refusal = _connect_refusal(channel, timeout=max(0.0, deadline - time.monotonic()))
        # An ``UNIMPLEMENTED`` from the probe path means the channel is up: the readiness wait was too short.
        if refusal is not None and refusal.code() is grpc.StatusCode.UNIMPLEMENTED:
            return channel
        channel.close()
        if refusal is None:
            message = f'gRPC channel to {target} is not ready within {open_timeout}s'
            raise wire.ConnectRefused(wire.Refusal.COLD, message) from not_ready
        raise wire.ConnectRefused(_refusal(refusal), str(refusal)) from refusal
    return channel


class GrpcClientWire(wire.ClientWire):
    """The client side of the gRPC wire, whose port carries sessions alone."""

    SCHEME = 'grpc'
    SECURE_SCHEME = 'grpcs'

    def api_url(self, address: wire.SessionAddress) -> None:
        """None: the HTTP API answers on the server's own port."""
        return None

    def dial(
        self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> GrpcClientConnection:
        """A client's end of one session on ``address``. Raises ``wire.ConnectRefused`` when the channel does not open.

        A TLS address dials a TLS edge in front of the server's plaintext port.
        """
        dialled = target(address.host, address.port)
        channel = _ready_channel(dialled, address.secure, open_timeout)
        # gRPC metadata keys are lower case, as the server's authorization check reads them.
        metadata = tuple((key.lower(), value) for key, value in (headers or {}).items()) + (
            (SESSION_PATH_HEADER, address.path),
            (SESSION_QUERY_HEADER, address.query),
        )
        return GrpcClientConnection(channel, dialled, metadata)

    def probe(self, address: wire.SessionAddress, open_timeout: float) -> wire.Refusal | None:
        """A channel that reaches readiness, or answers ``PROBE_PATH``, names a server that is up."""
        try:
            channel = _ready_channel(target(address.host, address.port), address.secure, open_timeout)
        except wire.ConnectRefused as refused:
            return refused.refusal
        channel.close()
        return None
