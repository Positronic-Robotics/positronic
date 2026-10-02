import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from enum import Enum
from types import MappingProxyType
from typing import Any

from positronic_wire import wire
from positronic_wire.wire import ClientWire

from positronic import telemetry, telemetry_keys
from positronic.utils.versions import resolve_version

from . import protocol
from .protocol import deserialise, serialise, typed_commands

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0
# One transport handshake, whichever the wire makes, and the retries until a cold backend answers.
DEFAULT_OPEN_TIMEOUT = 10.0
DEFAULT_CONNECT_DEADLINE = 900.0
# How long ``infer`` retries to rebuild a connection that dropped before its first answer.
# FOOTGUN: the arm holds its last setpoint throughout, and an attended trial (``timeout_sec=None``) has no
# episode deadline behind this one to end that stall.
DEFAULT_RECONNECT_DEADLINE = 45.0

# What ``wire_timing`` reports: the uplink, and the wait that follows it. The link and the receiver
# cost the two minus ``served_ms``, because the server's own span sits inside the second one.
SEND_MS = 'send_ms'
RECV_MS = 'recv_ms'


class InferenceSession:
    """One connection using the protocol declared by its server. Finish inference before closing it.

    ``reopen`` returns a new session on the same server. A connection that drops before the server's first
    answer reconnects through it, once, and sends the observation again. ``ready_by`` is the
    ``time.monotonic()`` instant past which the status handshake waits for no further update.
    """

    # The timing block of the last decoded inference response; empty when the server sent none, and
    # empty while a round trip is in flight. Declared here so an implementation that skips ``__init__``
    # still carries it.
    served_timing: Mapping[str, float] = MappingProxyType({})
    # This client's own halves of the last round trip, under ``SEND_MS`` and ``RECV_MS``.
    wire_timing: Mapping[str, float] = MappingProxyType({})

    def __init__(
        self,
        conn: wire.ClientConnection,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        *,
        ready_by: float = math.inf,
        reopen: Callable[[], 'InferenceSession'] | None = None,
    ):
        self._conn = conn
        self._infer_timeout = infer_timeout
        self._reopen = reopen
        ready = self._handshake(ready_by)
        self._protocol = resolve_version(protocol.VERSIONS, ready.get(protocol.PROTOCOL_VERSION, 1), 'policy protocol')
        self._metadata = ready[protocol.META]
        self._session_id = ready[protocol.SESSION_ID] if self._protocol is protocol.ProtocolVersion.V2 else None
        self._closed = False
        self._answered = False

    def _handshake(self, ready_by: float, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        The server must send an update at least every ``timeout_per_message`` seconds. No wait for a further
        update begins past ``ready_by``, so a server that reports loading for ever is left there.
        """
        wait = timeout_per_message
        while True:
            try:
                response = deserialise(self._conn.recv(timeout=wait))
            except TimeoutError:
                if wait < timeout_per_message:
                    raise TimeoutError('Server was not ready in the time left to connect') from None
                raise TimeoutError(
                    f'Server did not send status update within {timeout_per_message}s. '
                    f'Server may have crashed or model loading is taking too long without progress updates.'
                ) from None
            if protocol.ERROR in response:
                raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
            try:
                status = protocol.ServerStatus(response.get(protocol.STATUS))
            except ValueError:
                raise RuntimeError(f'Unexpected server response: {response}') from None

            if status is protocol.ServerStatus.READY:
                return response
            if status is protocol.ServerStatus.ERROR:
                raise RuntimeError('Server error: Unknown error')

            message = response.get(protocol.MESSAGE, status)
            logger.info(f'Server status: [{status}] {message}')
            left = ready_by - time.monotonic()
            if left <= 0:
                raise TimeoutError('Server was not ready in the time left to connect')
            wait = min(timeout_per_message, left)

    @property
    def protocol_version(self) -> protocol.ProtocolVersion:
        return self._protocol

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any], prefix: Sequence[Mapping[str, Any]] | None = None) -> Any:
        """Send an observation and get the served session's result, with every robot-command channel typed.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too. ``prefix`` holds the
        commands the model continues from, and travels beside the observation.
        """
        if self._closed:
            raise wire.PeerDisconnected('The inference session is closed')
        try:
            return self._round_trip(obs, prefix)
        except wire.PeerDisconnected as dropped:
            # After an answer the server's session holds state for this episode, which a new session lacks.
            if self._reopen is None or self._answered:
                raise
            logger.warning('Inference connection dropped before its first answer (%s); reconnecting', dropped)
            self._adopt(self._reopen())
            # A second drop is a server that cannot serve this observation, and reaches the caller.
            return self._round_trip(obs, prefix)

    def _adopt(self, reopened: 'InferenceSession') -> None:
        """Carry on over ``reopened``'s connection, which must serve what this session opened on."""
        # The caller built its rig-side stack from this session's handshake, and the episode records that handshake.
        if reopened.protocol_version is not self._protocol or reopened.metadata != self._metadata:
            reopened.close()
            raise wire.PeerDisconnected('The server this session reconnected to declares other metadata')
        self._conn, self._session_id, self._closed = reopened._conn, reopened.session_id, False

    def _round_trip(self, obs: dict[str, Any], prefix: Sequence[Mapping[str, Any]] | None) -> Any:
        self.served_timing = self.wire_timing = {}
        if self._protocol is protocol.ProtocolVersion.V1:
            if prefix is not None:
                raise ValueError('A V1 server takes no prefix')
            request = obs
        else:
            request = {protocol.SESSION_ID: self._session_id, protocol.OBSERVATION: obs}
            if prefix is not None:
                request[protocol.PREFIX] = [dict(commands) for commands in prefix]
        serialised = serialise(request)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        # The pair reads as the uplink and then the wait the server's own time sits inside: each span
        # holds the socket alone. A send outlasting its own bytes is an uplink too slow for the payload.
        wire_bytes = {telemetry_keys.ATTR_WIRE_BYTES: len(serialised)}
        send_started = time.time_ns()
        try:
            try:
                self._conn.send(serialised)
            finally:
                # A send that raises gets timed too, so the span is recorded on the way out.
                sent = time.time_ns()
                telemetry.record_span(telemetry_keys.SPAN_WIRE_SEND, send_started, sent, **wire_bytes)
            try:
                received = self._conn.recv(timeout=self._infer_timeout)
            finally:
                answered = time.time_ns()
                telemetry.record_span(telemetry_keys.SPAN_WIRE_RECV, sent, answered)
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._closed = True
            self._conn.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        except wire.PeerDisconnected:
            self._closed = True
            self._conn.close()
            raise
        self._answered = True
        self.wire_timing = {SEND_MS: (sent - send_started) / 1e6, RECV_MS: (answered - sent) / 1e6}
        response = deserialise(received)
        self.served_timing = response.get(protocol.TIMING) or {} if isinstance(response, dict) else {}
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            if response.get(protocol.STATUS) == protocol.ServerStatus.ERROR:
                self._closed = True
                self._conn.close()
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        return typed_commands(response[protocol.RESULT])

    def close(self) -> None:
        """End the server session and wait for its cleanup before closing the connection."""
        if self._closed:
            return
        self._closed = True
        if self._protocol is protocol.ProtocolVersion.V1:
            logger.info('InferenceSession.close: %s', self._conn.close())
            return
        message = {protocol.SESSION_ID: self._session_id, protocol.END_SESSION: True}
        try:
            # The server can acknowledge and close before the transport confirms the final write.
            with suppress(wire.PeerDisconnected):
                self._conn.send(serialise(message))
            response = deserialise(self._conn.recv(timeout=self._infer_timeout))
            if protocol.ERROR in response:
                raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
            if response != message:
                raise RuntimeError(f'Unexpected end-session response: {response}')
        finally:
            logger.info('InferenceSession.close: %s', self._conn.close())


class ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class ConnectRetries:
    """The retry policy over one run of refused connect attempts.

    A ``FORBIDDEN`` refusal means a cold backend or a refused credential, and gets ``MAX_FORBIDDEN_ATTEMPTS``
    attempts.
    """

    MAX_FORBIDDEN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._forbidden_attempts = 0

    def take(self, refusal: wire.Refusal) -> ConnectOutcome:
        """Spend a refused connect against the budget."""
        if refusal is wire.Refusal.FORBIDDEN:
            self._forbidden_attempts += 1
            again = self._forbidden_attempts < self.MAX_FORBIDDEN_ATTEMPTS
        else:
            again = refusal is wire.Refusal.COLD
        return ConnectOutcome.RETRY if again else ConnectOutcome.SURFACE


class InferenceClient:
    """The connection to one inference server: a wire, a session address, and the settings each session opens with.

    ``headers`` carry the credentials; the address carries none. ``open_timeout`` bounds one transport
    handshake, whichever the wire makes — a TCP or TLS one, or a connect to a Unix socket —
    ``connect_deadline`` the retries until a cold backend answers, ``infer_timeout`` one inference
    round trip, and ``reconnect_deadline`` the rebuild of a connection that drops before its first answer.
    No attempt begins past a deadline, and no handshake waits for a further status update past it.
    """

    def __init__(
        self,
        client_wire: ClientWire[Any],
        address: wire.SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        reconnect_deadline: float = DEFAULT_RECONNECT_DEADLINE,
    ):
        if not isinstance(address, client_wire.ADDRESS):
            raise ValueError(
                f'{client_wire.NAME} dials a {client_wire.ADDRESS.__name__}, and this is a '
                f'{type(address).__name__}; build the address the wire names'
            )
        self._wire = client_wire
        self._address = address
        self.session_url = client_wire.session_url(address)
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout
        self.reconnect_deadline = reconnect_deadline

    def _open_session(self, ready_by: float, reopen: Callable[[], InferenceSession] | None) -> InferenceSession:
        """One attempt at a session, whose handshake waits for no status update past the ``time.monotonic()``
        instant ``ready_by``. The connection closes when the handshake does not finish.

        A refusal sent as a protocol frame (a rejected session param) raises past every
        transport handler, and a connection may hold a reader thread until it is closed.
        """
        conn = self._wire.dial(self._address, self.headers, self.open_timeout)
        try:
            return InferenceSession(conn, infer_timeout=self.infer_timeout, ready_by=ready_by, reopen=reopen)
        except BaseException:
            conn.close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the server's model.

        Raises ``wire.ConnectRefused`` when the wire refuses the session and no retry clears it.
        """
        return self._connect(self.connect_deadline, reopen=lambda: self._connect(self.reconnect_deadline, reopen=None))

    def _connect(self, deadline_sec: float, reopen: Callable[[], InferenceSession] | None) -> InferenceSession:
        """A session on the server, retrying a cold backend for ``deadline_sec`` of wall clock."""
        deadline = time.monotonic() + deadline_sec
        backoff = 1.0
        retries = ConnectRetries()
        while True:
            try:
                return self._open_session(deadline, reopen)
            except wire.ConnectRefused as e:
                refusal, not_ready = e.refusal, e
            # A status handshake the server did not finish: a backend that is not ready.
            except (TimeoutError, wire.PeerDisconnected) as e:
                refusal, not_ready = wire.Refusal.COLD, e
            if retries.take(refusal) is ConnectOutcome.SURFACE:
                raise not_ready
            logger.info('Server not ready (cold start?): %s; retrying in %.0fs', not_ready, backoff)
            time.sleep(max(0.0, min(backoff, deadline - time.monotonic())))
            backoff = min(backoff * 2, 30.0)
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{not_ready} (connecting to {self.session_url})') from not_ready

    def keepalive(self) -> int | None:
        """Reset the server's idle timer, outside any session. Returns the seconds the server stays alive after
        the call, or ``None`` for a server with no idle timeout.

        A server binds its wires only after its model has loaded and warmed, so any answer means it is ready.
        Raises ``wire.KeepaliveUnsupported`` where the server serves sessions but not the call.
        """
        return self._wire.keepalive(self._address, self.headers, self.open_timeout)
