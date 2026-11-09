import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import Enum
from threading import Condition, Lock, local
from typing import Any, TypeVar

from positronic_model_server import protocol, serialization
from positronic_wire import wire
from positronic_wire.wire import ClientWire

from positronic import telemetry, telemetry_keys
from positronic.offboard import protocol as legacy_protocol
from positronic.utils import check_positive_int
from positronic.utils.versions import Version, resolve_version

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
        self._protocol = resolve_version(
            {**legacy_protocol.VERSIONS, protocol.ProtocolVersion.V3.value: Version(protocol.ProtocolVersion.V3)},
            ready.get(protocol.PROTOCOL_VERSION, 1),
            'policy protocol',
        )
        self._metadata = ready[protocol.META]
        self._max_in_flight = check_positive_int(ready.get(protocol.MAX_IN_FLIGHT, 1), protocol.MAX_IN_FLIGHT)
        self._session_id = None if self._protocol is protocol.ProtocolVersion.V1 else ready[protocol.SESSION_ID]
        self._closed = False
        self._closing = False
        self._answered = False
        self._startup = Lock()
        self._send_lock = Lock()
        self._responses = Condition()
        self._pending: deque[object] = deque()
        self._timings = local()
        self.served_timing = self.wire_timing = {}

    @property
    def served_timing(self) -> Mapping[str, float]:
        """Server timings of this thread's latest call."""
        return self._timings.served

    @served_timing.setter
    def served_timing(self, value: Mapping[str, float]) -> None:
        self._timings.served = value

    @property
    def wire_timing(self) -> Mapping[str, float]:
        """Send and response-wait timings of this thread's latest call."""
        return self._timings.wire

    @wire_timing.setter
    def wire_timing(self, value: Mapping[str, float]) -> None:
        self._timings.wire = value

    def _handshake(self, ready_by: float, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        The server must send an update at least every ``timeout_per_message`` seconds. No wait for a further
        update begins past ``ready_by``, so a server that reports loading for ever is left there.
        """
        wait = timeout_per_message
        while True:
            try:
                raw = self._conn.recv(timeout=wait)
                response = serialization.deserialise(raw)
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
                return deserialise(raw) if response.get(protocol.PROTOCOL_VERSION, 1) in (1, 2) else response
            if status is protocol.ServerStatus.ERROR:
                raise RuntimeError('Server error: Unknown error')

            message = response.get(legacy_protocol.MESSAGE, status)
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

    @property
    def max_in_flight(self) -> int:
        """Server-declared client capacity, including completed answers not yet read."""
        return self._max_in_flight

    def infer(self, obs: dict[str, Any]) -> Any:
        """Send an observation and receive native v3 data, or typed robot commands for legacy versions.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too.

        Concurrent calls consume replies in actual send order, independent of observation timestamps.
        The first answer establishes the session before further sends, so a startup reconnect cannot
        lose queued observations.
        """
        self.served_timing = self.wire_timing = {}
        with self._startup:
            if not self._answered:
                return self._first_call(obs)
        return self._round_trip(obs)

    def _first_call(self, obs: dict[str, Any]) -> Any:
        if self._closed:
            raise wire.PeerDisconnected('The inference session is closed')
        try:
            return self._round_trip(obs)
        except wire.PeerDisconnected as dropped:
            # After an answer the server's session holds state for this episode, which a new session lacks.
            if self._reopen is None or self._answered:
                raise
            logger.warning('Inference connection dropped before its first answer (%s); reconnecting', dropped)
            self._adopt(self._reopen())
            # A second drop is a server that cannot serve this observation, and reaches the caller.
            return self._round_trip(obs)

    @staticmethod
    def _wire_values_equal(left: Any, right: Any) -> bool:
        """Compare wire values independently of mapping order and NumPy's elementwise equality."""
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return left.keys() == right.keys() and all(
                InferenceSession._wire_values_equal(value, right[key]) for key, value in left.items()
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                InferenceSession._wire_values_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        return serialise(left) == serialise(right)

    def _adopt(self, reopened: 'InferenceSession') -> None:
        """Carry on over ``reopened``'s connection, which must serve what this session opened on."""
        # The caller built its rig-side stack from this session's handshake, and the episode records that handshake.
        if (
            reopened.protocol_version is not self._protocol
            or reopened.max_in_flight != self._max_in_flight
            or not self._wire_values_equal(reopened.metadata, self._metadata)
        ):
            reopened.close()
            raise wire.PeerDisconnected('The server this session reconnected to declares other session settings')
        with self._send_lock:
            if not self._closing:
                self._conn, self._session_id, self._closed = reopened._conn, reopened.session_id, False
                return
        reopened.close()
        raise RuntimeError('The inference session is closing')

    def _round_trip(self, obs: dict[str, Any]) -> Any:
        self.served_timing = self.wire_timing = {}
        request = (
            obs
            if self._protocol is protocol.ProtocolVersion.V1
            else {protocol.SESSION_ID: self._session_id, protocol.OBSERVATION: obs}
        )
        serialised = (
            serialization.serialise(request) if self._protocol is protocol.ProtocolVersion.V3 else serialise(request)
        )
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        wire_bytes = {telemetry_keys.ATTR_WIRE_BYTES: len(serialised)}
        send_started = time.time_ns()
        token = object()
        try:
            try:
                with self._send_lock:
                    if self._closed:
                        raise wire.PeerDisconnected('The inference session is closed')
                    if self._closing:
                        raise RuntimeError('The inference session is closing')
                    with self._responses:
                        self._pending.append(token)
                    self._conn.send(serialised)
            finally:
                # A send that raises gets timed too, so the span is recorded on the way out.
                sent = time.time_ns()
                telemetry.record_span(telemetry_keys.SPAN_WIRE_SEND, send_started, sent, **wire_bytes)
            try:
                deadline = time.monotonic() + self._infer_timeout
                with self._responses:
                    ready = self._responses.wait_for(
                        lambda: self._closed or self._pending[0] is token, timeout=max(0.0, deadline - time.monotonic())
                    )
                    if self._closed:
                        raise wire.PeerDisconnected('The inference session is closed')
                    if not ready:
                        raise TimeoutError()
                received = self._conn.recv(timeout=max(0.0, deadline - time.monotonic()))
            finally:
                answered = time.time_ns()
                telemetry.record_span(telemetry_keys.SPAN_WIRE_RECV, sent, answered)
            self.wire_timing = {SEND_MS: (sent - send_started) / 1e6, RECV_MS: (answered - sent) / 1e6}
            return self._decode_answer(received)
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
        finally:
            with self._responses:
                if token in self._pending:
                    self._pending.remove(token)
                self._responses.notify_all()

    def _decode_answer(self, received: bytes) -> Any:
        self._answered = True
        response = (
            serialization.deserialise(received)
            if self._protocol is protocol.ProtocolVersion.V3
            else deserialise(received)
        )
        self.served_timing = response.get(protocol.TIMING) or {} if isinstance(response, dict) else {}
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            if response.get(protocol.STATUS) == protocol.ServerStatus.ERROR:
                self._closed = True
                self._conn.close()
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        result = response[protocol.RESULT]
        return result if self._protocol is protocol.ProtocolVersion.V3 else typed_commands(result)

    def close(self) -> None:
        """End the server session and wait for its cleanup before closing the connection."""
        with self._send_lock:
            self._closing = True
        with self._responses:
            drained = self._responses.wait_for(lambda: not self._pending or self._closed, timeout=self._infer_timeout)
            if not drained:
                self._closed = True
                self._responses.notify_all()
                self._conn.close()
                raise TimeoutError('Pending inference calls did not finish before session close')
        if self._closed:
            return
        try:
            if self._protocol is protocol.ProtocolVersion.V1:
                return
            message = {protocol.SESSION_ID: self._session_id, protocol.END_SESSION: True}
            # The server can acknowledge and close before the transport confirms the final write.
            with suppress(wire.PeerDisconnected):
                self._conn.send(serialization.serialise(message))
            raw = self._conn.recv(timeout=self._infer_timeout)
            response = (
                serialization.deserialise(raw) if self._protocol is protocol.ProtocolVersion.V3 else deserialise(raw)
            )
            if protocol.ERROR in response:
                raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
            if response != message:
                raise RuntimeError(f'Unexpected end-session response: {response}')
        finally:
            self._closed = True
            logger.info('InferenceSession.close: %s', self._conn.close())


class ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class ConnectRetries:
    """The retry policy over one run of refused connect attempts.

    ``COLD`` and ``SILENT`` retry to the deadline. A ``FORBIDDEN`` refusal means a cold backend or a refused
    credential, and gets ``MAX_FORBIDDEN_ATTEMPTS`` attempts.
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
            again = refusal in (wire.Refusal.COLD, wire.Refusal.SILENT)
        return ConnectOutcome.RETRY if again else ConnectOutcome.SURFACE


T = TypeVar('T')


class InferenceClient:
    """The connection to one inference server: a wire, a session address, and the settings each session opens with.

    ``headers`` carry the credentials; the address carries none. ``open_timeout`` bounds one transport
    handshake, whichever the wire makes — a TCP or TLS one, or a connect to a Unix socket —
    ``connect_deadline`` the retries until a cold backend answers, ``infer_timeout`` one inference
    round trip, and ``reconnect_deadline`` the rebuild of a connection that drops before its first answer.
    No attempt begins past a deadline, and no handshake waits for a further status update past it. The first
    dial gets the whole ``open_timeout``, so a deadline of 0 is one attempt, and no later dial waits past the
    deadline.

    An edge that holds a dial while it starts a backend wants an ``open_timeout`` as long as the cold start:
    one dial then waits through it. A shorter one makes the client re-dial, and such an edge can start a
    backend for each dial the client gave up on.
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

    def _open_session(
        self, ready_by: float, reopen: Callable[[], InferenceSession] | None, open_timeout: float
    ) -> InferenceSession:
        """One attempt at a session, whose dial waits ``open_timeout`` at most and whose handshake waits for no
        status update past the ``time.monotonic()`` instant ``ready_by``. The connection closes when the handshake
        does not finish.

        A refusal sent as a protocol frame (a rejected session param) raises past every
        transport handler, and a connection may hold a reader thread until it is closed.
        """
        conn = self._wire.dial(self._address, self.headers, open_timeout)
        try:
            return InferenceSession(conn, infer_timeout=self.infer_timeout, ready_by=ready_by, reopen=reopen)
        except BaseException:
            conn.close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the server's model.

        Raises ``wire.ConnectRefused`` when the wire refuses the session and no retry clears it.
        """
        # The reconnect deadline bounds the first dial of a reconnect too.
        return self._connect(
            self.connect_deadline,
            self.open_timeout,
            reopen=lambda: self._connect(
                self.reconnect_deadline, min(self.open_timeout, self.reconnect_deadline), reopen=None
            ),
        )

    def wait_for_server(self) -> None:
        """Return once a server answers on the address, without opening a session.

        Retries a cold server as ``new_session`` does, until the connect deadline. Raises ``wire.ConnectRefused``
        when the wire refuses the server and no retry clears it. On a wire that carries no probe, a keepalive that
        returns is the answer.
        """

        def probe(open_timeout: float) -> None:
            try:
                refusal = self._wire.probe(self._address, self.headers, open_timeout)
            except wire.ProbeUnsupported:
                self._wire.keepalive(self._address, self.headers, open_timeout)
                return
            if refusal is not None:
                raise wire.ConnectRefused(refusal, f'{self.session_url} refused the probe: {refusal.value}')

        self._retry(time.monotonic() + self.connect_deadline, probe, self.open_timeout)

    def _connect(
        self, deadline_sec: float, first_open_timeout: float, reopen: Callable[[], InferenceSession] | None
    ) -> InferenceSession:
        """A session on the server, retrying a cold backend for ``deadline_sec`` of wall clock. The first dial waits
        ``first_open_timeout`` at most."""
        deadline = time.monotonic() + deadline_sec
        return self._retry(
            deadline, lambda open_timeout: self._open_session(deadline, reopen, open_timeout), first_open_timeout
        )

    def _retry(self, deadline: float, attempt: Callable[[float], T], first_open_timeout: float) -> T:
        """``attempt``, retried while the server is cold, until the ``time.monotonic()`` instant ``deadline``.

        ``attempt`` takes the seconds its dial may wait. The first attempt gets ``first_open_timeout``, and a retry
        gets ``open_timeout`` or the time left, whichever is shorter.
        """
        pauses = wire.retry_pauses()
        retries = ConnectRetries()
        open_timeout = first_open_timeout
        while True:
            try:
                return attempt(open_timeout)
            except wire.ConnectRefused as e:
                refusal, not_ready = e.refusal, e
            # A status handshake the server did not finish: a backend that is not ready.
            except (TimeoutError, wire.PeerDisconnected) as e:
                refusal, not_ready = wire.Refusal.COLD, e
            if retries.take(refusal) is ConnectOutcome.SURFACE:
                raise not_ready
            pause = next(pauses)
            logger.info('Server not ready (cold start?): %s; retrying in %.0fs', not_ready, pause)
            time.sleep(max(0.0, min(pause, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{not_ready} (connecting to {self.session_url})') from not_ready
            open_timeout = max(0.0, min(self.open_timeout, deadline - time.monotonic()))

    def keepalive(self) -> int | None:
        """Reset the server's idle timer, outside any session. Returns the seconds the server stays alive after
        the call, or ``None`` for a server with no idle timeout.

        A server binds its wires only after its model has loaded and warmed, so any answer means it is ready.
        Raises ``wire.KeepaliveUnsupported`` where the server serves sessions but not the call.
        """
        return self._wire.keepalive(self._address, self.headers, self.open_timeout)
