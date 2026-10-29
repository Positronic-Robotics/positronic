import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from enum import Enum
from functools import partial
from http.client import HTTPMessage
from types import MappingProxyType
from typing import IO, Any, TypeVar

from positronic_model_server import protocol, serialization
from positronic_wire import wire
from positronic_wire.wire import ClientWire

from positronic import telemetry, telemetry_keys
from positronic.offboard import protocol as legacy_protocol
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


# A Modal sticky session ends this long after its last in-flight request, an open websocket included. Short, so a
# session whose dial never came frees its container soon.
MODAL_SESSION_IDLE_TIMEOUT = 120
MODAL_SESSION_START_PATH = '/_modal/sessions/start'
MODAL_SESSION_TERMINATE_PATH = '/_modal/sessions/terminate'
MODAL_SESSION_HEADER = 'Modal-Authorization'
# A session's idle timeout, on its start. A keepalive carries it too: the edge forwards the session id to the
# container and not the timeout, and the server reports the session's idle timeout when it knows it.
MODAL_SESSION_IDLE_TIMEOUT_HEADER = 'x-modal-server-session-idle-timeout'
# Modal's edge answers a session start that waited this long with a 5xx; the start then continues under its key.
_MODAL_START_REQUEST_TIMEOUT = 330.0
# The most of an answer to a session call that the client reads: a token in a JSON object fits in it many times.
_MAX_MODAL_ANSWER_BYTES = 16 * 1024


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Answers a redirect as an ``HTTPError``. A session call carries the endpoint's credentials, so it goes to the
    endpoint alone. ``urllib`` makes a redirected ``POST`` a ``GET`` without its body, which no session call is."""

    def redirect_request(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> None:
        return None


_SESSION_CALLS = urllib.request.build_opener(_RefuseRedirects)


def _session_token(answer: bytes) -> str:
    """The token in the answer to a session start.

    Raises ``wire.ConnectRefused`` with ``FINAL`` where the answer carries none. The token goes into a header, so it
    is one run of visible ASCII characters.
    """
    try:
        token = json.loads(answer)['token']
    except (ValueError, LookupError, TypeError, RecursionError) as e:
        raise wire.ConnectRefused(wire.Refusal.FINAL, 'Modal answered a session start without a token') from e
    if not isinstance(token, str) or not token or not all('!' <= c <= '~' for c in token):
        raise wire.ConnectRefused(wire.Refusal.FINAL, 'Modal answered a session start with a token no header holds')
    return token


class ModalSessions:
    """The sticky sessions that one client holds on a Modal Server (`@app.server` with `@modal.sessioned`).

    A start places one session on one container: on a free container, on a container that Modal starts for it, or
    on the next container that frees. Its token routes every request of the session to that container. The edge
    answers a start that waits past its own limit with a 5xx; the next attempt carries the same idempotency key, so
    the start continues and places one session. A start under a new key places another session, which holds a
    container until its idle timeout.

    The session calls go to ``base_url`` alone: the client follows no redirect. The client holds the sessions its
    open sessions dialed, and a spare that a call outside a session started for the next session to dial. A
    pickled copy holds none of them.
    """

    def __init__(self, base_url: str, idle_timeout: int = MODAL_SESSION_IDLE_TIMEOUT):
        self._base_url = base_url.rstrip('/')
        self.idle_timeout = idle_timeout
        # A call outside a session can run beside a session, so a lock guards the held sessions, and a second one
        # lets one call at a time start the spare.
        self._lock = threading.Lock()
        self._open: dict[str, None] = {}
        self._spare: str | None = None
        self._spare_start = threading.Lock()
        # Starts in flight (each waits for a container). A call that finds one waits for its token instead of
        # starting a spare, which would boot a second container on a cold app.
        self._starting = 0
        self._started = threading.Condition(self._lock)

    def __reduce__(self) -> tuple[type['ModalSessions'], tuple[str, int]]:
        # The held sessions belong to this process, and a lock does not pickle.
        return ModalSessions, (self._base_url, self.idle_timeout)

    def _post(self, path: str, headers: Mapping[str, str], timeout: float) -> bytes:
        """The answer to a ``POST`` on ``path`` at the endpoint. A refusal and a redirect raise
        ``urllib.error.HTTPError``."""
        request = urllib.request.Request(self._base_url + path, data=b'', method='POST', headers=dict(headers))
        with _SESSION_CALLS.open(request, timeout=timeout) as answer:
            return answer.read(_MAX_MODAL_ANSWER_BYTES)

    def start(self, headers: Mapping[str, str] | None, deadline: float) -> str:
        """The token of a new session. Raises `wire.ConnectRefused` when the edge refuses the start, and
        `TimeoutError` when no container takes it before the `time.monotonic()` instant `deadline`."""
        key, stamp = str(uuid.uuid4()), str(time.time())
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f'No Modal session started by the deadline ({self._base_url})')
            start_headers = {
                **(headers or {}),
                MODAL_SESSION_IDLE_TIMEOUT_HEADER: str(self.idle_timeout),
                'x-idempotency-key': key,
                'x-modal-timestamp': stamp,
            }
            try:
                return _session_token(
                    self._post(MODAL_SESSION_START_PATH, start_headers, min(left, _MODAL_START_REQUEST_TIMEOUT))
                )
            except urllib.error.HTTPError as refused:
                if refused.code < 500:
                    raise wire.ConnectRefused(
                        wire.Refusal.FORBIDDEN if refused.code in (401, 403) else wire.Refusal.FINAL,
                        f'Modal refused a session start with {refused.code}',
                    ) from None
            except (TimeoutError, urllib.error.URLError) as lost:
                logger.info('Modal session start got no answer (%s); asking again under the same key', lost)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def terminate(self, headers: Mapping[str, str] | None, token: str) -> None:
        """End the session, so its container takes the next one. A session that already ended answers 404.

        A session that this call does not end ends at its idle timeout, so a failure is logged and not raised.
        """
        try:
            self._post(MODAL_SESSION_TERMINATE_PATH, {**(headers or {}), 'x-modal-server-session-token': token}, 60)
        except urllib.error.HTTPError as answer:
            if answer.code != 404:
                logger.error('Modal session terminate answered %s; the session ends at its idle timeout', answer.code)
        except (TimeoutError, urllib.error.URLError) as lost:
            logger.error('Modal session terminate got no answer (%s); the session ends at its idle timeout', lost)

    def open(self, headers: Mapping[str, str] | None, deadline: float) -> str:
        """The token of a new session, held open: ``start``, counted as in flight."""
        return self._counted_start(headers, deadline, spare=False)

    def hold(self, token: str) -> None:
        """Hold open the session ``token`` names, which a dial reuses."""
        with self._lock:
            self._open[token] = None

    def take_spare(self) -> str | None:
        """The spare session, for a dial that then holds it open; ``None`` without one."""
        with self._lock:
            token, self._spare = self._spare, None
            return token

    def end(self, headers: Mapping[str, str] | None, token: str) -> None:
        """Forget the session ``token`` names, and end it."""
        self.forget(token)
        self.terminate(headers, token)

    def forget(self, token: str) -> None:
        """Drop ``token`` from the sessions the client holds."""
        with self._lock:
            self._open.pop(token, None)
            if self._spare == token:
                self._spare = None

    def call_token(self, headers: Mapping[str, str] | None, deadline: float) -> tuple[str, bool]:
        """The token that a call outside a session carries, and whether the client held it before the call.

        That is the token of an open session, or else of the spare, which this starts when there is none. While a
        start is in flight, the call waits for its token (up to the ``time.monotonic()`` instant ``deadline``)
        instead of starting a spare: on a cold app the start waits for a container, and a second start would boot
        another one.
        """
        with self._spare_start:
            with self._lock:
                while True:
                    held = next(reversed(self._open), None) or self._spare
                    if held is not None:
                        return held, True
                    left = deadline - time.monotonic()
                    if not self._starting or left <= 0:
                        break
                    self._started.wait(timeout=left)
            return self._counted_start(headers, deadline, spare=True), False

    def call_headers(self, token: str) -> dict[str, str]:
        """The headers of a call outside a session on the session ``token`` names."""
        return {MODAL_SESSION_HEADER: f'Bearer {token}', MODAL_SESSION_IDLE_TIMEOUT_HEADER: str(self.idle_timeout)}

    def _counted_start(self, headers: Mapping[str, str] | None, deadline: float, *, spare: bool) -> str:
        """``start``, counted as in flight so that a call meanwhile waits for its token. The token is held (as the
        spare, or open) in the same step that ends the count, so a waiting call never sees neither."""
        with self._lock:
            self._starting += 1
        token: str | None = None
        try:
            token = self.start(headers, deadline)
            return token
        finally:
            with self._lock:
                self._starting -= 1
                if token is not None:
                    if spare:
                        self._spare = token
                    else:
                        self._open[token] = None
                self._started.notify_all()


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
        on_close: Callable[[], None] | None = None,
    ):
        self._conn = conn
        self._on_close = on_close
        self._infer_timeout = infer_timeout
        self._reopen = reopen
        ready = self._handshake(ready_by)
        self._protocol = resolve_version(
            {**legacy_protocol.VERSIONS, protocol.ProtocolVersion.V3.value: Version(protocol.ProtocolVersion.V3)},
            ready.get(protocol.PROTOCOL_VERSION, 1),
            'policy protocol',
        )
        self._metadata = ready[protocol.META]
        self._session_id = None if self._protocol is protocol.ProtocolVersion.V1 else ready[protocol.SESSION_ID]
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

    def infer(self, obs: dict[str, Any]) -> Any:
        """Send an observation and receive native v3 data, or typed robot commands for legacy versions.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too.
        """
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
        if reopened.protocol_version is not self._protocol or not self._wire_values_equal(
            reopened.metadata, self._metadata
        ):
            reopened.close()
            raise wire.PeerDisconnected('The server this session reconnected to declares other metadata')
        self._conn, self._session_id, self._closed = reopened._conn, reopened.session_id, False
        # A reconnect can hold a new Modal session; its close ends that one. A reconnect that could not continue the
        # old Modal session already ended it (the failed attempt's ``on_close``).
        self._on_close, reopened._on_close = reopened._on_close, None

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
        """End the server session and wait for its cleanup before closing the connection.

        ``on_close`` runs once, also for a session whose connection a timeout, a drop or a server error already
        closed: on a Modal Server it terminates the Modal session, which otherwise holds its container until the
        session's idle timeout, and lets the client forget that session's token.
        """
        on_close, self._on_close = self._on_close, None
        try:
            if not self._closed:
                self._closed = True
                self._end_server_session()
        finally:
            if on_close is not None:
                on_close()

    def _end_server_session(self) -> None:
        if self._protocol is protocol.ProtocolVersion.V1:
            logger.info('InferenceSession.close: %s', self._conn.close())
            return
        message = {protocol.SESSION_ID: self._session_id, protocol.END_SESSION: True}
        try:
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

    ``modal_sessions`` is for a Modal Server with sticky sessions (``ModalSessions``): each session starts a Modal
    session before it dials, and dials with its token. The start waits for the container, so one session starts
    one container. The edge refuses every request without a token, so the calls outside a session carry one too.
    ``.modal_remote`` in ``positronic.cfg.policy`` sets it, and ``.modal_remote_legacy`` the long dial for a Modal
    web endpoint without sessions.
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
        modal_sessions: bool = False,
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
        self._modal_sessions = ModalSessions(_https_root(self.session_url)) if modal_sessions else None

    def _open_session(
        self,
        ready_by: float,
        reopen: Callable[[str | None], InferenceSession] | None,
        open_timeout: float,
        modal_token: str | None = None,
    ) -> InferenceSession:
        """One attempt at a session, whose dial waits ``open_timeout`` at most and whose handshake waits for no
        status update past the ``time.monotonic()`` instant ``ready_by``. The connection closes when the handshake
        does not finish.

        On a Modal Server with sticky sessions, the attempt starts a Modal session first, or dials the one that
        ``modal_token`` names: a reconnect reaches the container of the session it continues. A failed attempt ends
        its Modal session.

        A refusal sent as a protocol frame (a rejected session param) raises past every
        transport handler, and a connection may hold a reader thread until it is closed.
        """
        headers, token = self.headers, modal_token
        on_close: Callable[[], None] | None = None
        if (sessions := self._modal_sessions) is not None:
            if token is None:
                # The session waits for its container here, so the dial below opens at once. The start and the dial
                # fit in one connect deadline.
                token = sessions.open(self.headers, ready_by)
                open_timeout = max(0.0, min(open_timeout, ready_by - time.monotonic()))
            else:
                sessions.hold(token)
            headers = {MODAL_SESSION_HEADER: f'Bearer {token}'}
            on_close = partial(sessions.end, self.headers, token)

        try:
            conn = self._wire.dial(self._address, headers, open_timeout)
        except BaseException:
            if on_close is not None:
                on_close()
            raise
        try:
            return InferenceSession(
                conn,
                infer_timeout=self.infer_timeout,
                ready_by=ready_by,
                reopen=(lambda: reopen(token)) if reopen is not None else None,
                on_close=on_close,
            )
        except BaseException:
            conn.close()
            if on_close is not None:
                on_close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the server's model.

        Raises ``wire.ConnectRefused`` when the wire refuses the session and no retry clears it.
        """
        return self._connect(
            self.connect_deadline,
            reopen=lambda token: self._connect(self.reconnect_deadline, reopen=None, modal_token=token),
        )

    def wait_for_server(self) -> None:
        """Return once a server answers on the address, without opening a session.

        Retries a cold server as ``new_session`` does, until the connect deadline. Raises ``wire.ConnectRefused``
        when the wire refuses the server and no retry clears it. On a Modal Server with sticky sessions the probe
        carries a session token, as ``keepalive`` does: on a cold app its start wakes a container, and the next
        ``new_session`` dials that session.
        """
        deadline = time.monotonic() + self.connect_deadline

        def probe(headers: Mapping[str, str] | None, open_timeout: float) -> None:
            if (refusal := self._wire.probe(self._address, headers, open_timeout)) is not None:
                raise wire.ConnectRefused(refusal, f'{self.session_url} refused the probe: {refusal.value}')

        self._retry(deadline, lambda open_timeout: self._call_outside_a_session(probe, deadline, open_timeout))

    def _connect(
        self,
        deadline_sec: float,
        reopen: Callable[[str | None], InferenceSession] | None,
        modal_token: str | None = None,
    ) -> InferenceSession:
        """A session on the server, retrying a cold backend for ``deadline_sec`` of wall clock.

        A reconnect dials the Modal session that ``modal_token`` names, and a new session dials the spare that a
        call outside a session started. Only the first attempt dials that token: a failed attempt ends its Modal
        session, and the next one starts a new one. The edge answers a dial on a session that ended (its idle
        timeout passed, or it was ended) with 404, which reads as a permanent refusal: that one starts a new session
        at once and spends no retry.
        """
        deadline = time.monotonic() + deadline_sec
        if modal_token is None and self._modal_sessions is not None:
            modal_token = self._modal_sessions.take_spare()
        held = iter([modal_token] if modal_token is not None else [])

        def attempt(open_timeout: float) -> InferenceSession:
            token = next(held, None)
            if token is None:
                return self._open_session(deadline, reopen, open_timeout)
            try:
                return self._open_session(deadline, reopen, open_timeout, token)
            except wire.ConnectRefused as refused:
                if refused.refusal is not wire.Refusal.FINAL:
                    raise
                logger.info('The Modal session this dial named has ended (%s); starting a new one', refused)
            return self._open_session(deadline, reopen, open_timeout)

        return self._retry(deadline, attempt)

    def _retry(self, deadline: float, attempt: Callable[[float], T]) -> T:
        """``attempt``, retried while the server is cold, until the ``time.monotonic()`` instant ``deadline``.

        ``attempt`` takes the seconds its dial may wait. The first attempt gets the whole ``open_timeout``, and a
        retry gets ``open_timeout`` or the time left, whichever is shorter.
        """
        backoff = 1.0
        retries = ConnectRetries()
        open_timeout = self.open_timeout
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
            logger.info('Server not ready (cold start?): %s; retrying in %.0fs', not_ready, backoff)
            time.sleep(max(0.0, min(backoff, deadline - time.monotonic())))
            backoff = min(backoff * 2, 30.0)
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{not_ready} (connecting to {self.session_url})') from not_ready
            open_timeout = max(0.0, min(self.open_timeout, deadline - time.monotonic()))

    def keepalive(self) -> int | None:
        """Reset the server's idle timer, outside any session. Returns the seconds the server stays alive after
        the call, or ``None`` for a server with no idle timeout.

        A server binds its wires only after its model has loaded and warmed, so any answer means it is ready.
        Raises ``wire.KeepaliveUnsupported`` where the server serves sessions but not the call.

        On a Modal Server with sticky sessions the call carries a session token, so it reaches that session's
        container. Each call keeps the spare session alive; unused, it ends ``ModalSessions.idle_timeout`` after
        the last call.
        """
        return self._call_outside_a_session(
            partial(self._wire.keepalive, self._address), time.monotonic() + self.connect_deadline, self.open_timeout
        )

    def _call_outside_a_session(
        self, call: Callable[[Mapping[str, str] | None, float], T], deadline: float, timeout: float
    ) -> T:
        """``call`` on the headers of a call outside a session, and on ``timeout``.

        On a Modal Server with sticky sessions the headers carry a session token: the token of a session this client
        has open, or else of a spare session that this starts. The start waits for a container, up to the
        ``time.monotonic()`` instant ``deadline``, so on a cold app it wakes one, and the next ``new_session``
        dials the spare instead of starting another. The start and the call fit in one deadline: the call gets
        ``timeout`` or the time left, whichever is shorter. A token whose session has ended (the edge answers 404, a
        permanent refusal) is dropped, and ``call`` runs once more on a new session.
        """
        sessions = self._modal_sessions
        if sessions is None:
            return call(self.headers, timeout)

        def call_on(token: str) -> T:
            return call(sessions.call_headers(token), max(0.0, min(timeout, deadline - time.monotonic())))

        token, held = sessions.call_token(self.headers, deadline)
        try:
            return call_on(token)
        except wire.ConnectRefused as refused:
            if not held or refused.refusal is not wire.Refusal.FINAL:
                raise
            logger.info('The Modal session this call named has ended (%s); starting a new one', refused)
        sessions.forget(token)
        return call_on(sessions.call_token(self.headers, deadline)[0])


def _https_root(session_url: str) -> str:
    """The `https://host[:port]` of a `wss://` or `https://` session URL: where the Modal session calls go."""
    parts = urllib.parse.urlsplit(session_url)
    if parts.scheme not in ('wss', 'https') or not parts.hostname or '@' in parts.netloc:
        raise ValueError(f'Modal sessions need a TLS endpoint, and this session URL is {session_url}')
    return f'https://{parts.netloc}'
