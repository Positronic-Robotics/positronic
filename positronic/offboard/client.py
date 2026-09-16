import logging
import time
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

import httpx
from positronic_wire import wire
from positronic_wire.wire import ClientWire

from positronic import telemetry, telemetry_keys

from . import protocol
from .protocol import deserialise, serialise, typed_commands

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0
# One TCP/TLS handshake, and the retries until a cold backend answers.
DEFAULT_OPEN_TIMEOUT = 10.0
DEFAULT_CONNECT_DEADLINE = 900.0

# What ``wire_timing`` reports: the uplink, and the wait that follows it. The server's own span sits
# inside the second one, so the two minus ``served_ms`` is what the link and the receiver cost.
SEND_MS = 'send_ms'
RECV_MS = 'recv_ms'


class InferenceSession:
    """One session over one open connection, whichever wire carries it."""

    # The timing block of the last decoded inference response; empty when the server sent none, and
    # empty while a round trip is in flight. Declared here so an implementation that skips ``__init__``
    # still carries it.
    served_timing: Mapping[str, float] = MappingProxyType({})
    # This client's own halves of the last round trip, under ``SEND_MS`` and ``RECV_MS``.
    wire_timing: Mapping[str, float] = MappingProxyType({})

    def __init__(self, conn: wire.ClientConnection, infer_timeout: float = DEFAULT_INFER_TIMEOUT):
        self._conn = conn
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake()

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        The server must send an update at least every ``timeout_per_message`` seconds.
        """
        try:
            while True:
                response = deserialise(self._conn.recv(timeout=timeout_per_message))
                if protocol.ERROR in response:
                    raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
                try:
                    status = protocol.ServerStatus(response.get(protocol.STATUS))
                except ValueError:
                    raise RuntimeError(f'Unexpected server response: {response}') from None

                if status is protocol.ServerStatus.READY:
                    return response[protocol.META]
                if status is protocol.ServerStatus.ERROR:
                    raise RuntimeError('Server error: Unknown error')

                message = response.get(protocol.MESSAGE, status)
                logger.info(f'Server status: [{status}] {message}')

        except TimeoutError:
            raise TimeoutError(
                f'Server did not send status update within {timeout_per_message}s. '
                f'Server may have crashed or model loading is taking too long without progress updates.'
            ) from None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> Any:
        """Send an observation and get the served session's result, with every robot-command channel typed.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too.
        """
        self.served_timing = self.wire_timing = {}
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        # The pair reads as the uplink and then the wait the server's own time sits inside: each span
        # holds the socket alone. A send outlasting its own bytes is an uplink too slow for the payload.
        wire_bytes = {telemetry_keys.ATTR_WIRE_BYTES: len(serialised)}
        send_started = time.time_ns()
        self._conn.send(serialised)
        sent = time.time_ns()
        telemetry.record_span(telemetry_keys.SPAN_WIRE_SEND, send_started, sent, **wire_bytes)
        try:
            received = self._conn.recv(timeout=self._infer_timeout)
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._conn.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        answered = time.time_ns()
        telemetry.record_span(telemetry_keys.SPAN_WIRE_RECV, sent, answered)
        self.wire_timing = {SEND_MS: (sent - send_started) / 1e6, RECV_MS: (answered - sent) / 1e6}
        response = deserialise(received)
        self.served_timing = response.get(protocol.TIMING) or {} if isinstance(response, dict) else {}
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        return typed_commands(response[protocol.RESULT])

    def close(self):
        logger.info('InferenceSession.close: %s', self._conn.close())


class _ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class _ConnectRetries:
    """The retry policy over one ``new_session``'s connect attempts.

    A ``FORBIDDEN`` refusal means a cold backend or a refused credential, and gets ``MAX_FORBIDDEN_ATTEMPTS``
    attempts.
    """

    MAX_FORBIDDEN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._forbidden_attempts = 0

    def take(self, refusal: wire.Refusal) -> _ConnectOutcome:
        """Spend a refused connect against the budget."""
        if refusal is wire.Refusal.FORBIDDEN:
            self._forbidden_attempts += 1
            again = self._forbidden_attempts < self.MAX_FORBIDDEN_ATTEMPTS
        else:
            again = refusal is wire.Refusal.COLD
        return _ConnectOutcome.RETRY if again else _ConnectOutcome.SURFACE


class InferenceClient:
    """The connection to one inference server: a wire, a session address, and the settings each session opens with.

    ``headers`` carry the credentials; the address carries none. ``open_timeout`` bounds one TCP/TLS
    handshake, ``connect_deadline`` the retries until a cold backend answers, and ``infer_timeout`` one
    inference round trip.
    """

    def __init__(
        self,
        client_wire: ClientWire,
        address: wire.SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ):
        self._wire = client_wire
        self._address = address
        self.session_url = client_wire.session_url(address)
        self.api_url = client_wire.api_url(address)
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout

    def _open_session(self) -> InferenceSession:
        """One attempt at a session. The connection closes when the handshake does not finish.

        A refusal sent as a protocol frame (an unknown model, a rejected session param) raises past every
        transport handler, and a connection may hold a reader thread until it is closed.
        """
        conn = self._wire.dial(self._address, self.headers, self.open_timeout)
        try:
            return InferenceSession(conn, infer_timeout=self.infer_timeout)
        except BaseException:
            conn.close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names.

        Raises ``wire.ConnectRefused`` when the wire refuses the session and no retry clears it.
        """
        deadline = time.monotonic() + self.connect_deadline
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            try:
                return self._open_session()
            except wire.ConnectRefused as e:
                refusal, not_ready = e.refusal, e
            # A status handshake the server did not finish: a backend that is not ready.
            except (TimeoutError, wire.PeerDisconnected) as e:
                refusal, not_ready = wire.Refusal.COLD, e
            if retries.take(refusal) is _ConnectOutcome.SURFACE:
                raise not_ready
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{not_ready} (connecting to {self.session_url})') from not_ready
            logger.info('Server not ready (cold start?): %s; retrying in %.0fs', not_ready, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def list_models(self) -> list[str]:
        """List available models from the server."""
        if self.api_url is None:
            raise ValueError(f'{self.session_url} names a wire that carries sessions alone; list the models over HTTP')
        response = httpx.get(f'{self.api_url}/{wire.MODELS_ROUTE}', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
