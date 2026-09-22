import logging
import time
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

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
# One transport handshake, whichever the wire makes, and the retries until a cold backend answers.
DEFAULT_OPEN_TIMEOUT = 10.0
DEFAULT_CONNECT_DEADLINE = 900.0


class InferenceSession:
    """One session over one open connection, whichever wire carries it."""

    # The timing block of the last decoded inference response; empty when the server sent none, and
    # empty while a round trip is in flight. Declared here so an implementation that skips ``__init__``
    # still carries it.
    served_timing: Mapping[str, float] = MappingProxyType({})

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
        self.served_timing = {}
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        # The pair reads as the uplink and then the wait the server's own time sits inside: each span
        # holds the socket alone. A send outlasting its own bytes is an uplink too slow for the payload.
        wire_bytes = {telemetry_keys.ATTR_WIRE_BYTES: len(serialised)}
        with telemetry.span(telemetry_keys.SPAN_WIRE_SEND, **wire_bytes):
            self._conn.send(serialised)
        try:
            with telemetry.span(telemetry_keys.SPAN_WIRE_RECV):
                received = self._conn.recv(timeout=self._infer_timeout)
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._conn.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
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

    ``headers`` carry the credentials; the address carries none. ``open_timeout`` bounds one transport
    handshake, whichever the wire makes — a TCP or TLS one, or a connect to a Unix socket —
    ``connect_deadline`` the retries until a cold backend answers, and ``infer_timeout`` one inference
    round trip.
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
        """The models this server serves, read by the wire on the transport it carries sessions on."""
        return self._wire.list_models(self._address, self.headers, self.open_timeout)
