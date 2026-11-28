"""A client for a roboarena server: the msgpack frames, over the `roboarena` wire that carries them."""

import logging
import time
from collections.abc import Mapping
from threading import local
from typing import Any

from positronic_model_server.serialization import deserialize, serialize
from positronic_wire import registry, wire
from positronic_wire import roboarena as roboarena_wire

from positronic import telemetry, telemetry_keys
from positronic.offboard.client import (
    DEFAULT_UNRESOLVED_HOST_DEADLINE,
    ConnectOutcome,
    ConnectRetries,
    Dial,
    connection_attrs,
    time_left,
)

logger = logging.getLogger(__name__)

# The key that says what a frame asks of the server, and the two values this client sends.
ENDPOINT = 'endpoint'
INFER = 'infer'
RESET = 'reset'
# The text the backend answers a reset with.
RESET_ACKNOWLEDGEMENT = 'reset successful'

# The session a frame belongs to, on a server that keeps per-session history.
SESSION_ID = 'session_id'

# How long each read waits, in seconds: a backbone loading, one forward pass, one acknowledgement.
HANDSHAKE_TIMEOUT_S = 60.0
INFER_TIMEOUT_S = 120.0
RESET_TIMEOUT_S = 10.0
# How long a connect may take by default: the dial and the config read, each up to `HANDSHAKE_TIMEOUT_S`.
DEFAULT_CONNECT_DEADLINE_S = 2 * HANDSHAKE_TIMEOUT_S
# A silent peer must not hold a readiness probe for a handshake's wait.
READY_PROBE_TIMEOUT_S = 5.0


class RoboarenaClient:
    """One connection to a roboarena server, and the msgpack frames it carries."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 9000,
        headers: Mapping[str, str] | None = None,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE_S,
        unresolved_host_deadline: float = DEFAULT_UNRESOLVED_HOST_DEADLINE,
    ):
        """``headers`` go on every handshake this client opens: ``connect`` and ``probe`` alike.

        ``connect_deadline`` bounds one ``connect``: the dial and the config read share it, and neither waits
        more than ``HANDSHAKE_TIMEOUT_S``. A connect makes one attempt.

        ``unresolved_host_deadline`` bounds a run of ``probe`` calls that the resolver has no answer for. A probe
        that the server answers ends the run.
        """
        self._address = roboarena_wire.RoboarenaAddress(host, port)
        self._headers = headers
        self._connect_deadline = connect_deadline
        self._wire = registry.client_wire(roboarena_wire.RoboarenaClientWire.NAME)
        self._connection: wire.ClientConnection | None = None
        self._dial: Dial | None = None
        self._sent_on_conn = False
        self._calls = local()
        self._server_config: dict[str, Any] | None = None
        self._unresolved_host_deadline = unresolved_host_deadline
        self._probe_retries = ConnectRetries(unresolved_host_deadline)

    def connect(self) -> dict[str, Any]:
        """Open the connection and answer the config the server announces on it."""
        dial_started = time.monotonic()
        deadline = dial_started + self._connect_deadline
        connection = self._wire.dial(self._address, self._headers, min(HANDSHAKE_TIMEOUT_S, time_left(deadline)))
        dial = Dial(self.url, (time.monotonic() - dial_started) * 1000.0)
        try:
            announced: dict[str, Any] = deserialize(
                connection.recv(timeout=min(HANDSHAKE_TIMEOUT_S, time_left(deadline)))
            )
        except BaseException:
            # Nothing else holds this connection: `_connection` is assigned below, so an unclosed one
            # here leaks its socket until the process ends.
            connection.close(timeout=time_left(deadline))
            raise
        self._server_config = announced
        self._connection, self._dial, self._sent_on_conn = connection, dial, False
        logger.info(f'Connected to roboarena server, metadata: {announced}')
        return announced

    @property
    def url(self) -> str:
        """The URL this client dials."""
        return self._wire.session_url(self._address)

    @property
    def server_config(self) -> dict[str, Any]:
        """The ``PolicyServerConfig`` this backend announced on connect."""
        if self._server_config is None:
            raise RuntimeError('Not connected: the server announces its config on connect')
        return self._server_config

    def probe(self) -> wire.Refusal | None:
        """Probe the server once for the config it announces. Returns what ``ClientWire.probe`` returns.

        Raises ``TextAnswer`` when it answers in text, and ``wire.ConnectRefused`` on a refusal the connect retry
        policy surfaces.
        """
        refusal = self._wire.probe(self._address, self._headers, READY_PROBE_TIMEOUT_S)
        if refusal is None:
            # The server answered, so the next refusal starts a new run with a full budget.
            self._probe_retries = ConnectRetries(self._unresolved_host_deadline)
        elif self._probe_retries.take(refusal) is ConnectOutcome.SURFACE:
            raise wire.ConnectRefused(refusal, f'{self.url} refused the connection')
        return refusal

    @property
    def sent_alone(self) -> bool | None:
        """Whether this thread's latest inference went with no other inference in flight; ``None`` before it sends.

        A caller runs one exchange at a time, so each inference that sends goes alone.
        """
        return getattr(self._calls, 'alone', None)

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """The action chunk the server answers ``observation`` with.

        The server answers each request in turn on one connection, so a caller does not overlap two inferences.
        The send and the wait for the answer are the ``wire.send`` and ``wire.recv`` spans, as in
        ``InferenceSession``.
        """
        self._calls.alone = None
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        deadline = time.monotonic() + INFER_TIMEOUT_S
        try:
            serialised = serialize({**observation, ENDPOINT: INFER})
            new_connection, self._sent_on_conn = not self._sent_on_conn, True
            send_attrs = {
                telemetry_keys.ATTR_WIRE_BYTES: len(serialised),
                **connection_attrs(self._dial, new_connection),
            }
            self._calls.alone = True
            answer = self._exchange(serialised, send_attrs, deadline)
        except BaseException:
            # A reply that arrives after this read gave up stays queued, and the next inference reads it as its own.
            self.close()
            raise
        return deserialize(answer)

    def _exchange(self, serialised: bytes, send_attrs: Mapping[str, Any], deadline: float) -> bytes:
        """Send one request and read its answer by the ``time.monotonic()`` instant ``deadline``.

        The send and the read are each timed as their span. A failed send or read is timed too.
        """
        assert self._connection is not None
        send_started = time.time_ns()
        try:
            self._connection.send(serialised, timeout=time_left(deadline))
        finally:
            sent = time.time_ns()
            telemetry.record_span(telemetry_keys.SPAN_WIRE_SEND, send_started, sent, **send_attrs)
        try:
            return self._connection.recv(timeout=time_left(deadline))
        finally:
            telemetry.record_span(telemetry_keys.SPAN_WIRE_RECV, sent, time.time_ns())

    def reset(self, session_id: str | None = None) -> None:
        """End the server's history for ``session_id``, and read the acknowledgement off the connection.

        Dials a new connection when this client holds none.
        """
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        frame: dict[str, Any] = {ENDPOINT: RESET}
        if session_id is not None:
            frame[SESSION_ID] = session_id
        deadline = time.monotonic() + RESET_TIMEOUT_S
        try:
            self._connection.send(serialize(frame), timeout=time_left(deadline))
            self._connection.recv(timeout=time_left(deadline))
        except roboarena_wire.TextAnswer as e:
            # The peer ends the exchange with any text, so this connection carries nothing more.
            self.close()
            if e.text != RESET_ACKNOWLEDGEMENT:
                raise
            logger.debug(f'roboarena reset answered: {e.text}')
        except BaseException:
            # An acknowledgement that arrives after this read gave up stays queued, and the next inference reads it
            # as its own.
            self.close()
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
