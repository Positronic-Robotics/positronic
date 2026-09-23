"""A client for a roboarena server: the msgpack frames, over the `roboarena` wire that carries them."""

import logging
from collections.abc import Mapping
from enum import Enum
from typing import Any

from positronic_wire import registry, wire
from positronic_wire import roboarena as roboarena_wire

from positronic.offboard.client import ConnectOutcome, ConnectRetries
from positronic.utils.serialization import deserialize, serialize

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
# A silent peer must not hold a readiness probe for a handshake's wait.
READY_PROBE_TIMEOUT_S = 5.0


class ProbeOutcome(Enum):
    READY = 'ready'
    NOT_READY = 'not_ready'


class RoboarenaClient:
    """One connection to a roboarena server, and the msgpack frames it carries."""

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self._address = roboarena_wire.RoboarenaAddress(host, port)
        self._wire = registry.client_wire(roboarena_wire.RoboarenaClientWire.NAME)
        self._connection: wire.ClientConnection | None = None
        self._server_config: dict[str, Any] | None = None
        self._probe_retries = ConnectRetries()

    def connect(self) -> dict[str, Any]:
        """Open the connection and answer the config the server announces on it."""
        connection = self._wire.dial(self._address, None, HANDSHAKE_TIMEOUT_S)
        try:
            announced: dict[str, Any] = deserialize(connection.recv(timeout=HANDSHAKE_TIMEOUT_S))
        except BaseException:
            # Nothing else holds this connection: `_connection` is assigned below, so an unclosed one
            # here leaks its socket until the process ends.
            connection.close()
            raise
        self._server_config = announced
        self._connection = connection
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

    def probe(self) -> ProbeOutcome:
        """Probe the server once for the config it announces.

        Raises ``TextAnswer`` when it answers in text, and ``wire.ConnectRefused`` on a refusal the connect retry
        policy surfaces.
        """
        refusal = self._wire.probe(self._address, None, READY_PROBE_TIMEOUT_S)
        if refusal is None:
            return ProbeOutcome.READY
        if self._probe_retries.take(refusal) is ConnectOutcome.SURFACE:
            raise wire.ConnectRefused(refusal, f'{self.url} refused the connection')
        return ProbeOutcome.NOT_READY

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """The action chunk the server answers ``observation`` with."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        try:
            self._connection.send(serialize({**observation, ENDPOINT: INFER}))
            answer = self._connection.recv(timeout=INFER_TIMEOUT_S)
        except BaseException:
            # A reply that arrives after this read gave up stays queued, and the next inference reads it as its own.
            self.close()
            raise
        return deserialize(answer)

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
        try:
            self._connection.send(serialize(frame))
            self._connection.recv(timeout=RESET_TIMEOUT_S)
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
