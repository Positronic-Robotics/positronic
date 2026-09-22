"""A client for a roboarena server: the msgpack frames, over the `roboarena` wire that carries them."""

import logging
from collections.abc import Mapping
from typing import Any

from positronic_wire import registry, wire
from positronic_wire import roboarena as roboarena_wire

from positronic.utils.serialization import deserialize, serialize

logger = logging.getLogger(__name__)

# The key that says what a frame asks of the server, and the two values this client sends.
ENDPOINT = 'endpoint'
INFER = 'infer'
RESET = 'reset'

# The session a frame belongs to, on a server that keeps per-session history.
SESSION_ID = 'session_id'

# How long each read waits, in seconds: a backbone loading, one forward pass, one acknowledgement.
HANDSHAKE_TIMEOUT_S = 60.0
INFER_TIMEOUT_S = 120.0
RESET_TIMEOUT_S = 10.0
# A readiness poll answers between heartbeats, so it waits far less than a handshake a caller committed to.
READY_PROBE_TIMEOUT_S = 5.0


class RoboarenaClient:
    """One connection to a roboarena server, and the msgpack frames it carries."""

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self._address = roboarena_wire.RoboarenaAddress(host, port)
        self._wire = registry.client_wire(roboarena_wire.RoboarenaClientWire.NAME)
        self._connection: wire.ClientConnection | None = None
        self._server_config: dict[str, Any] | None = None

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
    def server_config(self) -> dict[str, Any]:
        """The ``PolicyServerConfig`` this backend announced on connect."""
        if self._server_config is None:
            raise RuntimeError('Not connected: the server announces its config on connect')
        return self._server_config

    def is_ready(self) -> bool:
        """Whether the server announces itself."""
        return self._wire.probe(self._address, None, READY_PROBE_TIMEOUT_S) is None

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """The action chunk the server answers ``observation`` with."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        try:
            self._connection.send(serialize({**observation, ENDPOINT: INFER}))
            answer = self._connection.recv(timeout=INFER_TIMEOUT_S)
        except BaseException:
            # A reply that arrives after this read gave up stays queued, and the next inference reads it
            # as its own: the arm would run a chunk computed for an observation it has moved on from.
            self.close()
            raise
        return deserialize(answer)

    def reset(self, session_id: str | None = None) -> None:
        """End the server's history for ``session_id``, and read the acknowledgement off the connection."""
        if self._connection is None:
            return
        frame: dict[str, Any] = {ENDPOINT: RESET}
        if session_id is not None:
            frame[SESSION_ID] = session_id
        self._connection.send(serialize(frame))
        try:
            self._connection.recv(timeout=RESET_TIMEOUT_S)
        except wire.PeerDisconnected as e:
            # The bundled backend acknowledges a reset in a text frame, which the wire reports as the
            # peer's own word. A session ends after its reset either way, so both readings end it here.
            logger.debug(f'roboarena reset answered: {e}')

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
