"""A client for a roboarena server, over the wire that carries its frames.

Roboarena is a cross-vendor protocol. Two servers here speak it: the DreamZero subprocess this repository
launches, and a partner's own server a rig dials. The frames are msgpack, which the wire package does not
depend on, so the framing lives here and the wire carries bytes.
"""

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

# How long a read waits, in seconds. The handshake covers a backbone that loads on connect; the inference
# covers one forward pass; the reset covers an acknowledgement the server sends at once.
HANDSHAKE_TIMEOUT_S = 60.0
INFER_TIMEOUT_S = 120.0
RESET_TIMEOUT_S = 10.0


class RoboarenaClient:
    """One connection to a roboarena server, and the msgpack frames it carries.

    The server announces its ``PolicyServerConfig`` as the first frame, which states the observation keys it
    wants, the geometry it wants them at, and whether it tracks sessions.
    """

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
            # A connection left behind here is the one the next episode infers over, which turns one
            # transient handshake failure into a second.
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
        """Whether the server announces itself, which is the readiness the protocol carries."""
        return self._wire.probe(self._address, None, HANDSHAKE_TIMEOUT_S) is None

    def infer(self, observation: Mapping[str, Any]) -> Any:
        """The action chunk the server answers ``observation`` with."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        self._connection.send(serialize({**observation, ENDPOINT: INFER}))
        return deserialize(self._connection.recv(timeout=INFER_TIMEOUT_S))

    def reset(self, session_id: str | None = None) -> None:
        """End the server's history for ``session_id``, and read the acknowledgement it answers with."""
        if self._connection is None:
            return
        frame: dict[str, Any] = {ENDPOINT: RESET}
        if session_id is not None:
            frame[SESSION_ID] = session_id
        self._connection.send(serialize(frame))
        self._connection.recv(timeout=RESET_TIMEOUT_S)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
