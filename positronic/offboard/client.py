import logging
from typing import Any

from websockets.sync.client import connect
from websockets.sync.connection import Connection

from .serialisation import deserialise, serialise

logger = logging.getLogger(__name__)


class InferenceSession:
    def __init__(self, websocket: Connection):
        self._websocket = websocket
        self._metadata = None

    def _handshake(self) -> dict[str, Any]:
        """Receive metadata from the server immediately after connection."""
        try:
            response = deserialise(self._websocket.recv())
            if 'error' in response:
                raise RuntimeError(f'Server error: {response["error"]}')
            return response['meta']
        except Exception as e:
            logger.error(f'Failed to handshake with server: {e}')
            raise

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._handshake()
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Send an observation and get an action.

        Both `obs` and the returned action must be wire-serializable: plain-data containers and
        scalars, plus numeric numpy arrays/scalars. Do not pass arbitrary Python objects.
        """
        if self._metadata is None:
            self._metadata = self._handshake()

        self._websocket.send(serialise(obs))
        response = deserialise(self._websocket.recv())

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')

        return response['result']

    def close(self):
        self._websocket.close()


class InferenceClient:
    def __init__(self, host: str, port: int):
        self.uri = f'ws://{host}:{port}'

    def new_session(self) -> InferenceSession:
        """
        Creates a new inference session.
        """
        return InferenceSession(connect(self.uri))
