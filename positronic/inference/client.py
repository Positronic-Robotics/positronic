import contextlib
import logging
from collections.abc import Generator
from typing import Any

import ormsgpack
from websockets.sync.client import connect
from websockets.sync.connection import Connection

logger = logging.getLogger(__name__)


class InferenceSession:
    def __init__(self, websocket: Connection):
        self._websocket = websocket
        self._metadata = None

    def _handshake(self) -> dict[str, Any]:
        """Receive metadata from the server immediately after connection."""
        try:
            metadata_packed = self._websocket.recv()
            response = ormsgpack.unpackb(metadata_packed)
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
        """
        if self._metadata is None:
            self._metadata = self._handshake()

        self._websocket.send(ormsgpack.packb(obs, option=ormsgpack.OPT_SERIALIZE_NUMPY))
        response = ormsgpack.unpackb(self._websocket.recv())

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')

        return response['result']


class InferenceClient:
    def __init__(self, host: str, port: int):
        self.uri = f'ws://{host}:{port}'

    @contextlib.contextmanager
    def start_session(self) -> Generator[InferenceSession, None, None]:
        """
        Starts a new inference session.
        """
        with connect(self.uri) as websocket:
            session = InferenceSession(websocket)
            yield session
