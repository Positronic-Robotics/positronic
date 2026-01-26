import logging
from typing import Any

import httpx
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from positronic.utils.serialization import deserialise, serialise

logger = logging.getLogger(__name__)


class InferenceSession:
    def __init__(self, websocket: Connection):
        self._websocket = websocket
        self._metadata = self._handshake()

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        Args:
            timeout_per_message: Timeout for each individual message (default: 30s).
                               Server must send updates at least this frequently.
        """
        try:
            while True:
                self._websocket.socket.settimeout(timeout_per_message)
                response = deserialise(self._websocket.recv())
                status = response.get('status')

                if status == 'ready':
                    return response['meta']

                if status in ('waiting', 'loading'):
                    message = response.get('message', status)
                    logger.info(f'Server status: [{status}] {message}')
                    continue

                if status == 'error' or 'error' in response:
                    raise RuntimeError(f'Server error: {response.get("error", "Unknown error")}')

                raise RuntimeError(f'Unexpected server response: {response}')

        except TimeoutError:
            raise TimeoutError(
                f'Server did not send status update within {timeout_per_message}s. '
                f'Server may have crashed or model loading is taking too long without progress updates.'
            ) from None
        finally:
            self._websocket.socket.settimeout(None)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Send an observation and get an action.

        Both `obs` and the returned action must be wire-serializable: plain-data containers and
        scalars, plus numeric numpy arrays/scalars. Do not pass arbitrary Python objects.
        """
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)

        self._websocket.send(serialised)
        response = deserialise(self._websocket.recv())
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')

        return response['result']

    def close(self):
        self._websocket.close()


class InferenceClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_uri = f'ws://{host}:{port}/api/v1/session'
        self.api_url = f'http://{host}:{port}/api/v1'

    def new_session(self, model_id: str | None = None, open_timeout: float = 10.0) -> InferenceSession:
        """
        Creates a new inference session.

        Args:
            model_id: Optional model ID to connect to
            open_timeout: Timeout for initial WebSocket connection (default: 10s).
                        This only covers TCP/HTTP handshake, not model loading.
                        Model loading timeout is controlled by per-message timeout in handshake.
        """
        uri = self.base_uri if model_id is None else f'{self.base_uri}/{model_id}'
        return InferenceSession(connect(uri, open_timeout=open_timeout))

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models')
        response.raise_for_status()
        data = response.json()
        return data['models']
