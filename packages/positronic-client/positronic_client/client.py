import collections.abc as cabc
import logging
import ssl
import time
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
from PIL import Image as PilImage
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from .serialization import deserialise, encode_jpeg, serialise

logger = logging.getLogger(__name__)

# Only the checkpoint pinned at server startup is pre-warmed; a session that requests any other model loads it
# cold, so its first ``infer`` can include the backend's JAX compilation. Bound each ``recv`` generously enough to
# outlast that compile (still surfacing a truly stalled/half-open connection), and let callers override per use.
DEFAULT_INFER_TIMEOUT = 180.0


class InferenceSession:
    """One inference session: streams observations, receives action chunks.

    Images are downsized before sending to reduce bandwidth: the server reports the sizes its codec expects via
    ``image_sizes`` in its metadata (see ``Codec.meta``), and every frame is fit into the reported size
    (aspect-preserving, never upscaled). The ``resize`` parameter acts as a fallback when the server does not
    report sizes; server-reported sizes always take precedence.

    This downsizing is a property of the WIRE, applied to every session — including a localhost server, where it
    still shrinks the msgpack payload on both ends while leaving the codec's output unchanged (the codec resizes
    to the same advertised target regardless; the fit just moves that resample before serialization). In-process
    inference never constructs a session, so nothing local-to-the-process is ever touched. A client-side codec
    whose output is already model-sized passes through untouched (fit never upscales).
    """

    def __init__(
        self,
        websocket: Connection,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        *,
        resize: int | None = None,
        compress_images: bool = False,
        serialise: Callable[[Any], bytes] = serialise,
        deserialise: Callable[[bytes], Any] = deserialise,
    ):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._resize = resize
        self._compress_images = compress_images
        self._serialise = serialise
        self._deserialise = deserialise
        self._metadata = self._handshake()

        self._image_sizes: dict[str, tuple[int, int]] = {}
        self._default_image_size: tuple[int, int] | None = None
        sizes = self._metadata.get('image_sizes')
        if isinstance(sizes, dict):
            self._image_sizes = {k: tuple(v) for k, v in sizes.items()}
        elif isinstance(sizes, tuple | list):
            self._default_image_size = tuple(sizes)

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        Args:
            timeout_per_message: Timeout for each individual message (default: 30s).
                               Server must send updates at least this frequently.
        """
        try:
            while True:
                response = self._deserialise(self._websocket.recv(timeout=timeout_per_message))
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

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = image.shape[:2]
        if w == width and h == height:
            return image
        return np.array(PilImage.fromarray(image).resize((width, height), resample=PilImage.Resampling.BILINEAR))

    @staticmethod
    def _fit(image: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, tw / w, th / h)
        return InferenceSession._resize_to(image, int(w * scale), int(h * scale))

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {key: self._prepare_value(key, value) for key, value in obs.items()}

    def _prepare_value(self, key: str, value: Any) -> Any:
        # Client-side codecs (e.g. GR00T) nest images inside dicts/lists, so recurse to reach every
        # image array rather than scanning the top level alone.
        if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
            return self._prepare_image(key, value)
        if isinstance(value, cabc.Mapping):
            return {k: self._prepare_value(k, v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._prepare_value(key, v) for v in value)
        return value

    def _prepare_image(self, key: str, image: np.ndarray) -> np.ndarray | dict[bytes, Any]:
        # Resize single RGB frames and temporal stacks of them alike (TemporalStack emits a
        # (T, H, W, 3) stack), so a stack of hd720 frames isn't shipped full-resolution.
        target = self._image_sizes.get(key, self._default_image_size)
        r = self._resize or 0
        tw, th = target or (r, r)
        if tw > 0 and th > 0:
            image = np.stack([self._fit(f, tw, th) for f in image]) if image.ndim == 4 else self._fit(image, tw, th)
        # Optionally JPEG-compress before sending: a raw HD frame — and especially a (T, H, W, 3)
        # stack — can exceed the ~2 MB websocket message cap of a Modal-fronted endpoint. Off by default.
        if self._compress_images:
            image = encode_jpeg(image)
        return image

    def infer(self, obs: dict[str, Any]) -> Any:
        """
        Send an observation and get an action.

        Both `obs` and the returned action must be wire-serializable: plain-data containers and
        scalars, plus numeric numpy arrays/scalars. Do not pass arbitrary Python objects.
        """
        serialised = self._serialise(self._prepare_obs(obs))
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)

        self._websocket.send(serialised)
        try:
            response = self._deserialise(self._websocket.recv(timeout=self._infer_timeout))
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._websocket.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')

        return response['result']

    def close(self):
        self._websocket.close()


class InferenceClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        model_id: str | None = None,
        headers: dict[str, str] | None = None,
        secure: bool = False,
        resize: int | None = None,
        compress_images: bool = False,
        serialise: Callable[[Any], bytes] = serialise,
        deserialise: Callable[[bytes], Any] = deserialise,
    ):
        self.host = host
        self.port = port
        self.headers = dict(headers) if headers else None
        self._model_id = model_id
        self._resize = resize
        self._compress_images = compress_images
        self._serialise = serialise
        self._deserialise = deserialise
        ws_scheme = 'wss' if secure else 'ws'
        http_scheme = 'https' if secure else 'http'
        default_port = 443 if secure else 80
        netloc = host if port == default_port else f'{host}:{port}'
        self.base_uri = f'{ws_scheme}://{netloc}/api/v1/session'
        self.api_url = f'{http_scheme}://{netloc}/api/v1'

    def new_session(
        self,
        model_id: str | None = None,
        open_timeout: float = 10.0,
        connect_deadline: float = 900.0,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ) -> InferenceSession:
        """
        Creates a new inference session.

        Args:
            model_id: Optional model ID to connect to; falls back to the client's pinned ``model_id``.
            open_timeout: Timeout for initial WebSocket connection (default: 10s).
                        This only covers TCP/HTTP handshake, not model loading.
                        Model loading timeout is controlled by per-message timeout in handshake.
        """
        model_id = model_id if model_id is not None else self._model_id
        uri = self.base_uri if model_id is None else f'{self.base_uri}/{model_id}'
        connect_kwargs: dict[str, object] = {'open_timeout': open_timeout}
        if self.headers:
            connect_kwargs['additional_headers'] = self.headers

        deadline = time.monotonic() + connect_deadline
        backoff = 1.0
        while True:
            ws = None
            try:
                ws = connect(uri, **connect_kwargs)
                return InferenceSession(
                    ws,
                    infer_timeout=infer_timeout,
                    resize=self._resize,
                    compress_images=self._compress_images,
                    serialise=self._serialise,
                    deserialise=self._deserialise,
                )
            # ``SSLCertVerificationError`` is an ``ssl.SSLError``, but a bad certificate is permanent
            # misconfiguration, not a cold start — surface it immediately instead of retrying to the deadline.
            except ssl.SSLCertVerificationError as e:
                raise type(e)(f'{e} (connecting to {self.host}:{self.port})') from e
            # A cold backend fails before the session is ready in several ways: the connect times out, the edge
            # resets TLS (``SSLError``), it rejects or drops the HTTP upgrade (``InvalidHandshake`` — e.g. a
            # 502/503 while the backend boots), or it accepts the socket and then drops or stalls the status
            # handshake inside ``InferenceSession`` (``ConnectionClosed``/``TimeoutError``). All mean "not ready
            # yet", so retry within the deadline instead of letting one kill the run.
            except (TimeoutError, ssl.SSLError, ConnectionClosed, InvalidHandshake) as e:
                if ws is not None:
                    ws.close()
                # A non-101 upgrade response only means "not ready" when it's a 5xx or 429; any other status
                # (401/403/404, …) is permanent misconfiguration and surfaces immediately.
                if isinstance(e, InvalidStatus) and not (
                    e.response.status_code >= 500 or e.response.status_code == 429
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{e} (connecting to {self.host}:{self.port})') from e
                logger.info('Server not ready (cold start?): %s; retrying in %.0fs', e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.host}:{self.port})') from e

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
