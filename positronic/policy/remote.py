import collections.abc as cabc
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Policy, Session


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server."""

    def __init__(self, ws_session: InferenceSession, resize: int | None, compress_images: bool = False):
        self._session = ws_session
        self._resize = resize
        self._compress_images = compress_images
        self._image_sizes: dict[str, tuple[int, int]] = {}
        self._default_image_size: tuple[int, int] | None = None

        sizes = ws_session.metadata.get('image_sizes')
        if isinstance(sizes, dict):
            self._image_sizes = {k: tuple(v) for k, v in sizes.items()}
        elif isinstance(sizes, tuple | list):
            self._default_image_size = tuple(sizes)

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
        return RemoteSession._resize_to(image, int(w * scale), int(h * scale))

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

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Forwards the observation to the remote server and returns the action trajectory.

        Command reconstruction is handled transparently by the deserialization layer.
        Single-action server responses are wrapped into a 1-element list to honor
        the ``Session.__call__`` contract (``list[dict] | None``).
        """
        result = self._session.infer(self._prepare_obs(obs))
        if isinstance(result, dict):
            return [result]
        return result

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self._session.metadata})

    def close(self):
        self._session.close()


class RemotePolicy(Policy):
    """Policy that creates sessions forwarding observations to a remote inference server.

    Images are resized before sending to reduce bandwidth. The server reports
    expected sizes via ``image_sizes`` in its metadata (see ``Codec.meta``).
    The ``resize`` parameter acts as a fallback when the server does not report
    sizes. Server-reported sizes always take precedence.

    ``headers`` / ``secure`` are forwarded to the underlying ``InferenceClient``
    for authenticated / TLS-fronted endpoints (e.g. Modal, behind a reverse proxy).
    """

    def __init__(
        self,
        host: str,
        port: int,
        resize: int | None = None,
        model_id: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        secure: bool = False,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        compress_images: bool = False,
    ):
        self._client = InferenceClient(host, port, headers=headers, secure=secure)
        self._resize = resize
        self._model_id = model_id
        self._infer_timeout = infer_timeout
        self._compress_images = compress_images
        # Server metadata cached after the first session is created or `meta`
        # is read. Needed so consumers like ``SampledPolicy._get_keys`` see
        # ``server.checkpoint_path`` etc. before any session exists.
        self._server_meta: dict[str, Any] | None = None

    def _ensure_server_meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            ws_session = self._client.new_session(model_id=self._model_id, infer_timeout=self._infer_timeout)
            try:
                self._server_meta = dict(ws_session.metadata)
            finally:
                ws_session.close()
        return self._server_meta

    def new_session(self, context=None) -> RemoteSession:
        ws_session = self._client.new_session(model_id=self._model_id, infer_timeout=self._infer_timeout)
        if self._server_meta is None:
            self._server_meta = dict(ws_session.metadata)
        return RemoteSession(ws_session, self._resize, compress_images=self._compress_images)

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self._ensure_server_meta()})

    def close(self):
        self._client = None
