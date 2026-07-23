from typing import Any

from positronic_client.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession

from positronic.utils import flatten_dict
from positronic.utils.serialization import deserialise, serialise

from .base import Policy, Session


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server."""

    def __init__(self, ws_session: InferenceSession):
        self._session = ws_session

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Forwards the observation to the remote server and returns the action trajectory.

        Command reconstruction is handled transparently by the deserialization layer.
        Single-action server responses are wrapped into a 1-element list to honor
        the ``Session.__call__`` contract (``list[dict] | None``).
        """
        result = self._session.infer(obs)
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

    The underlying ``InferenceSession`` downsizes images before sending to reduce bandwidth, driven by the
    ``image_sizes`` the server reports in its metadata (see ``Codec.meta``); ``resize`` is the fallback when the
    server does not report sizes. Sessions speak positronic's wire dialect, so roboarm commands round-trip as
    objects.

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
        self._client = InferenceClient(
            host,
            port,
            headers=headers,
            secure=secure,
            resize=resize,
            compress_images=compress_images,
            serialise=serialise,
            deserialise=deserialise,
        )
        self._model_id = model_id
        self._infer_timeout = infer_timeout
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
        return RemoteSession(ws_session)

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self._ensure_server_meta()})

    def close(self):
        self._client = None
