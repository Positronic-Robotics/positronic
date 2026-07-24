import collections.abc as cabc
import logging
import urllib.parse
from typing import Any

import numpy as np
import pos3

from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Policy, PolicyWrapper, Session
from .recording import Recorder
from .spec import from_spec
from .wrappers import ChunkedSchedule

logger = logging.getLogger(__name__)


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server."""

    def __init__(self, ws_session: InferenceSession, compress_images: bool = False):
        self._session = ws_session
        self._compress_images = compress_images

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._compress_images:
            return obs
        return {key: self._prepare_value(key, value) for key, value in obs.items()}

    def _prepare_value(self, key: str, value: Any) -> Any:
        # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
        if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
            # JPEG-compress before sending: a raw HD frame — and especially a (T, H, W, 3) stack — can
            # exceed the ~2 MB websocket message cap of a Modal-fronted endpoint.
            return encode_jpeg(value)
        if isinstance(value, cabc.Mapping):
            return {k: self._prepare_value(k, v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._prepare_value(key, v) for v in value)
        return value

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


class _Endpoint(Policy):
    """The wire connection to one inference server: sessions forward observations as-is.

    Image size is the declared stack's business — a server that wants smaller frames declares
    ``RestrictImageSize`` in front of the marker. ``compress_images`` is this endpoint's own, because
    the message-size cap belongs to the connection rather than to the policy.

    ``headers`` / ``secure`` / ``params`` are forwarded to the underlying
    ``InferenceClient`` — auth / TLS for fronted endpoints (e.g. Modal, behind a
    reverse proxy) and session query parameters the server applies as overrides
    to its pipe config.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_id: str | None,
        *,
        headers: dict[str, str] | None,
        secure: bool,
        params: dict[str, Any] | str | None,
        infer_timeout: float,
        compress_images: bool,
    ):
        self._client = InferenceClient(host, port, headers=headers, secure=secure, params=params)
        self._model_id = model_id
        self._infer_timeout = infer_timeout
        self._compress_images = compress_images
        # Fetched lazily, via a throwaway session when ``meta`` is read before any session exists.
        self._server_meta: dict[str, Any] | None = None

    def server_meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            ws_session = self._client.new_session(model_id=self._model_id, infer_timeout=self._infer_timeout)
            try:
                self._server_meta = dict(ws_session.metadata)
            finally:
                ws_session.close()
        return self._server_meta

    def new_session(self, context=None, now=None) -> RemoteSession:
        ws_session = self._client.new_session(model_id=self._model_id, infer_timeout=self._infer_timeout)
        if self._server_meta is None:
            self._server_meta = dict(ws_session.metadata)
        return RemoteSession(ws_session, compress_images=self._compress_images)

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self.server_meta()})

    def close(self):
        self._client = None


class RemotePolicy(Policy):
    """Policy running against a remote inference server, owning the stack in front of the connection.

    The server's ``ready`` handshake may declare the local half of its policy pipeline (the
    ``local_stack`` spec — see ``positronic.policy.spec``); the declared wrappers are built here,
    once, and every session runs through them. ``local`` is the operator's bypass: when set, the
    declaration is ignored (and logged) and the given stack is used instead. When the server
    declares nothing and no override is given, the standard ``ChunkedSchedule`` applies.

    ``recording_dir`` places ``Recorder`` taps around the stack, recording the raw and wire
    boundaries.

    ``params`` become query parameters on every session URL; the server applies them as
    overrides to its pipe config, so the declared ``local_stack`` reflects them too. A dict is
    JSON-encoded per value; a str is a ready query string forwarded verbatim.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_id: str | None = None,
        *,
        local: PolicyWrapper | None = None,
        recording_dir: str | None = None,
        headers: dict[str, str] | None = None,
        secure: bool = False,
        params: dict[str, Any] | str | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        compress_images: bool = False,
    ):
        self._endpoint = _Endpoint(
            host,
            port,
            model_id,
            headers=headers,
            secure=secure,
            params=params,
            infer_timeout=infer_timeout,
            compress_images=compress_images,
        )
        self._local = local
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None
        self._stacked: Policy | None = None

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        local: PolicyWrapper | None = None,
        recording_dir: str | None = None,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        compress_images: bool = False,
    ) -> 'RemotePolicy':
        """Build a RemotePolicy from one URL carrying host, port, model id, and session params.

        Accepted forms: ``host``, ``host:port``, and ``scheme://host[:port][/api/v1/session[/<model_id>]]``,
        each with an optional ``?query``. ``https``/``wss`` enable TLS (bare or ``http``/``ws`` forms don't);
        the port defaults to 443 for TLS schemes and 8000 otherwise; the query string is forwarded verbatim
        as session params, so the URL's literals reach the server exactly as written.
        """
        split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
        if split.scheme not in ('', 'http', 'ws', 'https', 'wss'):
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        if not split.hostname:
            raise ValueError(f'No host in {url!r}')
        path = split.path.rstrip('/')
        model_id = None
        if path and path != '/api/v1/session':
            prefix, _, model_id = path.rpartition('/')
            if prefix != '/api/v1/session' or not model_id:
                raise ValueError(f'Unexpected path {split.path!r} in {url!r}; expected /api/v1/session[/<model_id>]')
        secure = split.scheme in ('https', 'wss')
        # urlsplit strips the brackets an IPv6 host needs back in a netloc.
        host = f'[{split.hostname}]' if ':' in split.hostname else split.hostname
        return cls(
            host,
            split.port or (443 if secure else 8000),
            model_id,
            local=local,
            recording_dir=recording_dir,
            headers=headers,
            secure=secure,
            params=split.query or None,
            infer_timeout=infer_timeout,
            compress_images=compress_images,
        )

    def _resolve_stack(self) -> PolicyWrapper | None:
        declared = self._endpoint.server_meta().get('local_stack')
        if self._local is not None:
            if declared is not None:
                logger.info('Operator-supplied local stack bypasses the server declaration (ignored: %r)', declared)
            return self._local
        if declared is not None:
            try:
                return from_spec(declared)
            except Exception as e:
                version = self._endpoint.server_meta().get('positronic_version', 'unknown')
                raise ValueError(f'Cannot build the server-declared local stack (server positronic {version})') from e
        logger.info('Server declared no local stack; running the standard ChunkedSchedule')
        return ChunkedSchedule()

    def _policy(self) -> Policy:
        if self._stacked is None:
            stack = self._resolve_stack()
            if self._recording_dir is not None:
                rec = Recorder(self._recording_dir)
                if stack is None:
                    # With no stack the raw and wire boundaries coincide, so a single tap.
                    stack = rec.tap('raw')
                else:
                    stack = rec.tap('raw') | stack | rec.tap('server')
            self._stacked = stack.wrap(self._endpoint) if stack is not None else self._endpoint
        return self._stacked

    def new_session(self, context=None, now=None) -> Session:
        return self._policy().new_session(context, now)

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy().meta

    def close(self):
        self._endpoint.close()
