import collections.abc as cabc
import logging
from typing import Any

import numpy as np
import pos3

from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Policy, PolicyWrapper, Session
from .codec import RestrictImageSize
from .recording import Recorder
from .spec import from_spec
from .wrappers import ChunkedSchedule

logger = logging.getLogger(__name__)


def _operator_override(name: str, value: Any, declared: Any) -> bool:
    """Whether the operator's ``value`` stands in for a ``name`` the server did not declare.

    A server that declares its own ``name`` is the authority on it, so an override against one is a
    contradiction rather than a preference and raises.

    TODO(#514): drop the overrides and this helper once every server declares.
    """
    if value is None:
        return False
    if declared is not None:
        raise ValueError(
            f'--policy.{name} was given, but the server declares its own {name} ({declared!r}); drop the '
            f'override, or change the pipeline the server serves'
        )
    logger.warning('--policy.%s is deprecated; it applies only because the server declares no %s', name, name)
    return True


def _legacy_bound(sizes: Any) -> RestrictImageSize | None:
    """A wire bound read off the ``image_sizes`` a server reports, for one that declares no stack.

    A ``(width, height)`` pair bounds every image directly; a per-camera mapping collapses to the widest
    and tallest it names, so no camera is sent smaller than the server encodes it. A codec that encodes
    no images names no geometry, and so bounds nothing.

    TODO(#514): drop this and its caller once every server declares its own stack.
    """
    pairs = list(sizes.values()) if isinstance(sizes, cabc.Mapping) else [sizes]
    if not pairs:
        return None
    return RestrictImageSize(max(w for w, _ in pairs), max(h for _, h in pairs))


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
            # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's websocket message cap.
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

    ``InferenceClient`` reads the server, the model, and the session params off the URL.
    ``compress_images`` stands in for a server that declares no wire settings of its own.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None, infer_timeout: float, compress_images: bool | None):
        self._client = InferenceClient(url, headers=headers, infer_timeout=infer_timeout)
        self._compress_override = compress_images
        # Both filled on first contact: the metadata via a throwaway session if ``meta`` is read before any
        # real one exists, the compression flag from that metadata.
        self._server_meta: dict[str, Any] | None = None
        self._compress: bool | None = None

    def server_meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            ws_session = self._client.new_session()
            try:
                self._server_meta = dict(ws_session.metadata)
            finally:
                ws_session.close()
        return self._server_meta

    def _compression(self) -> bool:
        """Whether the rig JPEG-encodes frames: what the server declared, or the operator's stand-in."""
        if self._compress is None:
            declared = self.server_meta().get('compress_images')
            override = _operator_override('compress_images', self._compress_override, declared)
            self._compress = bool(self._compress_override if override else declared)
        return self._compress

    def new_session(self, context=None, now=None) -> RemoteSession:
        # Resolved before connecting, so a session that contradicts the declaration leaves no socket open.
        compress = self._compression()
        ws_session = self._client.new_session()
        return RemoteSession(ws_session, compress_images=compress)

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({'type': 'remote', 'server': self.server_meta()})

    def close(self):
        self._client = None


class RemotePolicy(Policy):
    """Policy running against a remote inference server, owning the stack in front of the connection.

    One URL names the server, the model, and the session params — see ``InferenceClient`` for the forms
    it takes. ``headers`` stay their own argument: they carry credentials, which a URL that gets pasted
    around should not.

    The server's ``ready`` handshake declares the local half of its policy pipeline (the
    ``local_stack`` spec — see ``positronic.policy.spec``) along with the wire settings of the
    ``remote`` marker. The declared wrappers are built here, once, and every session runs through
    them. A server that declares no stack gets the standard ``ChunkedSchedule`` — or the operator's
    ``local`` — with the ``image_sizes`` such a server reports bounding the wire either way.

    ``local`` and ``compress_images`` stand in for a server that declares neither — see
    ``_operator_override``. ``recording_dir`` taps the raw and wire boundaries around the stack.
    """

    def __init__(
        self,
        url: str,
        *,
        local: PolicyWrapper | None = None,
        recording_dir: str | None = None,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        compress_images: bool | None = None,
    ):
        self._endpoint = _Endpoint(url, headers=headers, infer_timeout=infer_timeout, compress_images=compress_images)
        self._local = local
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None
        self._stacked: Policy | None = None

    def _resolve_stack(self) -> PolicyWrapper | None:
        meta = self._endpoint.server_meta()
        declared = meta.get('local_stack')
        # Settles the operator's stack against the declaration before either is built.
        _operator_override('local', self._local, declared)
        if declared is not None:
            try:
                return from_spec(declared)
            except Exception as e:
                version = meta.get('positronic_version', 'unknown')
                raise ValueError(f'Cannot build the server-declared local stack (server positronic {version})') from e
        # The bound is a wire setting rather than part of the stack, so it outlives whichever stack runs.
        stack = ChunkedSchedule() if self._local is None else self._local
        bound = _legacy_bound(meta['image_sizes']) if 'image_sizes' in meta else None
        if bound is None:
            logger.info('Server declares no local stack and names no image geometry; frames go out unbounded')
            return stack
        logger.warning(
            'Server declares no local stack; bounding frames to %r from the image_sizes %r it reports',
            bound.to_spec()['args'],
            meta['image_sizes'],
        )
        return stack | bound

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
