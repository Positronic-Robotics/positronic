import collections.abc as cabc
import time
from typing import Any

import numpy as np
import pos3

from positronic import keys, telemetry, telemetry_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Answer, AnySession, ChunkSession, Layer, Policy, Runtime
from .recording import Recorder
from .spec import from_spec

# The name the wire round trip is served under. A policy whose sessions are ``RemoteSession``s declares it.
INFER = 'infer'


def _prepare_value(value: Any) -> Any:
    # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
    if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's websocket message cap.
        return encode_jpeg(value)
    if isinstance(value, cabc.Mapping):
        return {k: _prepare_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_prepare_value(v) for v in value)
    return value


def _prepare_obs(obs: cabc.Mapping[str, Any], compress_images: bool) -> dict[str, Any]:
    if not compress_images:
        return dict(obs)
    return {key: _prepare_value(value) for key, value in obs.items()}


def round_trip(
    ws_session: InferenceSession, obs: cabc.Mapping[str, Any], compress_images: bool
) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span.

    The observation is prepared here rather than in the session, because a JPEG encode of an HD frame
    stack must not run on the thread that calls the session. The span starts after it, because that
    encode is not inference.
    """
    prepared = _prepare_obs(obs, compress_images)
    infer_start_ns = time.time_ns()
    try:
        return ws_session.infer(prepared)
    finally:
        telemetry.record_span(telemetry_keys.SPAN_POLICY_INFER, infer_start_ns, time.time_ns())


class RemoteSession(ChunkSession):
    """Per-episode session that forwards observations to a remote inference server.

    ``compress_images`` comes from what the server declared (see ``RemoteMarker``).
    """

    def __init__(self, ws_session: InferenceSession, rt: Runtime, compress_images: bool = False):
        self._session = ws_session
        self._rt = rt
        self._compress_images = compress_images

    def __call__(self, obs: cabc.Mapping[str, Any], time_ns: int) -> Answer:
        """One round trip to the server, and the caller's handle on the trajectory it brings back.

        A server answer of one action becomes a 1-element list, which is the form a chunk takes.
        """
        answer = self._rt.fns[INFER](self._session, obs, self._compress_images)
        return answer.map(lambda result: [result] if isinstance(result, dict) else result)

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({keys.TYPE: 'remote', keys.SERVER: self._session.metadata})

    def close(self):
        self._session.close()


class _Endpoint(Policy):
    """The wire connection to one inference server: sessions forward observations under the border's settings.

    ``InferenceClient`` reads the server, the model, and the session params off the URL.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None, infer_timeout: float):
        self._client = InferenceClient(url, headers=headers, infer_timeout=infer_timeout)
        # Filled on first contact, via a throwaway session if ``meta`` is read before any real one exists.
        self._server_meta: dict[str, Any] | None = None

    def server_meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            ws_session = self._client.new_session()
            try:
                self._server_meta = dict(ws_session.metadata)
            finally:
                ws_session.close()
        return self._server_meta

    def new_session(self, context=None, rt=None) -> RemoteSession:
        if rt is None:
            raise ValueError('A remote session runs its inference on a runtime: pass rt to new_session.')
        compress = bool(self.server_meta().get(keys.COMPRESS_IMAGES))
        ws_session = self._client.new_session()
        return RemoteSession(ws_session, rt, compress_images=compress)

    @property
    def functions(self) -> cabc.Mapping[str, cabc.Callable[..., Any]]:
        return {INFER: round_trip}

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({keys.TYPE: 'remote', keys.SERVER: self.server_meta()})

    def close(self):
        self._client = None


class RemotePolicy(Policy):
    """Policy running against a remote inference server, owning the stack in front of the connection.

    One URL names the server, the model, and the session params — see ``InferenceClient`` for the forms
    it takes. ``headers`` stay their own argument: they carry credentials, which a URL that gets pasted
    around should not.

    The server's ``ready`` handshake declares the local half of its policy pipeline (the
    ``local_stack`` spec — see ``positronic.policy.spec``) along with the wire settings of the
    ``remote`` marker. The declared layers are built here, once, and every session runs through
    them; a handshake that declares no stack is an error.

    ``recording_dir`` taps the raw and wire boundaries around the stack.
    """

    def __init__(
        self,
        url: str,
        *,
        recording_dir: str | None = None,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ):
        self._endpoint = _Endpoint(url, headers=headers, infer_timeout=infer_timeout)
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None
        self._stacked: Policy | None = None

    def _resolve_stack(self) -> Layer:
        meta = self._endpoint.server_meta()
        version = meta.get(keys.POSITRONIC_VERSION, 'unknown')
        declared = meta.get(keys.LOCAL_STACK)
        try:
            stack = from_spec(declared) if declared is not None else None
        except Exception as e:
            raise ValueError(f'Cannot build the server-declared local stack (server positronic {version})') from e
        if stack is None:
            raise ValueError(
                f'Server declares no rig-side stack (server positronic {version}); the rig runs what the '
                f'handshake declares and nothing else, so serve it from a pipeline that declares one'
            )
        return stack

    def _policy(self) -> Policy:
        if self._stacked is None:
            stack = self._resolve_stack()
            if self._recording_dir is not None:
                rec = Recorder(self._recording_dir)
                stack = rec.tap('raw') | stack | rec.tap('server')
            self._stacked = stack.wrap(self._endpoint)
        return self._stacked

    def new_session(self, context=None, rt=None) -> AnySession:
        return self._policy().new_session(context, rt)

    @property
    def functions(self) -> cabc.Mapping[str, cabc.Callable[..., Any]]:
        return self._policy().functions

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy().meta

    def close(self):
        self._endpoint.close()
