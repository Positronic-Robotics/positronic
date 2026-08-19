import collections.abc as cabc
import time
from typing import Any

import numpy as np
import pos3

from positronic import keys, telemetry, telemetry_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Policy, PolicyWrapper, Session
from .recording import Recorder
from .spec import from_spec


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server.

    ``compress_images`` comes from what the server declared (see ``RemoteMarker``).
    """

    def __init__(self, ws_session: InferenceSession, compress_images: bool = False):
        self._session = ws_session
        self._compress_images = compress_images

    def _prepare_obs(self, obs: cabc.Mapping[str, Any]) -> dict[str, Any]:
        if not self._compress_images:
            return dict(obs)
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

    def __call__(self, obs: cabc.Mapping[str, Any]) -> list[dict[str, Any]] | None:
        """Forwards the observation to the remote server and returns the action trajectory.

        Single-action server responses are wrapped into a 1-element list to honor
        the ``Session.__call__`` contract (``list[dict] | None``).
        """
        # Timed from after preparation: JPEG-encoding a stack of HD frames is client-side work, and folding it
        # into the round-trip would inflate the inference percentiles. ``finally`` times a raising one too.
        prepared = self._prepare_obs(obs)
        infer_start_ns = time.time_ns()
        try:
            result = self._session.infer(prepared)
        finally:
            telemetry.record_span(telemetry_keys.SPAN_POLICY_INFER, infer_start_ns, time.time_ns())
        if isinstance(result, dict):
            return [result]
        return result

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

    def server_meta(self, ready_deadline: float | None = None) -> dict[str, Any]:
        if self._server_meta is None:
            ws_session = self._client.new_session(ready_deadline)
            try:
                self._server_meta = dict(ws_session.metadata)
            finally:
                ws_session.close()
        return self._server_meta

    def wait_ready(self, timeout: float) -> None:
        """Wait for this server to report itself ready, bounded, or raise naming it and its last state.

        The wait IS the handshake: metadata arrives in the ``ready`` frame and is kept, so resolving
        the stack later costs no second connection.
        """
        self.server_meta(ready_deadline=time.monotonic() + timeout)

    def new_session(self, context=None, now=None) -> RemoteSession:
        compress = bool(self.server_meta().get(keys.COMPRESS_IMAGES))
        ws_session = self._client.new_session()
        return RemoteSession(ws_session, compress_images=compress)

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
    ``remote`` marker. The declared wrappers are built here, once, and every session runs through
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

    def _resolve_stack(self) -> PolicyWrapper:
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

    def new_session(self, context=None, now=None) -> Session:
        return self._policy().new_session(context, now)

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy().meta

    def wait_ready(self, timeout: float) -> None:
        """Ready means both: the server reports itself servable AND its declared stack builds here.

        A ready server can still declare a stack this rig cannot construct; `_resolve_stack` raises.
        """
        self._endpoint.wait_ready(timeout)
        self._policy()

    def close(self):
        self._endpoint.close()
