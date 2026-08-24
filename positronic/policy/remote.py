import collections.abc as cabc
import time
from typing import Any

import numpy as np
import pos3

from positronic import keys, telemetry, telemetry_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Answer, Layer, Policy, Runtime, Session
from .recording import Recorder
from .spec import from_spec

# The name the wire round-trip is served under; a policy whose sessions are ``RemoteSession``s declares it.
INFER = 'infer'


def round_trip(ws_session: InferenceSession, obs: dict[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span. ``finally`` times a raising one too."""
    infer_start_ns = time.time_ns()
    try:
        return ws_session.infer(obs)
    finally:
        telemetry.record_span(telemetry_keys.SPAN_POLICY_INFER, infer_start_ns, time.time_ns())


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server.

    One round-trip is in flight at a time: the call that starts it, and every call until it answers, return
    ``None``; the call that finds it answered returns its trajectory, or drops it after a ``cancel``.

    ``compress_images`` comes from what the server declared (see ``RemoteMarker``).
    """

    def __init__(self, ws_session: InferenceSession, rt: Runtime | None, compress_images: bool = False):
        self._session = ws_session
        self._rt = rt
        self._compress_images = compress_images
        self._answer: Answer | None = None
        self._cancelled = False

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
        """Starts a round-trip on ``obs`` when none is in flight, and answers the trajectory of one that
        has come back.

        Single-action server responses are wrapped into a 1-element list to honor
        the ``Session.__call__`` contract (``list[dict] | None``).
        """
        if self._rt is None:
            raise ValueError(
                'RemoteSession needs a runtime to run inference: pass rt to new_session. The harness '
                'supplies it; a direct RemotePolicy.new_session() outside the harness must too.'
            )
        if self._answer is None:
            # Preparation stays the session's own work: JPEG-encoding a stack of HD frames is client-side, and
            # folding it into the function would inflate the inference percentiles.
            self._answer = self._rt.fns[INFER](self._session, self._prepare_obs(obs))
            return None
        if not self._answer.done():
            return None
        answer, self._answer = self._answer, None
        # Read before anything else is decided: ``result`` re-raises what the round-trip raised, so a call
        # that failed is heard whether its chunk is still wanted or not.
        result = answer.result()
        if self._cancelled:
            self._cancelled = False
            return None
        return [result] if isinstance(result, dict) else result

    def cancel(self):
        # The chunk in flight was planned for a world the cancel says is gone, so the round-trip is read for
        # its failure and its chunk thrown away.
        self._cancelled = self._answer is not None

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({keys.TYPE: 'remote', keys.SERVER: self._session.metadata})

    def close(self):
        assert self._answer is None or self._answer.done(), (
            'close the runtime serving this session first: the round-trip in flight is talking over the '
            'websocket this closes'
        )
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

    def new_session(self, context=None, now=None, rt=None) -> RemoteSession:
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

    def new_session(self, context=None, now=None, rt=None) -> Session:
        return self._policy().new_session(context, now, rt)

    @property
    def functions(self) -> cabc.Mapping[str, cabc.Callable[..., Any]]:
        return self._policy().functions

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy().meta

    def close(self):
        self._endpoint.close()
