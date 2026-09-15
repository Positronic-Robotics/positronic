import collections.abc as cabc
import logging
import time
from collections import deque
from typing import Any

import numpy as np
import pos3

from positronic import keys, telemetry, telemetry_keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.policy import keys as policy_keys
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Answer, DelegatingSession, Layer, Policy, Runtime, Session
from .layers import TemporalStack
from .recording import Recorder
from .spec import WIRE_LAYERS, from_spec

logger = logging.getLogger(__name__)

# The name the wire round trip is served under. A policy whose sessions are ``RemoteSession``s declares it.
INFER = 'infer'


def _prepare_value(value: Any) -> Any:
    # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
    if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's message cap.
        return encode_jpeg(value)
    if isinstance(value, cabc.Mapping):
        return {k: _prepare_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_prepare_value(v) for v in value)
    return value


def prepare_obs(obs: cabc.Mapping[str, Any], compress_images: bool) -> dict[str, Any]:
    if not compress_images:
        return dict(obs)
    return {key: _prepare_value(value) for key, value in obs.items()}


def round_trip(
    session: InferenceSession, obs: cabc.Mapping[str, Any], compress_images: bool
) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span.

    The observation is prepared here rather than in the session, because a JPEG encode of an HD frame
    stack must not run on the thread that calls the session. The span starts after it, because that
    encode is not inference.
    """
    with telemetry.span(telemetry_keys.SPAN_POLICY_PREPARE):
        prepared = prepare_obs(obs, compress_images)
    infer_start_ns = time.time_ns()
    try:
        return session.infer(prepared)
    finally:
        # The server's timing fields ride on the round-trip span under the ``served.`` prefix.
        served = {f'{telemetry_keys.ATTR_SERVED_PREFIX}{k}': v for k, v in session.served_timing.items()}
        telemetry.record_span(telemetry_keys.SPAN_POLICY_INFER, infer_start_ns, time.time_ns(), **served)


class RemoteSession(Session):
    """Per-episode session that forwards observations to a remote inference server.

    One round trip is in flight at a time. The call that starts it answers ``None``, and so does every
    call until the round trip comes back. The call that finds it answered returns its trajectory, or drops
    that trajectory after a ``cancel``.

    ``compress_images`` comes from what the server declared (see ``RemoteMarker``).
    """

    def __init__(self, session: InferenceSession, rt: Runtime, compress_images: bool = False):
        self._session = session
        self._rt = rt
        self._compress_images = compress_images
        self._answer: Answer | None = None
        self._cancelled = False

    def __call__(self, obs: cabc.Mapping[str, Any], time_ns: int) -> list[dict[str, Any]] | None:
        """The trajectory of a round trip that has come back, and ``None`` while one is in flight.

        A server answer of one action becomes a 1-element list, which is the form ``Session.__call__``
        returns.
        """
        if self._answer is None:
            self._answer = self._rt.fns[INFER](self._session, obs, self._compress_images)
            return None
        if not self._answer.done():
            return None
        answer, cancelled = self._answer, self._cancelled
        # The answer and the flag are cleared before the read, because ``result`` raises what the round
        # trip raised. A cancel then ends with the answer it was made against, and never drops the next
        # chunk.
        self._answer, self._cancelled = None, False
        result = answer.result()
        if cancelled:
            return None
        return [result] if isinstance(result, dict) else result

    def push_frame(self, key, obs_time_ns, value):
        # On the calling thread, and a JPEG encode with it: the frame must be on the wire before the
        # observation that names it, and the round trip runs off-thread.
        self._session.push_frame(key, obs_time_ns, _prepare_value(value) if self._compress_images else value)

    def cancel(self):
        # The cancel says the world the chunk applies to has gone. The session still reads the round trip
        # for its failure, and drops the chunk that comes with it.
        self._cancelled = self._answer is not None

    @property
    def meta(self) -> dict[str, Any]:
        return flatten_dict({policy_keys.TYPE: 'remote', policy_keys.SERVER: self._session.metadata})

    def close(self):
        in_flight = self._answer is not None and not self._answer.done()
        logger.info('RemoteSession.close: answer_in_flight=%s', in_flight)
        assert not in_flight, (
            'close the runtime serving this session first: the round trip in flight uses the connection this closes'
        )
        self._session.close()
        logger.info('RemoteSession.close: session closed')


class StreamedTemporalStack(Layer):
    """``TemporalStack`` against a server that declared ``stream_frames``: the frames go ahead, the ids follow.

    The stack the server assembles is the one ``TemporalStack`` would have sent: per offset, the latest
    frame at or before ``now + offset``. The frame for an offset is pushed as soon as the tick at or after
    ``trajectory_end + offset`` passes, so the server holds most of the stack before the chunk ends. At
    the request, whatever the sample picks that was not pushed goes ahead of it, and the observation
    carries the ids instead of the stack.
    """

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, keys_: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool):
            super().__init__(inner)
            self._keys = keys_
            self._offsets_ns = [int(round(off * 1e9)) for off in offsets_sec]
            self._pad_start = pad_start
            self._entries: deque[tuple[int, dict[str, Any]]] = deque()
            self._pushed: set[int] = set()
            self._trajectory_end_ns: int | None = None
            self._awaiting = False

        def _append(self, now: int, obs) -> None:
            if self._entries and self._entries[-1][0] == now:
                return
            self._entries.append((now, {k: np.array(obs[k]) for k in self._keys}))
            cutoff = now + min(self._offsets_ns)
            while len(self._entries) >= 2 and self._entries[1][0] <= cutoff:
                self._entries.popleft()
            self._pushed &= {t for t, _ in self._entries}

        def _at_or_before(self, target_ns: int) -> tuple[int, dict[str, Any]] | None:
            picked = None
            for t, values in self._entries:
                if t > target_ns:
                    break
                picked = (t, values)
            return picked

        def _sample(self, now: int) -> list[tuple[int, dict[str, Any]]]:
            picked = []
            for off in self._offsets_ns:
                entry = self._at_or_before(now + off)
                if entry is None:
                    if not self._pad_start:
                        continue
                    entry = self._entries[0]
                picked.append(entry)
            return picked

        def _push(self, t: int, values: dict[str, Any]) -> None:
            if t in self._pushed:
                return
            for k in self._keys:
                self._inner.push_frame(k, t, values[k])
            self._pushed.add(t)

        def __call__(self, obs, time_ns):
            now = int(obs[keys.OBS_TIME_NS])
            self._append(now, obs)
            if self._trajectory_end_ns is not None:
                for off in self._offsets_ns:
                    target = self._trajectory_end_ns + off
                    if now >= target:
                        entry = self._at_or_before(target) or self._entries[0]
                        self._push(*entry)
            fires = self._trajectory_end_ns is None or now >= self._trajectory_end_ns
            picked = self._sample(now)
            if fires and not self._awaiting:
                for entry in picked:
                    self._push(*entry)
                self._awaiting = True
            ids = [t for t, _ in picked]
            result = self._inner({**obs, **{k: {protocol.FRAME_IDS: ids} for k in self._keys}}, time_ns)
            if result is not None:
                self._awaiting = False
                self._trajectory_end_ns = int(round(result[-1][keys.ACTION_TIMESTAMP] * 1e9)) if result else None
            return result

        def cancel(self):
            self._entries.clear()
            self._pushed.clear()
            self._trajectory_end_ns = None
            self._awaiting = False
            super().cancel()

    WIRE_NAME = TemporalStack.WIRE_NAME

    def __init__(self, keys: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool = True):
        self._keys = tuple(keys)
        self._offsets_sec = tuple(offsets_sec)
        self._pad_start = pad_start

    def make_session(self, inner: Session):
        return StreamedTemporalStack._Session(inner, self._keys, self._offsets_sec, self._pad_start)


# The vocabulary a rig builds the declared stack from when the server takes frames ahead.
_STREAMED_LAYERS = {**WIRE_LAYERS, TemporalStack.WIRE_NAME: StreamedTemporalStack}


class _Endpoint(Policy):
    """The wire connection to one inference server: sessions forward observations under the border's settings.

    ``InferenceClient`` reads the server, the model, and the session params off the URL.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None, infer_timeout: float):
        self._client = InferenceClient.from_url(url, headers=headers, infer_timeout=infer_timeout)
        # Filled on first contact, through a session opened for it alone.
        self._server_meta: dict[str, Any] | None = None

    def server_meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            session = self._client.new_session()
            try:
                self._server_meta = dict(session.metadata)
            finally:
                session.close()
        return self._server_meta

    def new_session(self, context=None, rt=None) -> RemoteSession:
        if rt is None:
            raise ValueError('A remote session runs its inference on a runtime: pass rt to new_session.')
        compress = bool(self.server_meta().get(offboard_keys.COMPRESS_IMAGES))
        session = self._client.new_session()
        return RemoteSession(session, rt, compress_images=compress)

    @property
    def functions(self) -> cabc.Mapping[str, cabc.Callable[..., Any]]:
        return {INFER: round_trip}


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
        version = meta.get(offboard_keys.POSITRONIC_VERSION, 'unknown')
        declared = meta.get(offboard_keys.LOCAL_STACK)
        layers = _STREAMED_LAYERS if meta.get(offboard_keys.STREAM_FRAMES) else WIRE_LAYERS
        try:
            stack = from_spec(declared, layers) if declared is not None else None
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

    def new_session(self, context=None, rt=None) -> Session:
        return self._policy().new_session(context, rt)

    @property
    def functions(self) -> cabc.Mapping[str, cabc.Callable[..., Any]]:
        return self._policy().functions

    def close(self):
        self._endpoint.close()
