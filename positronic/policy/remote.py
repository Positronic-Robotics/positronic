import collections.abc as cabc
import time
from functools import partial
from typing import Any

import numpy as np
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.policy import keys as policy_keys
from positronic.utils import flatten_dict
from positronic.utils.serialization import encode_jpeg

from .base import Policy, PolicyRun, Processor, Runtime
from .codec import Codec
from .spec import from_spec


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


class RemotePolicy(Policy):
    """Run the server-declared client stack around an ordinary remote inference call.

    ``wire`` names the transport and ``address`` is the address it dials. Each episode owns its connection.
    Submitted calls finish before the harness closes the generator and its connection.
    Codecs wrap the remote callable, so their work is included in submission time.
    """

    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ):
        self._client = InferenceClient(registry.client_wire(wire), address, headers=headers, infer_timeout=infer_timeout)
        self._server_meta: dict[str, Any] | None = None

    def meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            session = self._client.new_session()
            try:
                self._server_meta = dict(session.metadata)
            finally:
                session.close()
        return flatten_dict({policy_keys.TYPE: 'remote', policy_keys.SERVER: self._server_meta})

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        try:
            meta = session.metadata
            self._server_meta = dict(meta)
            declared = meta.get(offboard_keys.LOCAL_STACK)
            if declared is None:
                raise ValueError('Server declares no client processor stack')
            stack = from_spec(declared)
            if not isinstance(stack, Processor):
                raise ValueError('The declared client stack must be a processor')
            infer = partial(round_trip, session, compress_images=bool(meta.get(offboard_keys.COMPRESS_IMAGES)))
            if (codec_spec := meta.get(offboard_keys.LOCAL_CODEC)) is not None:
                codec = from_spec(codec_spec)
                if not isinstance(codec, Codec):
                    raise ValueError('The declared client codec must be a codec')
                infer = codec.wrap(infer)
            yield from stack.run(runtime, infer)
        finally:
            session.close()
