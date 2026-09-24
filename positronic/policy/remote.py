import collections.abc as cabc
from contextlib import closing
from threading import Lock
from typing import Any

import numpy as np
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.offboard.protocol import ProtocolVersion
from positronic.policy import keys as policy_keys
from positronic.utils import flatten_dict
from positronic.utils.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg

from .base import Policy, PolicyRun, Processor, Runtime
from .codec import Codec
from .compatibility import StackV1
from .spec import from_spec


def _prepare_value(value: Any, jpeg_quality: int) -> Any:
    # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
    if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's message cap.
        return encode_jpeg(value, jpeg_quality)
    if isinstance(value, cabc.Mapping):
        return {k: _prepare_value(v, jpeg_quality) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_prepare_value(v, jpeg_quality) for v in value)
    return value


def prepare_obs(
    obs: cabc.Mapping[str, Any], compress_images: bool, jpeg_quality: int = DEFAULT_JPEG_QUALITY
) -> dict[str, Any]:
    if not compress_images:
        return dict(obs)
    return {key: _prepare_value(value, jpeg_quality) for key, value in obs.items()}


def round_trip(
    session: InferenceSession,
    obs: cabc.Mapping[str, Any],
    compress_images: bool,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span.

    Image preparation has its own span; the inference span covers only the server round trip.
    """
    with telemetry.span(telemetry_keys.SPAN_POLICY_PREPARE):
        prepared = prepare_obs(obs, compress_images, jpeg_quality)
    with telemetry.span(telemetry_keys.SPAN_POLICY_INFER) as span:
        try:
            return session.infer(prepared)
        finally:
            served = {f'{telemetry_keys.ATTR_SERVED_PREFIX}{k}': v for k, v in session.served_timing.items()}
            telemetry.set_attrs(span, **served)


def declared_stack(meta: cabc.Mapping[str, Any], protocol_version: ProtocolVersion) -> Processor:
    """The client stack a server's handshake declares, in the form ``protocol_version`` runs."""
    declared = meta.get(offboard_keys.LOCAL_STACK)
    if declared is None:
        raise ValueError('Server declares no client processor stack')
    stack = from_spec(declared)
    if protocol_version is ProtocolVersion.V1 and isinstance(stack, Codec):
        stack = StackV1(stack)
    if not isinstance(stack, Processor):
        raise ValueError('The declared client stack must be a processor')
    return stack


class RemotePolicy(Policy):
    """Run the server-declared client stack around an ordinary remote inference call.

    ``wire`` names the transport and ``address`` is the address it dials. ``jpeg_quality`` sets the JPEG
    quality of images sent to a server that asks for compressed images.
    Each run owns a server session and its connection. Submitted calls finish before the harness
    closes the generator; closing the session waits for the server to release its state, then closes
    the connection. The declared stack determines when client codecs run.
    """

    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self._client = InferenceClient(
            registry.client_wire(wire), address, headers=headers, infer_timeout=infer_timeout
        )
        self._server_meta: dict[str, Any] | None = None
        self._jpeg_quality = jpeg_quality

    def meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            session = self._client.new_session()
            try:
                self._server_meta = dict(session.metadata)
            finally:
                session.close()
        meta: dict[str, Any] = {policy_keys.TYPE: 'remote', policy_keys.SERVER: self._server_meta}
        if self._server_meta.get(offboard_keys.COMPRESS_IMAGES):
            meta[policy_keys.JPEG_QUALITY] = self._jpeg_quality
        return flatten_dict(meta)

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        connection_lock = Lock()
        try:
            meta = session.metadata
            self._server_meta = dict(meta)
            stack = declared_stack(meta, session.protocol_version)
            compress_images = bool(meta.get(offboard_keys.COMPRESS_IMAGES))

            def infer(obs: cabc.Mapping[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
                with connection_lock:
                    return round_trip(session, obs, compress_images, self._jpeg_quality)

            with closing(runtime.start(stack, infer)) as run:
                obs = yield
                while True:
                    try:
                        step = run.send(obs)
                    except StopIteration:
                        return
                    obs = yield step
        finally:
            # Generator failure can reach cleanup while inference still owns the connection.
            with connection_lock:
                session.close()
