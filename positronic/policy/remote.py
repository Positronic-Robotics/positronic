import collections.abc as cabc
import logging
from concurrent.futures import Future, wait
from contextlib import closing
from threading import Lock, Thread
from typing import Any

import numpy as np
from positronic_model_server import keys as offboard_keys
from positronic_model_server.protocol import ProtocolVersion
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, EncodedFrame, encode_jpeg, encoded_frames
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.offboard import protocol as legacy_protocol
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.policy import keys as policy_keys
from positronic.utils import flatten_dict

from .base import Policy, PolicyRun, Processor, Runtime
from .compatibility import from_v1_spec
from .spec import from_spec

# Where a value sits in an observation: its key, then the mapping keys and list indices below it.
ValuePath = tuple[str | int, ...]
# The frames of a context push, by the path of their stack.
EncodedFrames = cabc.Mapping[ValuePath, cabc.Sequence[EncodedFrame]]


def _is_image(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3


def _prepare_value(value: Any, jpeg_quality: int, path: ValuePath, encoded: EncodedFrames) -> Any:
    # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
    if _is_image(value):
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's message cap.
        return encode_jpeg(value, jpeg_quality, encoded.get(path, ()))
    if isinstance(value, cabc.Mapping):
        return {k: _prepare_value(v, jpeg_quality, (*path, k), encoded) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_prepare_value(v, jpeg_quality, (*path, i), encoded) for i, v in enumerate(value))
    return value


def prepare_obs(
    obs: cabc.Mapping[str, Any],
    compress_images: bool,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    encoded: EncodedFrames | None = None,
) -> dict[str, Any]:
    """``obs`` as it goes on the wire. A stack at a path of ``encoded`` uses its JPEG bytes; see ``encode_jpeg``."""
    if not compress_images:
        return dict(obs)
    encoded = encoded or {}
    return {key: _prepare_value(value, jpeg_quality, (key,), encoded) for key, value in obs.items()}


def round_trip(
    session: InferenceSession,
    obs: cabc.Mapping[str, Any],
    compress_images: bool,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    encoded: EncodedFrames | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """One inference over the wire, timed as the ``policy.infer`` span.

    Image preparation has its own span; the inference span covers only the server round trip.
    """
    with telemetry.span(telemetry_keys.SPAN_POLICY_PREPARE):
        prepared = prepare_obs(obs, compress_images, jpeg_quality, encoded)
    with telemetry.span(telemetry_keys.SPAN_POLICY_INFER) as span:
        try:
            return session.infer(prepared)
        finally:
            served = {f'{telemetry_keys.ATTR_SERVED_PREFIX}{k}': v for k, v in session.served_timing.items()}
            telemetry.set_attrs(span, **served)


def _image_stacks(value: Any, path: ValuePath) -> cabc.Iterator[tuple[ValuePath, np.ndarray]]:
    """Every ``(T, H, W, 3)`` stack in ``value``, at any depth of mappings and lists, with its path."""
    if _is_image(value):
        if value.ndim == 4:
            yield path, value
    elif isinstance(value, cabc.Mapping):
        for key, item in value.items():
            yield from _image_stacks(item, (*path, key))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _image_stacks(item, (*path, index))


def _nest(values: cabc.Mapping[ValuePath, Any]) -> dict[str, Any]:
    """The values in nested mappings, one level per path element, each element as a string key.

    The server reads only the frames of a push, so a list index can stand as a key. A string key keeps msgpack's
    strict map keys.
    """
    nested: dict[str, Any] = {}
    for path, value in values.items():
        node = nested
        for part in path[:-1]:
            node = node.setdefault(str(part), {})
        node[str(path[-1])] = value
    return nested


def push_context(session: InferenceSession, obs: cabc.Mapping[str, Any], jpeg_quality: int) -> EncodedFrames:
    """Send the image stacks of ``obs`` as a context push, timed as the ``policy.context`` span, and return the
    pushed frames by the path of their stack. A stack inside a mapping or a list goes in the push under its own
    key, inside the same containers. An ``obs`` without a stack sends nothing and returns none."""
    with telemetry.span(telemetry_keys.SPAN_POLICY_CONTEXT):
        stacks = dict(_image_stacks(obs, ()))
        if not stacks:
            return {}
        markers = {path: encode_jpeg(stack, jpeg_quality) for path, stack in stacks.items()}
        session.push_context(_nest(markers))
        return {path: encoded_frames(stacks[path], marker, jpeg_quality) for path, marker in markers.items()}


def declared_stack(meta: cabc.Mapping[str, Any], protocol_version: ProtocolVersion) -> Processor:
    """The client stack a server's handshake declares, in the form ``protocol_version`` runs."""
    declared = meta.get(offboard_keys.LOCAL_STACK)
    if declared is None:
        raise ValueError('Server declares no client processor stack')
    if protocol_version is ProtocolVersion.V1:
        stack = from_v1_spec(declared, meta, from_spec)
    else:
        stack = from_spec(declared)
    if not isinstance(stack, Processor):
        raise ValueError('The declared client stack must be a processor')
    return stack


logger = logging.getLogger(__name__)

# FOOTGUN: the session close waits for the server to acknowledge the end of the session, for up to the infer
# timeout. A finished run stays alive through that wait, and a watcher that reads liveness calls it healthy.
_CLOSE_TIMEOUT_S = 5.0


def _close_within_bound(session: Any) -> None:
    """Close the session. A close that the server does not answer within the bound continues in the background."""
    closed: Future[None] = Future()

    def close() -> None:
        try:
            session.close()
        except BaseException as failure:
            closed.set_exception(failure)
        else:
            closed.set_result(None)

    Thread(target=close, name='RemotePolicy.close', daemon=True).start()
    if wait([closed], timeout=_CLOSE_TIMEOUT_S).done:
        closed.result()
        return
    logger.warning(
        'The server did not answer the session close within %.1fs. The run continues. The close continues in the '
        'background, and the session closes its connection when the server answers or its infer timeout ends.',
        _CLOSE_TIMEOUT_S,
    )
    closed.add_done_callback(_log_late_close_failure)


def _log_late_close_failure(closed: Future[None]) -> None:
    if (failure := closed.exception()) is not None:
        logger.error('The session close failed after the run continued: %r', failure)


class RemotePolicy(Policy):
    """Run the server-declared client stack around an ordinary remote inference call.

    ``wire`` names the transport and ``address`` is the address it dials. ``jpeg_quality`` sets the JPEG
    quality of images sent to a server that asks for compressed images.
    Each run owns a server session and its connection. Submitted calls finish before the harness
    closes the generator; closing the session waits for the server to release its state, then closes the
    connection, and the run waits for that close only up to a bound. The declared stack determines when client
    codecs run.

    A call from the declared stack with ``context_only=True`` is a context push: it sends the image stacks of the
    observation for the server to hold, and returns ``None``. A server that does not declare the push gets
    nothing. The next requests reuse the JPEG bytes of the pushed frames, and send each one as its digest.
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
        self._client_meta: dict[str, Any] = {}
        self._jpeg_quality = jpeg_quality

    def meta(self) -> dict[str, Any]:
        if self._server_meta is None:
            with closing(self._client.new_session()) as session:
                server_meta = dict(session.metadata)
                client_meta = (
                    declared_stack(server_meta, session.protocol_version).meta()
                    if session.protocol_version is ProtocolVersion.V3
                    else {}
                )
            self._server_meta, self._client_meta = server_meta, client_meta
        meta: dict[str, Any] = {**self._client_meta, policy_keys.TYPE: 'remote', policy_keys.SERVER: self._server_meta}
        if self._server_meta.get(policy_keys.COMPRESS_IMAGES):
            meta[policy_keys.JPEG_QUALITY] = self._jpeg_quality
        return flatten_dict(meta)

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        connection_lock = Lock()
        try:
            server_meta = dict(session.metadata)
            stack = declared_stack(server_meta, session.protocol_version)
            client_meta = stack.meta() if session.protocol_version is ProtocolVersion.V3 else {}
            self._server_meta, self._client_meta = server_meta, client_meta
            compress_images = session.protocol_version is not ProtocolVersion.V3 and bool(
                server_meta.get(policy_keys.COMPRESS_IMAGES)
            )
            supports_context_push = compress_images and (
                server_meta.get(policy_keys.CONTEXT_PUSH) == legacy_protocol.CONTEXT_PUSH_VERSION
            )
            # The frames of the last context push, by the path of their stack.
            pushed: EncodedFrames = {}

            def infer(
                obs: cabc.Mapping[str, Any], *, context_only: bool = False
            ) -> list[dict[str, Any]] | dict[str, Any] | None:
                nonlocal pushed
                with connection_lock:
                    if not context_only:
                        return round_trip(session, obs, compress_images, self._jpeg_quality, pushed)
                    if supports_context_push:
                        pushed = {}
                        pushed = push_context(session, obs, self._jpeg_quality)
                    return None

            with closing(runtime.start(stack, infer)) as run:
                obs = yield
                while True:
                    try:
                        step = run.send(obs)
                    except StopIteration as end:
                        return end.value
                    obs = yield step
        finally:
            # Generator failure can reach cleanup while inference still owns the connection.
            with connection_lock:
                _close_within_bound(session)
