import collections.abc as cabc
import logging
from concurrent.futures import Future, wait
from contextlib import closing
from functools import partial
from threading import Lock, Thread
from typing import Any

import numpy as np
from positronic_model_server import keys as offboard_keys
from positronic_model_server.protocol import ProtocolVersion
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg
from positronic_model_server.tree import map_leaves
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic.policy import keys as policy_keys

from .base import Policy, PolicyRun, Processor, Runtime
from .compatibility import from_v1_spec
from .spec import from_spec


def _prepare_value(value: Any, jpeg_quality: int) -> Any:
    if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
        # A raw HD frame — especially a (T, H, W, 3) stack — can exceed a proxy's message cap.
        return encode_jpeg(value, jpeg_quality)
    return value


def prepare_obs(
    obs: cabc.Mapping[str, Any], compress_images: bool, jpeg_quality: int = DEFAULT_JPEG_QUALITY
) -> dict[str, Any]:
    if not compress_images:
        return dict(obs)
    return map_leaves(obs, partial(_prepare_value, jpeg_quality=jpeg_quality))


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


def _server_frames(declared: dict[str, Any], from_stack: dict[str, Any]) -> dict[str, Any]:
    """The frame of each pose signal that a server states in its metadata or through its client stack."""
    # A server repeats its client stack's frames in its metadata, and the harness checks only the merged copy.
    if disagree := sorted(name for name in declared.keys() & from_stack.keys() if declared[name] != from_stack[name]):
        raise ValueError(
            f'The server states two different {roboarm_keys.EE_FRAMES} for {disagree}: one in its metadata and one '
            'in its client stack'
        )
    return {**from_stack, **declared}


class RemotePolicy(Policy):
    """Run the server-declared client stack around an ordinary remote inference call.

    ``wire`` names the transport and ``address`` is the address it dials. ``jpeg_quality`` sets the JPEG
    quality of images sent to a server that asks for compressed images.
    Each run owns a server session and its connection. Submitted calls finish before the harness
    closes the generator; closing the session waits for the server to release its state, then closes the
    connection, and the run waits for that close only up to a bound. The declared stack determines when client
    codecs run. ``meta`` holds only the client's own values. Each run records the server's values in the episode
    metadata before its first yield, with the frames that the server and its declared stack state as ``EE_FRAMES``.
    ``wait_for_server`` waits for the server without a session.
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
        self._jpeg_quality = jpeg_quality

    def meta(self) -> dict[str, Any]:
        return {policy_keys.TYPE: 'remote'}

    def wait_for_server(self) -> None:
        """Return once the server has loaded its model. Opens no session."""
        self._client.wait_for_server()

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        connection_lock = Lock()
        try:
            server_meta = dict(session.metadata)
            stack = declared_stack(server_meta, session.protocol_version)
            client_meta = dict(stack.meta()) if session.protocol_version is ProtocolVersion.V3 else {}
            # Either side of the server's deployment may state the frames, so the run records both as one entry keyed
            # by pose signal.
            frames = _server_frames(
                server_meta.get(roboarm_keys.EE_FRAMES, {}), client_meta.pop(roboarm_keys.EE_FRAMES, {})
            )
            episode_meta: dict[str, Any] = {**client_meta, policy_keys.SERVER: server_meta}
            if frames:
                episode_meta[roboarm_keys.EE_FRAMES] = frames
            if server_meta.get(policy_keys.COMPRESS_IMAGES):
                episode_meta[policy_keys.JPEG_QUALITY] = self._jpeg_quality
            for name, value in episode_meta.items():
                runtime.set_metadata(name, value)
            compress_images = session.protocol_version is not ProtocolVersion.V3 and bool(
                server_meta.get(policy_keys.COMPRESS_IMAGES)
            )

            def infer(obs: cabc.Mapping[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
                with connection_lock:
                    return round_trip(session, obs, compress_images, self._jpeg_quality)

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
