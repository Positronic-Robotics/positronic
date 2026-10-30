import collections.abc as cabc
import logging
from concurrent.futures import Future, wait
from contextlib import closing
from functools import partial
from threading import Lock, Thread
from types import MappingProxyType
from typing import Any, ParamSpec, TypeVar

import numpy as np
from positronic_model_server import keys as offboard_keys
from positronic_model_server.protocol import ProtocolVersion
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg
from positronic_model_server.tree import map_leaves
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import telemetry, telemetry_keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, FrameMiss, InferenceClient, InferenceSession
from positronic.offboard.frame_stream import FrameCodec, FrameStreamer, declared_codecs, streams_stack
from positronic.policy import keys as policy_keys

from .base import Answer, Policy, PolicyRun, Processor, Runtime
from .codecs import EncodeImages, TransformImages
from .codecs.image import ResizeCache
from .compatibility import from_v1_spec
from .processors import ChunkedSchedule, TemporalStack
from .spec import ComponentFactory, from_spec

P = ParamSpec('P')
T = TypeVar('T')


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


def declared_stack(
    meta: cabc.Mapping[str, Any],
    protocol_version: ProtocolVersion,
    overrides: cabc.Mapping[ComponentFactory, ComponentFactory] = MappingProxyType({}),
) -> Processor:
    """The client stack a server's handshake declares, in the form ``protocol_version`` runs.

    ``overrides`` maps a registered factory to the factory that this client builds in its place (``from_spec``).
    """
    declared = meta.get(offboard_keys.LOCAL_STACK)
    if declared is None:
        raise ValueError('Server declares no client processor stack')
    build = partial(from_spec, overrides=overrides)
    if protocol_version is ProtocolVersion.V1:
        stack = from_v1_spec(declared, meta, build)
    else:
        stack = build(declared)
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


class _ReportingAnswer(Answer[T]):
    """``answer``, reporting its result to ``on_read`` at the first read."""

    def __init__(self, answer: Answer[T], on_read: cabc.Callable[[T], None]) -> None:
        self._answer = answer
        self._on_read = on_read
        self._reported = False

    def done(self) -> bool:
        return self._answer.done()

    def result(self) -> T:
        result = self._answer.result()
        if not self._reported:
            self._reported = True
            self._on_read(result)
        return result

    def cancel(self) -> None:
        self._answer.cancel()


class _ReportingRuntime(Runtime):
    """``inner``, with each answer reporting its result to ``on_read`` at the first read, and each submit to
    ``on_submit`` before the call starts."""

    def __init__(
        self, inner: Runtime, on_read: cabc.Callable[[Any], None], on_submit: cabc.Callable[[], None] | None = None
    ) -> None:
        self._inner = inner
        self._on_read = on_read
        self._on_submit = on_submit

    def set_metadata(self, name: str, value: Any) -> None:
        self._inner.set_metadata(name, value)

    @property
    def time_ns(self) -> int:
        return self._inner.time_ns

    @property
    def tick(self) -> int:
        return self._inner.tick

    def submit(self, function: cabc.Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        if self._on_submit is not None:
            self._on_submit()
        return _ReportingAnswer(self._inner.submit(function, *args, **kwargs), self._on_read)


def _recorded(factory: ComponentFactory, built: list[Any]) -> ComponentFactory:
    """``factory``, recording each component it builds in ``built``, in build order."""

    def build(**args: Any) -> Any:
        component = factory(**args)
        built.append(component)
        return component

    return build


def _transform_row(transforms: cabc.Sequence[TransformImages], images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The frames of one row, transformed as the declared stack transforms them, in its order."""
    for transform in transforms:
        images = transform.transform_frames(images)
    return images


class _PassImages(EncodeImages):
    """A declared ``encode_images`` in a streaming session: it leaves the frames as arrays, so that the frame
    streamer compares the stack with its rows and encodes the frames the request carries itself, at the declared
    quality."""

    @property
    def quality(self) -> int:
        return self._quality

    def encode(self, data: dict) -> dict:
        return data


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
    quality of images sent to a server that asks for compressed images. ``frame_stream`` names the codec
    (``jpeg`` or ``h264``) of a frame stream (``positronic.offboard.frame_stream``), for a server that declares one
    for a temporal stack that the stream carries; otherwise requests go in full.
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
        frame_stream: str | None = None,
    ):
        self._client = InferenceClient(
            registry.client_wire(wire), address, headers=headers, infer_timeout=infer_timeout
        )
        self._frame_codec = None if frame_stream is None else FrameCodec(frame_stream)
        self._jpeg_quality = jpeg_quality

    def meta(self) -> dict[str, Any]:
        return {policy_keys.TYPE: 'remote'}

    def wait_for_server(self) -> None:
        """Return once the server has loaded its model. Opens no session."""
        self._client.wait_for_server()

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        connection_lock = Lock()
        streamer: FrameStreamer | None = None
        try:
            server_meta = dict(session.metadata)
            compress_images = session.protocol_version is not ProtocolVersion.V3 and bool(
                server_meta.get(policy_keys.COMPRESS_IMAGES)
            )
            stack, streamer, quality = self._stack_and_streamer(session, compress_images, connection_lock, runtime)
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
            policy_runtime = (
                runtime if streamer is None else _ReportingRuntime(runtime, streamer.on_answer, streamer.on_submit)
            )

            def infer(obs: cabc.Mapping[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
                if streamer is None or streamer.call == 0:
                    with connection_lock:
                        return round_trip(session, obs, compress_images, quality)
                streamed = streamer.request(obs)
                with connection_lock:
                    try:
                        return round_trip(session, streamed, compress_images, quality)
                    except FrameMiss as miss:
                        # The protocol takes the request in full after a frame_miss answer.
                        logger.warning(
                            'The server does not have the streamed rows %s; sending the request in full', miss.missing
                        )
                        return round_trip(session, obs, compress_images, quality)

            with closing(policy_runtime.start(stack, infer)) as run:
                obs = yield
                while True:
                    if streamer is not None:
                        streamer.on_tick(obs)
                    try:
                        step = run.send(obs)
                    except StopIteration as end:
                        return end.value
                    obs = yield step
        finally:
            if streamer is not None:
                streamer.close()
            # Generator failure can reach cleanup while inference still owns the connection.
            with connection_lock:
                _close_within_bound(session)

    def _stack_and_streamer(
        self, session: InferenceSession, compress_images: bool, connection_lock: Lock, runtime: Runtime
    ) -> tuple[Processor, FrameStreamer | None, int]:
        """The declared stack of ``session``, its frame streamer when this policy streams frames on it, and the
        JPEG quality of the images its requests carry."""
        meta, version = session.metadata, session.protocol_version
        if self._frame_codec is None:
            return declared_stack(meta, version), None, self._jpeg_quality
        streams = version in (ProtocolVersion.V1, ProtocolVersion.V2)
        if not (compress_images and streams and self._frame_codec in declared_codecs(meta)):
            logger.warning('The session offers no %s frame stream; requests go in full', self._frame_codec.value)
            return declared_stack(meta, version), None, self._jpeg_quality
        # The streamer mirrors the declared temporal stack, and transforms each row as the stack transforms its
        # frames. The stack's transforms share one cache with the streamer, so a frame is resized once. A declared
        # encode_images leaves the stack's frames as arrays; the streamer and the request encode them at its quality.
        cache = ResizeCache()
        temporal: list[TemporalStack] = []
        transforms: list[TransformImages] = []
        schedules: list[ChunkedSchedule] = []
        encodes: list[_PassImages] = []
        overrides: dict[ComponentFactory, ComponentFactory] = {
            TemporalStack.from_spec_args: _recorded(TemporalStack.from_spec_args, temporal),
            TransformImages.from_spec_args: _recorded(partial(TransformImages.from_spec_args, cache=cache), transforms),
            ChunkedSchedule.from_spec_args: _recorded(ChunkedSchedule.from_spec_args, schedules),
            EncodeImages.from_spec_args: _recorded(_PassImages.from_spec_args, encodes),
        }
        stack = declared_stack(meta, version, overrides)
        if len(temporal) != 1 or not streams_stack(temporal[0].keys, temporal[0].offsets_sec):
            logger.warning(
                'The declared stack holds no temporal stack that a frame stream carries; requests go in full'
            )
            return declared_stack(meta, version), None, self._jpeg_quality
        (stacked,) = temporal
        quality = encodes[0].quality if encodes else self._jpeg_quality
        stack_meta = stack.meta()

        def send(message: dict[str, Any]) -> None:
            with connection_lock:
                session.send_message(message)

        streamer = FrameStreamer(
            send,
            lambda: runtime.time_ns,
            codec=self._frame_codec,
            keys=stacked.keys,
            offsets_sec=stacked.offsets_sec,
            pad_start=stacked.pad_start,
            fps=stack_meta[policy_keys.ACTION_FPS],
            horizon_sec=stack_meta.get(policy_keys.ACTION_HORIZON_SEC),
            # The schedule's own estimate of its lead, fed the same delays: one estimate per session, as per episode.
            lead=schedules[0].inference_lead.run().update if len(schedules) == 1 else None,
            resize=partial(_transform_row, transforms) if transforms else None,
            jpeg_quality=quality,
        )
        return stack, streamer, quality
