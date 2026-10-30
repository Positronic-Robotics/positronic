import collections.abc as cabc
import logging
from concurrent.futures import Executor as WorkerExecutor
from concurrent.futures import Future, wait
from functools import partial
from threading import Lock, Thread
from types import MappingProxyType
from typing import Any, ParamSpec, TypeVar

import numpy as np
from positronic_model_server import keys as offboard_keys
from positronic_model_server.protocol import ProtocolVersion
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg
from positronic_model_server.spec import NAME
from positronic_model_server.tree import map_leaves
from positronic_wire import registry
from positronic_wire.wire import SessionAddress

from positronic import keys, telemetry, telemetry_keys
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.offboard.client import (
    DEFAULT_CONNECT_DEADLINE,
    DEFAULT_INFER_TIMEOUT,
    DEFAULT_OPEN_TIMEOUT,
    FrameMiss,
    InferenceClient,
    InferenceSession,
)
from positronic.offboard.frame_stream import FrameCodec, FrameStreamer, StackRows, declared_codecs, streams_stack
from positronic.policy import keys as policy_keys

from .base import Answer, Policy, PolicyRun, Processor, Runtime, Step
from .codecs import EncodeImages, PolicyCommandError, TransformImages
from .codecs.action import JointPositionAction
from .codecs.image import ResizeCache
from .compatibility import (
    ACTION_HORIZON,
    ACTION_TIMESTAMP,
    CHUNKED_SCHEDULE,
    RESTRICT_IMAGE_SIZE,
    PauseOnUnavailable,
    RTCScheduleV1,
    from_v1_spec,
    unnest_seq,
)
from .executor import ExecutionRuntime
from .processors import EndOnField, InferenceSchedule, RequestTiming, TemporalStack, TrajectoryPlayer
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
    Raise ``PolicyCommandError`` when this client cannot build that stack.
    """
    declared = meta.get(offboard_keys.LOCAL_STACK)
    if declared is None:
        raise PolicyCommandError('Server declares no client processor stack')
    build = partial(from_spec, overrides=overrides)
    try:
        if protocol_version is ProtocolVersion.V1:
            stack = from_v1_spec(declared, meta, build)
        else:
            stack = build(declared)
    except Exception as e:
        raise PolicyCommandError(f'This client cannot build the client stack that the server declares: {e!r}') from e
    if not isinstance(stack, Processor):
        raise PolicyCommandError('The declared client stack must be a processor')
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

    ``wire`` names the transport and ``address`` is the address it dials. ``open_timeout`` and ``connect_deadline``
    bound the connect, as in ``InferenceClient``. ``jpeg_quality`` sets the JPEG quality of images sent to a server that
    asks for compressed images. ``frame_stream`` names the codec (``jpeg`` or ``h264``) of a frame stream
    (``positronic.offboard.frame_stream``), for a server that declares one for a declared stack that the stream mirrors
    (``_stream_refusal``); otherwise requests go in full.
    ``frame_stream_snap_sec`` lets a streamed request that comes within that many seconds after its predicted time
    refer to every streamed row (``FrameStreamer``'s ``snap_sec``); 0 keeps every observation exact.
    Each run owns a server session and its connection. Submitted calls finish before the harness
    closes the generator; closing the session waits for the server to release its state, then closes the
    connection, and the run waits for that close only up to a bound. The declared stack determines when client
    codecs run. When this client knows every component of the declared stack, a streamed request sizes and stacks
    only the frames it carries and refers to the others by the stack entry they sample
    (``positronic.offboard.frame_stream``). ``meta`` holds only the client's own values. Each run records the
    server's values in the episode metadata before its first yield, with the frames that the server and its
    declared stack state as ``EE_FRAMES``, and the ID of the session that serves the episode as ``SESSION_ID``.
    ``wait_for_server`` waits for the server without a session.
    """

    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        frame_stream: str | None = None,
        frame_stream_snap_sec: float = 0.0,
    ):
        self._client = InferenceClient(
            registry.client_wire(wire),
            address,
            headers=headers,
            open_timeout=open_timeout,
            connect_deadline=connect_deadline,
            infer_timeout=infer_timeout,
        )
        self._frame_codec = None if frame_stream is None else FrameCodec(frame_stream)
        if not frame_stream_snap_sec >= 0:
            raise ValueError('frame_stream_snap_sec must not be negative')
        self._frame_stream_snap_sec = frame_stream_snap_sec
        self._jpeg_quality = jpeg_quality

    def meta(self) -> dict[str, Any]:
        return {policy_keys.TYPE: 'remote'}

    def wait_for_server(self) -> None:
        """Return once the server has loaded its model. Opens no session."""
        self._client.wait_for_server()

    @staticmethod
    def _require_robot_commands(step: Step) -> None:
        """Raise ``PolicyCommandError`` when a robot command channel of ``step`` holds no command object."""
        for name, value in step.commands.items():
            if keys.is_robot_command(name) and not isinstance(value, command.CommandType):
                raise PolicyCommandError(
                    f'{name} carries a {type(value).__name__}, not a command. A protocol v3 server answers the output '
                    'of its model, and the client stack that it declares decodes that output into commands, for '
                    f'example with {JointPositionAction.WIRE_NAME}. Only a v1 or v2 server can answer a command as a '
                    'plain mapping.'
                )

    @staticmethod
    def _close_stack(execution: ExecutionRuntime | None, run: PolicyRun | None) -> None:
        """Drain calls before closing the run that owns their resources."""
        if execution is not None:
            execution.close()
        if run is not None:
            run.close()

    @staticmethod
    def _episode_meta(
        server_meta: dict[str, Any],
        client_meta: dict[str, Any],
        frames: dict[str, Any],
        jpeg_quality: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        """The values that a run records in the episode metadata before its first yield. ``jpeg_quality`` is that of
        the images its requests carry."""
        episode_meta: dict[str, Any] = {**client_meta, policy_keys.SERVER: server_meta}
        if frames:
            episode_meta[roboarm_keys.EE_FRAMES] = frames
        if server_meta.get(policy_keys.COMPRESS_IMAGES):
            episode_meta[policy_keys.JPEG_QUALITY] = jpeg_quality
        if session_id is not None:
            episode_meta[policy_keys.SESSION_ID] = session_id
        return episode_meta

    @staticmethod
    def _stacked(obs: cabc.Mapping[str, Any]) -> dict[str, Any]:
        """``obs`` with each ``StackRows`` sized and stacked: the observation of a request that goes in full."""
        return {key: value.stack() if isinstance(value, StackRows) else value for key, value in obs.items()}

    @staticmethod
    def _infer(
        streamer: FrameStreamer | None,
        call: cabc.Callable[[cabc.Mapping[str, Any]], T],
        connection_lock: Lock,
        obs: cabc.Mapping[str, Any],
    ) -> T:
        """One inference: ``call`` with ``obs``, or under the connection lock with the request ``streamer`` makes of
        it, which refers to the streamed rows. A session without a stream calls as ``InferenceSession`` allows."""
        if streamer is None:
            return call(obs)
        if streamer.call == 0:
            full = RemotePolicy._stacked(obs)
            with connection_lock:
                return call(full)
        streamed = streamer.request(obs)
        with connection_lock:
            try:
                return call(streamed)
            except FrameMiss as miss:
                # The protocol takes the request in full after a frame_miss answer.
                streamer.frame_misses += 1
                logger.warning(
                    'The server does not have the streamed rows %s; sending the request in full', miss.missing
                )
                return call(RemotePolicy._stacked(obs))

    class _ReportingAnswer(Answer[T]):
        """``answer``, reporting its result, ``elapsed_sec`` and ``replans`` to ``on_read`` at the first read."""

        def __init__(self, answer: Answer[T], on_read: cabc.Callable[[T, float, int], None], replans: int) -> None:
            self._answer = answer
            self._on_read = on_read
            self._replans = replans
            self._reported = False

        def done(self) -> bool:
            return self._answer.done()

        @property
        def elapsed_sec(self) -> float:
            return self._answer.elapsed_sec

        @property
        def execution_sec(self) -> float:
            return self._answer.execution_sec

        def result(self) -> T:
            result = self._answer.result()
            if not self._reported:
                self._reported = True
                self._on_read(result, self._answer.elapsed_sec, self._replans)
            return result

        def cancel(self) -> None:
            self._answer.cancel()

    class _ReportingRuntime(Runtime):
        """``inner``, with each answer reporting its result, its ``elapsed_sec`` and the ``replans`` count at its submit
        to ``on_read`` at the first read, and each submit to ``on_submit`` before the call starts."""

        def __init__(
            self,
            inner: Runtime,
            on_read: cabc.Callable[[Any, float, int], None],
            replans: cabc.Callable[[], int],
            on_submit: cabc.Callable[[], None] | None = None,
        ) -> None:
            self._inner = inner
            self._on_read = on_read
            self._replans = replans
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
            replans = self._replans()
            if self._on_submit is not None:
                self._on_submit()
            return RemotePolicy._ReportingAnswer(self._inner.submit(function, *args, **kwargs), self._on_read, replans)

        def _submit_on(
            self,
            workers: WorkerExecutor,
            limit: int,
            function: cabc.Callable[P, T],
            /,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Answer[T]:
            replans = self._replans()
            answer = self._inner._submit_on(workers, limit, function, *args, **kwargs)
            return RemotePolicy._ReportingAnswer(answer, self._on_read, replans)

        def has_ready_answers(self) -> bool:
            return self._inner.has_ready_answers()

    def run(self, runtime: Runtime) -> PolicyRun:
        session = self._client.new_session()
        connection_lock = Lock()
        streamer: FrameStreamer | None = None
        execution = None
        run = None
        try:
            try:
                server_meta = dict(session.metadata)
                execution = ExecutionRuntime(runtime, max_in_flight=session.max_in_flight)
                server_compresses_images = bool(server_meta.get(policy_keys.COMPRESS_IMAGES))
                compress_images = session.protocol_version is not ProtocolVersion.V3 and server_compresses_images
                stack, streamer, quality, fed = self._stack_and_streamer(
                    session, compress_images, connection_lock, runtime
                )
                client_meta = dict(stack.meta()) if session.protocol_version is ProtocolVersion.V3 else {}
                # Either side of the server's deployment may state the frames, so the run records both as one entry
                # keyed by pose signal.
                frames = _server_frames(
                    server_meta.get(roboarm_keys.EE_FRAMES, {}), client_meta.pop(roboarm_keys.EE_FRAMES, {})
                )
            except PolicyCommandError:
                raise
            except Exception as e:
                raise PolicyCommandError(f'This client cannot use the metadata that the server states: {e!r}') from e
            recorded_session_id = session.session_id
            for name, value in self._episode_meta(
                server_meta, client_meta, frames, quality, recorded_session_id
            ).items():
                runtime.set_metadata(name, value)
            answered_session_id = recorded_session_id

            def call(obs: cabc.Mapping[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
                """One round trip, under the connection lock, that records the session that answered it."""
                nonlocal answered_session_id
                answer = round_trip(session, obs, compress_images, quality)
                answered_session_id = session.session_id
                return answer

            infer = partial(self._infer, streamer, call, connection_lock)
            policy_runtime = (
                execution
                if streamer is None
                else self._ReportingRuntime(execution, streamer.on_answer, lambda: streamer.replans, streamer.on_submit)
            )
            run = policy_runtime.start(stack, infer)
            obs = yield
            while True:
                if streamer is not None and not fed:
                    streamer.on_tick(obs)
                try:
                    step = run.send(obs)
                    if isinstance(step, Step):
                        self._require_robot_commands(step)
                except StopIteration as end:
                    return end.value
                except PolicyCommandError:
                    raise
                # A data error from the declared stack or its step is the policy's; a transport error is not one.
                except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
                    raise PolicyCommandError(
                        f'The client stack that the server declares cannot use the answer or the observation: {e!r}'
                    ) from e
                finally:
                    # A connection that drops before its first answer reopens under a new session ID. The episode
                    # names that session once it answered, so the ID it records has a session that served it.
                    if answered_session_id != recorded_session_id:
                        recorded_session_id = answered_session_id
                        runtime.set_metadata(policy_keys.SESSION_ID, recorded_session_id)
                obs = yield step
        finally:
            self._close_stack(execution, run)
            if streamer is not None:
                streamer.close()
            # A frame stream row may still be on its way out under the connection lock.
            with connection_lock:
                _close_within_bound(session)

    @staticmethod
    def _recorded(factory: ComponentFactory, built: list[Any]) -> ComponentFactory:
        """``factory``, recording each component it builds in ``built``, in build order."""

        def build(**args: Any) -> Any:
            component = factory(**args)
            built.append(component)
            return component

        return build

    @staticmethod
    def _transform_row(
        transforms: cabc.Sequence[TransformImages], images: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """The frames of one row, transformed as the declared stack transforms them, in its order."""
        for transform in transforms:
            images = transform.transform_frames(images)
        return images

    class _PassImages(EncodeImages):
        """A declared ``encode_images`` in a streaming session: it leaves the frames it would encode as arrays, so
        that the frame streamer compares the stack with its rows and encodes the frames the request carries itself, at
        the declared quality. An image ``encode_images`` would leave, such as a float frame, it encodes at
        ``fallback_quality``: the quality the request encodes that image at without a stream."""

        def __init__(self, *, fallback_quality: int, **args: Any) -> None:
            super().__init__(**args)
            self._fallback_quality = fallback_quality

        @property
        def quality(self) -> int:
            return self._quality

        @property
        def encodes_every_image(self) -> bool:
            return self._images is None

        def encode(self, data: dict) -> dict:
            return map_leaves(data, self._pass_value)

        def _pass_value(self, value: Any) -> Any:
            if self.encodes(value):
                return value  # the stream and the request encode it at the declared quality
            return _prepare_value(value, self._fallback_quality)

    class _StreamerHooks:
        """The calls that the components of a streaming session's stack make on its frame streamer, which is built after
        them: the schedule's replans and the temporal stack's entries."""

        def __init__(self) -> None:
            self.streamer: FrameStreamer | None = None

        def _built(self) -> FrameStreamer:
            if self.streamer is None:
                raise RuntimeError('A component of the stack calls its frame streamer before the streamer exists')
            return self.streamer

        def replan(self, now_ns: int) -> None:
            self._built().on_replan(now_ns)

        def entry(self, t: float, entry: dict[str, np.ndarray] | None) -> tuple[str, ...]:
            return self._built().on_entry(t, entry)

    # The components that a streaming session's stack may hold for its temporal stack to give ``StackRows``: each one
    # passes a stack on untouched, or (an image transform) defers its transform to the rows a request carries.
    _ROWS_PASS_THROUGH = frozenset({
        PauseOnUnavailable.WIRE_NAME,
        EndOnField.WIRE_NAME,
        TemporalStack.WIRE_NAME,
        RTCScheduleV1.WIRE_NAME,
        CHUNKED_SCHEDULE,
        InferenceSchedule.WIRE_NAME,
        TrajectoryPlayer.WIRE_NAME,
        TransformImages.WIRE_NAME,
        RESTRICT_IMAGE_SIZE,
        EncodeImages.WIRE_NAME,
        ACTION_TIMESTAMP,
        ACTION_HORIZON,
    })

    @staticmethod
    def _predicts(schedule: Policy) -> bool:
        """Whether the frame streamer predicts the calls of ``schedule``: a chunked schedule's, one call at a time,
        each after the previous chunk less its lead (``ChunkedSchedule``), or an RTC schedule's, one call at a time,
        each ``call_after_sec`` after the previous call (``RTCSchedule``)."""
        if not isinstance(schedule, InferenceSchedule):
            return False
        reference, chained = schedule.request_timing.reference, schedule.merge.chained
        Reference = RequestTiming.Reference
        return (reference is Reference.BEFORE_END and chained) or (reference is Reference.AFTER_START and not chained)

    @staticmethod
    def _stream_refusal(
        parts: cabc.Sequence[cabc.Mapping[str, Any]],
        temporal: cabc.Sequence[TemporalStack],
        schedules: cabc.Sequence[Policy],
        encodes: cabc.Sequence[_PassImages],
    ) -> str | None:
        """Why the frame stream does not mirror the declared stack of ``parts``, built with ``temporal``,
        ``schedules`` and ``encodes``, or None when it does. It mirrors one temporal stack whose rows it carries, one
        schedule whose calls it predicts, and an ``encode_images`` only as the stack's last part, over every image:
        the rows and the requests then carry the frames that the stack would send."""
        if len(temporal) != 1 or not streams_stack(temporal[0].keys, temporal[0].offsets_sec):
            return 'it holds no temporal stack whose rows the stream carries'
        if len(schedules) != 1 or not RemotePolicy._predicts(schedules[0]):
            return 'it holds no schedule whose calls the stream predicts'
        last = parts[-1].get(NAME) == EncodeImages.WIRE_NAME
        if encodes and not (len(encodes) == 1 and encodes[0].encodes_every_image and last):
            return 'its encode_images is not one last encoder of every image'
        return None

    def _stack_and_streamer(
        self, session: InferenceSession, compress_images: bool, connection_lock: Lock, runtime: Runtime
    ) -> tuple[Processor, FrameStreamer | None, int, bool]:
        """The declared stack of ``session``, its frame streamer when this policy streams frames on it, the JPEG
        quality of the images its requests carry, and whether the stack's temporal stack feeds the streamer."""
        meta, version = session.metadata, session.protocol_version
        if self._frame_codec is None:
            return declared_stack(meta, version), None, self._jpeg_quality, False
        streams = version in (ProtocolVersion.V1, ProtocolVersion.V2)
        if not (compress_images and streams and self._frame_codec in declared_codecs(meta)):
            logger.warning('The session offers no %s frame stream; requests go in full', self._frame_codec.value)
            return declared_stack(meta, version), None, self._jpeg_quality, False
        if session.max_in_flight != 1:
            # The streamer plans the rows of one next call: with overlapping calls a row has no one call to go to.
            logger.warning(
                'The session allows %d calls in flight; the frame stream predicts one at a time; requests go in full',
                session.max_in_flight,
            )
            return declared_stack(meta, version), None, self._jpeg_quality, False
        # The streamer mirrors the declared temporal stack, and transforms each row as the stack transforms its
        # frames. The stack's transforms share one cache with the streamer, so a frame is resized once. A declared
        # encode_images leaves the stack's frames as arrays; the streamer and the request encode them at its quality.
        cache = ResizeCache()
        temporal: list[TemporalStack] = []
        transforms: list[TransformImages] = []
        schedules: list[InferenceSchedule] = []
        rtc: list[RTCScheduleV1] = []
        encodes: list[RemotePolicy._PassImages] = []
        # A stack of known components gives the streamer its buffer's entries, and each request its stacks as
        # StackRows: the request sizes and stacks only the rows it carries, and refers to the others by entry.
        parts = unnest_seq(meta[offboard_keys.LOCAL_STACK])
        fed = all(part.get(NAME) in self._ROWS_PASS_THROUGH for part in parts)
        hooks = self._StreamerHooks()
        stacking = partial(TemporalStack.from_spec_args, on_entry=hooks.entry) if fed else TemporalStack.from_spec_args
        overrides: dict[ComponentFactory, ComponentFactory] = {
            TemporalStack.from_spec_args: self._recorded(stacking, temporal),
            TransformImages.from_spec_args: self._recorded(
                partial(TransformImages.from_spec_args, cache=cache), transforms
            ),
            InferenceSchedule.from_spec_args: self._recorded(
                partial(InferenceSchedule.from_spec_args, on_replan=hooks.replan), schedules
            ),
            RTCScheduleV1.from_spec_args: self._recorded(RTCScheduleV1.from_spec_args, rtc),
            EncodeImages.from_spec_args: self._recorded(
                partial(self._PassImages.from_spec_args, fallback_quality=self._jpeg_quality), encodes
            ),
        }
        stack = declared_stack(meta, version, overrides)
        refusal = self._stream_refusal(parts, temporal, [*schedules, *(v1.schedule for v1 in rtc)], encodes)
        if refusal is not None:
            logger.warning('The frame stream does not mirror the declared stack: %s; requests go in full', refusal)
            return declared_stack(meta, version), None, self._jpeg_quality, False
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
            # The schedule's own estimate of its lead, fed the same elapsed times: one estimate per session, as per
            # episode.
            lead=schedules[0].latency_estimate.run().update if schedules else None,
            resize=partial(self._transform_row, transforms) if transforms else None,
            jpeg_quality=quality,
            call_after_sec=rtc[0].schedule.request_timing.seconds if rtc else None,
            snap_sec=self._frame_stream_snap_sec,
        )
        hooks.streamer = streamer
        return stack, streamer, quality, fed
