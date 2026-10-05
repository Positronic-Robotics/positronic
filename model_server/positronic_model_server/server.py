"""Serve a warmed native model and its sessions in one process."""

import asyncio
import hmac
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, TypeVar
from uuid import uuid4

from positronic_wire import wire

from . import keys, protocol, serialization, server_wire, spec

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class Session:
    """A prepared, warm session. Its callable owns session state while sharing loaded weights.

    ``close`` releases that state. ``output_images`` selects result values for JPEG serialization.
    """

    infer: Callable[[Any], Any]
    client_stack: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    output_images: Sequence[serialization.JpegEncoding] = ()
    close: Callable[[], None] = lambda: None


@dataclass
class Model:
    """Loaded, warm model resources and the operation that prepares each session.

    ``parameters`` declares accepted session names and their defaults. ``prepare_session`` receives
    resolved values, validates their model-specific meaning and returns only once that session is warm.
    A preparation that raises must release any resources it acquired before returning a session.
    """

    prepare_session: Callable[[dict[str, Any]], Session]
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    close: Callable[[], None] = lambda: None


class ModelServer:
    """One model, serialized operations, and independently configured transport listeners.

    Loading, session preparation, inference and cleanup all run on one worker thread. The network
    loop remains available during these operations. The factory must finish shared warm-up before
    returning its model; listeners bind afterwards. Each session must retain its own parameters and
    history, and its warm-up must leave the session in its intended initial state.
    """

    WAITING_INTERVAL_SEC = 5.0

    def __init__(
        self, load_model: Callable[[], Model], *, idle_timeout_min: float | None = None, auth_token: str | None = None
    ):
        if auth_token is not None and not (auth_token and all('!' <= c <= '~' for c in auth_token)):
            raise ValueError('auth_token must be non-empty printable ASCII without spaces; pass None to serve open')
        if idle_timeout_min is not None and not 0 < idle_timeout_min < math.inf:
            raise ValueError('idle_timeout_min must be positive and finite, or None')
        self._load_model = load_model
        self._auth_token = auth_token
        self._idle_seconds = None if idle_timeout_min is None else idle_timeout_min * 60
        self._model: Model | None = None
        self._worker: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._sessions: set[asyncio.Task] = set()
        self._last_activity = time.monotonic()

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        if self._auth_token is None:
            return True
        value = headers.get(protocol.AUTH_HEADER.lower(), '')
        return hmac.compare_digest(value.encode(), protocol.bearer(self._auth_token).encode())

    def _keepalive(self) -> int | None:
        self._last_activity = time.monotonic()
        return None if self._idle_seconds is None else math.floor(self._idle_seconds)

    async def _call(self, function: Callable[[], T]) -> T:
        """Keep an operation alive until it finishes, including while its caller is cancelled."""
        assert self._worker is not None
        pending = asyncio.get_running_loop().run_in_executor(self._worker, function)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise

    async def _prepare(self, conn: server_wire.ServerConnection, prepare: Callable[[], None]) -> None:
        pending = asyncio.create_task(self._call(prepare))
        try:
            while not pending.done():
                await conn.send(serialization.serialise({protocol.STATUS: protocol.ServerStatus.WAITING}))
                await asyncio.wait({pending}, timeout=self.WAITING_INTERVAL_SEC)
            await pending
        finally:
            # A disconnected opener still owns the session being prepared until its cleanup completes.
            await asyncio.shield(pending)

    async def _answer(self, conn: server_wire.ServerConnection, session: Session, session_id: str) -> None:
        while True:
            raw = await conn.receive()
            self._last_activity = time.monotonic()
            started = time.perf_counter_ns()
            request = serialization.deserialise(raw)
            if request[protocol.SESSION_ID] != session_id:
                raise ValueError('The session ID does not belong to this connection')
            if request.get(protocol.END_SESSION) is True:
                return
            observation = request[protocol.OBSERVATION]
            decoded = time.perf_counter_ns()
            await conn.send(await self._infer_response(session, observation, started, decoded))

    async def _infer_response(self, session: Session, observation: Any, started: int, decoded: int) -> bytes:
        timing = {protocol.TIMING_DECODE: (decoded - started) / 1e6}

        def infer() -> bytes:
            called = time.perf_counter_ns()
            timing[protocol.TIMING_QUEUED] = (called - decoded) / 1e6
            try:
                result = session.infer(observation)
            finally:
                timing[protocol.TIMING_MODEL] = timing[protocol.TIMING_INFER] = (time.perf_counter_ns() - called) / 1e6
            timing[protocol.TIMING_SERVED] = (time.perf_counter_ns() - started) / 1e6
            # Pack before releasing the worker: native results can reference reusable model buffers.
            result = serialization.encode_images(result, session.output_images)
            return serialization.serialise({protocol.RESULT: result, protocol.TIMING: timing})

        try:
            return await self._call(infer)
        except Exception as error:
            logger.exception('Inference failed')
            return serialization.serialise({protocol.ERROR: str(error)})

    @asynccontextmanager
    async def _prepared_session(self, conn: server_wire.ServerConnection, params: dict[str, Any]):
        model = self._model
        assert model is not None
        session: Session | None = None

        def prepare() -> None:
            nonlocal session
            session = model.prepare_session(json.loads(json.dumps(params, allow_nan=False)))
            spec.validate(session.client_stack)

        try:
            await self._prepare(conn, prepare)
            assert session is not None
            yield session
        finally:
            if session is not None:
                await self._call(session.close)

    async def _exchange_session(self, conn: server_wire.ServerConnection) -> None:
        model = self._model
        assert model is not None
        requested = spec.parse_params(conn.query_params)
        params = spec.resolve_params(model.parameters, requested)
        session_id = uuid4().hex
        async with self._prepared_session(conn, params) as session:
            metadata = {
                **conn.served_address.meta,
                **model.metadata,
                **session.metadata,
                keys.LOCAL_STACK: session.client_stack,
                keys.SESSION_PARAMS: requested,
                keys.EFFECTIVE_PARAMS: params,
                keys.MODEL_SERVER_VERSION: version('positronic-model-server'),
            }
            await conn.send(
                serialization.serialise({
                    protocol.STATUS: protocol.ServerStatus.READY,
                    protocol.PROTOCOL_VERSION: protocol.ProtocolVersion.V3,
                    protocol.SESSION_ID: session_id,
                    protocol.META: metadata,
                })
            )
            await self._answer(conn, session, session_id)
        await conn.send(serialization.serialise({protocol.SESSION_ID: session_id, protocol.END_SESSION: True}))

    async def _serve_session(self, conn: server_wire.ServerConnection) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._sessions.add(task)
        self._last_activity = time.monotonic()
        try:
            await self._exchange_session(conn)
        except wire.PeerDisconnected:
            logger.info('Session disconnected: %s', conn.peer)
        except Exception as error:
            logger.exception('Session failed: %s', conn.peer)
            try:
                await conn.send(
                    serialization.serialise({protocol.STATUS: protocol.ServerStatus.ERROR, protocol.ERROR: str(error)})
                )
                await conn.refuse(str(error))
            except wire.PeerDisconnected:
                logger.info('Session disconnected before receiving its error: %s', conn.peer)
        finally:
            self._sessions.discard(task)
            self._last_activity = time.monotonic()

    async def _idle_watchdog(self) -> None:
        assert self._idle_seconds is not None
        while True:
            await asyncio.sleep(min(self._idle_seconds, 30))
            if not self._sessions and time.monotonic() - self._last_activity >= self._idle_seconds:
                return

    def _load(self) -> None:
        self._model = self._load_model()
        spec.resolve_params(self._model.parameters, {})

    async def _run(self, wires: Sequence[server_wire.Wire], on_ready: Callable[[], None] | None) -> None:
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        started: list[server_wire.Wire] = []
        serving: list[asyncio.Task] = []
        ending: list[asyncio.Task] = []

        try:
            await self._call(self._load)
            if self._stop.is_set():
                return
            for transport in wires:
                started.append(transport)
                await transport.start(self._serve_session, self._keepalive, self._authorized)
            self._last_activity = time.monotonic()
            serving = [asyncio.create_task(transport.serve()) for transport in started]
            ending = [asyncio.create_task(self._stop.wait())]
            if self._idle_seconds is not None:
                ending.append(asyncio.create_task(self._idle_watchdog()))
            if on_ready is not None:
                on_ready()
            await asyncio.wait(serving + ending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in ending:
                task.cancel()
            await asyncio.gather(*ending, return_exceptions=True)
            await self._finish(started, serving)

    async def _finish(self, started: Sequence[server_wire.Wire], serving: Sequence[asyncio.Task]) -> None:
        stops = await asyncio.gather(*(transport.stop() for transport in started), return_exceptions=True)
        for stopped, task in zip(stops, serving, strict=False):
            if isinstance(stopped, BaseException):
                task.cancel()
        outcomes = await asyncio.gather(*serving, return_exceptions=True)
        for task in self._sessions:
            if not task.cancelling():
                task.cancel()
        sessions = await asyncio.gather(*self._sessions, return_exceptions=True)
        failures = [value for value in [*stops, *sessions, *outcomes] if isinstance(value, Exception)]
        try:
            if self._model is not None:
                await self._call(self._model.close)
        except Exception as error:
            failures.append(error)
        finally:
            self._model, self._loop, self._stop = None, None, None
        for error in failures[1:]:
            logger.error('Additional shutdown failure', exc_info=error)
        if failures:
            raise failures[0]

    def serve(self, wires: Sequence[server_wire.Wire], on_ready: Callable[[], None] | None = None) -> None:
        """Load and warm once, then serve until shutdown, idle expiry, or a transport ends."""
        if not wires:
            raise ValueError('At least one transport is required')
        if self._worker is not None:
            raise RuntimeError('This server is already running')
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='model') as worker:
            self._worker = worker
            try:
                asyncio.run(self._run(wires, on_ready))
            except KeyboardInterrupt:
                logger.info('Server interrupted')
            finally:
                self._worker = None

    def shutdown(self) -> None:
        """Request shutdown from any thread; running model work finishes before cleanup."""
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
