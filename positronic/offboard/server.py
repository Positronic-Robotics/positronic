"""The inference server: serves a policy pipeline (see ``positronic.offboard.spec``) over the offboard protocol."""

import asyncio
import hmac
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any
from uuid import uuid4

import configuronic as cfn
from fastapi import APIRouter, Depends, Header, HTTPException
from positronic_wire import wire
from starlette.datastructures import QueryParams

from positronic import telemetry
from positronic.offboard import keys as offboard_keys
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy.base import Obs

from . import grpc_wire, protocol, server_wire, websocket_wire
from .protocol import AUTH_HEADER, AUTH_TOKEN_ENV, bearer, deserialise, serialise

logger = logging.getLogger(__name__)


def _literal_value(raw: str) -> Any:
    """JSON-decode one query value, or keep it as the raw string when it does not parse."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _session_params(query_params: QueryParams) -> dict[str, Any]:
    """Decode session query params into pipeline-config override kwargs (dotted keys reach nested args)."""
    items = query_params.multi_items()
    if len(items) != len(dict(query_params)):
        counts = Counter(key for key, _ in items)
        dupes = sorted(key for key, n in counts.items() if n > 1)
        raise ValueError(f'Duplicate session param keys: {dupes}')
    return {key: _literal_value(raw) for key, raw in items}


class _ServedTiming:
    """What one inference cost the server, in milliseconds on the server's own clock.

    Every figure is a duration. ``served_ms`` opens when the observation arrives and brackets the
    phases inside it. The answer's serialisation and send fall outside every figure: the report rides
    in that answer.
    """

    def __init__(self) -> None:
        self._opened = time.time_ns()
        self._phases: dict[str, float] = {}

    def _record(self, key: str, start_ns: int, end_ns: int) -> None:
        self._phases[key] = (end_ns - start_ns) / 1e6

    @contextmanager
    def phase(self, key: str) -> Iterator[None]:
        """A block no session call brackets, under a wire key the protocol already spells."""
        started = time.time_ns()
        try:
            yield
        finally:
            self._record(key, started, time.time_ns())

    def report(self) -> dict[str, float]:
        """The phases closed so far, under the span bracketing them."""
        return {protocol.TIMING_SERVED: (time.time_ns() - self._opened) / 1e6, **self._phases}

    def _record_span(self, name: str, start_ns: int, end_ns: int) -> None:
        key = protocol.timing_key(name)
        index = 2
        while key in self._phases:
            key = protocol.timing_key(f'{name}_{index}')
            index += 1
        self._record(key, start_ns, end_ns)

    def infer(self, function: Callable[[Obs], Any], obs: Obs) -> Any:
        """Bind this request's timing for the duration of the call."""
        with telemetry.timings_to(self._record_span):
            return function(obs)


class PolicyServer:
    """Serve a callable model with explicit server codecs and a declared client processor stack.

    A config-launched pipeline accepts session parameters as dotted configuration overrides.
    An instantiated PolicyDeployment refuses session parameters. The server loads the source's one
    checkpoint at startup and serves it to every session.
    """

    def __init__(
        self,
        pipeline: cfn.Config | PolicyDeployment,
        idle_timeout_min: float | None = None,
        auth_token: str | None = None,
    ):
        self._pipeline_cfg = pipeline if isinstance(pipeline, cfn.Config) else None
        self._pipeline = pipeline.instantiate() if isinstance(pipeline, cfn.Config) else pipeline
        assert isinstance(self._pipeline, PolicyDeployment), (
            f'PolicyServer requires a PolicyDeployment, got {type(self._pipeline).__name__}'
        )
        self._pipeline.local.to_spec()
        self._source = self._pipeline.source
        # Set by ``serve`` before any wire binds, and closed when it returns.
        self._checkpoint_id: str | None = None
        self._model: Model | None = None

        self.idle_timeout_min = idle_timeout_min
        self._active_sessions = 0
        self._last_activity = time.monotonic()
        # Backend calls run in a worker thread, so the event loop keeps servicing other connections, but are
        # serialized here: sessions may share one backend client, which concurrent calls would corrupt.
        self._infer_lock = asyncio.Lock()

        # Set while ``serve`` runs; ``shutdown`` reaches the loop from another thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None

        # ``None`` serves open, so a broken secret must not reach that path by accident. Empty would read
        # as open; anything an ``Authorization`` header cannot carry — a newline off the end of a file, a
        # non-ASCII byte — gates the server against everybody, because no client can send the value back.
        if auth_token is not None and not (auth_token and all('!' <= c <= '~' for c in auth_token)):
            raise ValueError('auth_token must be non-empty printable ASCII without spaces; pass None to serve open')
        self._auth_token = auth_token

        self._api = APIRouter()
        self._api.get(wire.MODELS_PATH, dependencies=[Depends(self._require_http_auth)])(self.get_models)

    @property
    def api(self) -> APIRouter:
        """The server's own HTTP routes: the model route, which answers the one checkpoint this server serves."""
        return self._api

    def _token_matches(self, authorization: str | None) -> bool:
        if self._auth_token is None:
            return True
        if authorization is None:
            return False
        # Compared as bytes because a header carries any byte the peer sends: ``compare_digest`` raises on
        # a non-ASCII ``str``, which would answer a malformed header with a 500 instead of a refusal.
        return hmac.compare_digest(authorization.encode(), bearer(self._auth_token).encode())

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        """Whether the session headers carry the bearer token this server gates on."""
        return self._token_matches(headers.get(AUTH_HEADER.lower()))

    def _require_http_auth(self, authorization: str | None = Header(default=None, alias=AUTH_HEADER)) -> None:
        if not self._token_matches(authorization):
            raise HTTPException(status_code=401, detail='Invalid or missing bearer token')

    async def get_models(self) -> dict:
        return {wire.MODELS_KEY: [self._checkpoint_id]}

    def _session_pipeline(self, params: dict[str, Any]) -> PolicyDeployment:
        """The launch pipeline, or a per-session variant with ``params`` applied as config overrides."""
        if not params:
            return self._pipeline
        if self._pipeline_cfg is None:
            raise ValueError(
                'Session params require a config-launched pipeline; this server was launched from an '
                'instantiated PolicyDeployment'
            )
        # ``override_data``: values came off the wire, so a string stays a string and never names a
        # Python object to import.
        pipeline = self._pipeline_cfg.override_data(**params).instantiate()
        if pipeline.source != self._source:
            raise ValueError('Session params must not change the model source; it is fixed at launch')
        return pipeline

    async def _answer_observations(
        self, conn: server_wire.ServerConnection, infer: Callable[[Obs], Any], session_id: str
    ) -> None:
        """Answer observations until the client ends this session or disconnects."""
        while True:
            message = await conn.receive()
            self._last_activity = time.monotonic()
            timing = _ServedTiming()
            with timing.phase(protocol.TIMING_DECODE):
                request = deserialise(message)
                if request[protocol.SESSION_ID] != session_id:
                    raise ValueError('The session ID does not belong to this connection')
                if request.get(protocol.END_SESSION) is True:
                    return
                raw_obs = request[protocol.OBSERVATION]
            try:
                # Plain acquire, not the keepalive helper: the client is awaiting a ``result`` and would
                # mis-parse a ``waiting`` message. Its ``infer_timeout`` bounds the wait.
                with timing.phase(protocol.TIMING_QUEUED):
                    await self._infer_lock.acquire()
                try:
                    with timing.phase(protocol.TIMING_INFER):
                        work = asyncio.create_task(asyncio.to_thread(timing.infer, infer, raw_obs))
                        try:
                            actions = await asyncio.shield(work)
                        except asyncio.CancelledError:
                            # Session cleanup must wait for the worker that still uses its state.
                            await work
                            raise
                finally:
                    self._infer_lock.release()
                answer = serialise({protocol.RESULT: actions, protocol.TIMING: timing.report()})
                await conn.send(answer)
            except wire.PeerDisconnected:
                raise
            except Exception as e:
                logger.error(f'Error processing message: {e}', exc_info=True)
                await conn.send(serialise({protocol.ERROR: str(e)}))

    async def _serve_session(self, conn: server_wire.ServerConnection):
        logger.info(f'Connected to {conn.peer}')

        self._active_sessions += 1
        self._last_activity = time.monotonic()
        try:
            model = self._model
            assert model is not None, 'A session arrived before the model loaded'
            pipeline = self._session_pipeline(_session_params(conn.query_params))
            session_id = uuid4().hex
            meta = {
                **conn.served_address.meta,
                **model.meta(),
                **(pipeline.codec.meta if pipeline.codec is not None else {}),
                **pipeline.local.meta(),
                offboard_keys.CHECKPOINT_ID: self._checkpoint_id,
                offboard_keys.LOCAL_STACK: pipeline.local.to_spec(),
                offboard_keys.COMPRESS_IMAGES: pipeline.compress_images,
                offboard_keys.POSITRONIC_VERSION: _pkg_version('positronic'),
            }
            infer = partial(model, session_id=session_id)
            infer = telemetry.traced(protocol.MODEL_CALL)(infer)
            if pipeline.codec is not None:
                infer = pipeline.codec.wrap(infer)
            try:
                await conn.send(
                    serialise({
                        protocol.STATUS: protocol.ServerStatus.READY,
                        protocol.PROTOCOL_VERSION: protocol.CURRENT_VERSION,
                        protocol.META: meta,
                        protocol.SESSION_ID: session_id,
                    })
                )
                await self._answer_observations(conn, infer, session_id)
            finally:
                async with self._infer_lock:
                    await asyncio.to_thread(model.end_session, session_id)
            await conn.send(serialise({protocol.SESSION_ID: session_id, protocol.END_SESSION: True}))

        except wire.PeerDisconnected:
            logger.info('Client disconnected')
        except Exception as e:
            logger.error(f'Failed session: {e}', exc_info=True)
            try:
                await conn.send(serialise({protocol.STATUS: protocol.ServerStatus.ERROR, protocol.ERROR: str(e)}))
                await conn.refuse(str(e))
            except wire.PeerDisconnected:
                logger.debug('The client was gone before the error reached it', exc_info=True)
            except Exception:
                logger.error(f'Failed to tell {conn.peer} its session failed: {e}', exc_info=True)
        finally:
            self._active_sessions = max(0, self._active_sessions - 1)
            self._last_activity = time.monotonic()

    def _load(self) -> None:
        self._checkpoint_id = self._source.checkpoint_id()
        logger.info(f'Loading checkpoint {self._checkpoint_id}')
        self._model = self._source.load(self._checkpoint_id, logger.info)

    async def _idle_watchdog(self):
        """Return once no session has touched the server for ``idle_timeout_min``."""
        assert self.idle_timeout_min is not None
        timeout_s = self.idle_timeout_min * 60
        poll = min(timeout_s, 30)
        while True:
            await asyncio.sleep(poll)
            if self._active_sessions > 0:
                continue
            idle = time.monotonic() - self._last_activity
            if idle >= timeout_s:
                logger.warning(f'No activity for {idle:.0f}s (idle timeout {timeout_s:.0f}s); shutting down server')
                return

    @staticmethod
    def _raise_first_wire_failure(started: Sequence[server_wire.Wire], outcomes: Sequence[Any]):
        """Raise the first wire that ended on an error, and log every other one."""
        failed = [(w, e) for w, e in zip(started, outcomes, strict=True) if isinstance(e, Exception)]
        # Only one failure can raise; this logs the rest, and nothing else does.
        for w, error in failed[1:]:
            logger.error(f'{type(w).__name__} also failed: {error}', exc_info=error)
        if failed:
            # A wire that ended on an error raises; a silent return reads as a shutdown.
            raise failed[0][1]

    def serve(self, wires: Sequence[server_wire.Wire], on_ready: Callable[[], None] | None = None):
        """Serve sessions on every wire in ``wires``, until one of them ends or the server goes idle.

        Every wire shares this server's model and inference lock. ``on_ready`` runs on the server's
        own loop once every wire has bound; a caller that asked for port 0 reads the port there.
        """
        if not wires:
            raise ValueError('wires must hold at least one wire; a server with none binds nothing and answers nobody')

        async def _run():
            self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
            # A wire binds when it starts; the ``finally`` stops every started one, even when a later one cannot bind.
            started: list[server_wire.Wire] = []
            serving: list[asyncio.Task] = []
            ending: list[asyncio.Task] = []
            try:
                for w in wires:
                    await w.start(self._serve_session, self._authorized, self.api)
                    started.append(w)
                self._last_activity = time.monotonic()
                if on_ready is not None:
                    on_ready()
                serving = [asyncio.create_task(w.serve()) for w in started]
                # What else ends the server: a caller's ``shutdown``, and the idle timeout.
                ending = [asyncio.create_task(self._stop.wait())]
                if self.idle_timeout_min and self.idle_timeout_min > 0:
                    ending.append(asyncio.create_task(self._idle_watchdog()))
                await asyncio.wait(serving + ending, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in ending:
                    task.cancel()
                for w in started:
                    await w.stop()
                # Each wire ends the sessions it carries before this returns and the model closes.
                outcomes = await asyncio.gather(*serving, return_exceptions=True)

            self._raise_first_wire_failure(started, outcomes)

        try:
            self._load()
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')
        finally:
            self._loop, self._stop = None, None
            if self._model is not None:
                self._model.close()
                self._model = None

    def shutdown(self):
        """Ask a running ``serve`` to end, from any thread. A server that is not serving ignores it."""
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)


# Named rather than positional: configuronic addresses an override by parameter name.
websocket = cfn.Config(
    websocket_wire.WebsocketWire, served_address=cfn.Config(server_wire.ServedHostPort, host='0.0.0.0', port=8000)
)
grpc = cfn.Config(grpc_wire.GrpcWire, served_address=cfn.Config(server_wire.ServedHostPort, host='0.0.0.0', port=8001))


@cfn.config()
def socket_at(uds: str) -> websocket_wire.ServedUnixSocket:
    """The Unix socket a wire binds, named on the command line."""
    return websocket_wire.ServedUnixSocket(Path(uds))


@cfn.config(websocket=websocket, grpc=None, idle_timeout_min=None)
def serve(
    pipeline: cfn.Config,
    websocket: server_wire.Wire | None,
    grpc: server_wire.Wire | None,
    idle_timeout_min: float | None,
):
    """The CLI entry point every vendor server exposes: bind ``pipeline``, and the commands are configs of this.

    Everything the served model is — codec, source, checkpoint — is reached through the pipeline
    itself. GR00T names its checkpoints with ``--pipeline.source.model_source=...``; LeRobot and OpenPI
    use ``--pipeline.source.checkpoints_dir=...``. The server serves one checkpoint of them, chosen here.

    Each wire carries the address it binds, and this binds what it is given::

        --websocket.served_address.port=9000
        --websocket.served_address=@positronic.offboard.server.socket_at --websocket.served_address.uds=/run/p.sock
        --grpc=@positronic.offboard.server.grpc --grpc.served_address.port=8001

    The bearer token comes from ``AUTH_TOKEN_ENV``; a flag would put a secret in the process arguments.
    Unset serves open.
    """
    server = PolicyServer(pipeline, idle_timeout_min=idle_timeout_min, auth_token=os.environ.get(AUTH_TOKEN_ENV))
    server.serve([w for w in (websocket, grpc) if w is not None])
