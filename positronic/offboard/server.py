"""The inference server: serves a policy pipeline (see ``positronic.policy.spec``) over the offboard protocol."""

import asyncio
import hmac
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from importlib.metadata import version as _pkg_version
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, WebSocketException, status
from starlette.datastructures import QueryParams

from positronic.offboard import keys as offboard_keys
from positronic.policy import Policy, Recorder
from positronic.policy.base import Layer
from positronic.policy.executor import blocking
from positronic.policy.spec import ModelSource, Pipeline, split

from . import protocol
from .protocol import deserialise, serialise

logger = logging.getLogger(__name__)

AUTH_TOKEN_ENV = 'AUTH_TOKEN'

AUTH_HEADER = 'Authorization'


def bearer(token: str) -> str:
    """The ``AUTH_HEADER`` value carrying ``token``."""
    return f'Bearer {token}'


async def _acquire_with_keepalives(lock: asyncio.Lock, websocket: WebSocket | None, message: str):
    """Acquire ``lock``, emitting ``waiting`` keepalives while queued behind another holder.

    A peer may hold the lock for a slow load, first-call compile or inference; a silent wait here
    would trip the client handshake's 30s per-message timeout before ``ready`` is sent.
    """
    while True:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=10.0)
            return
        except TimeoutError:
            if websocket is not None:
                await websocket.send_bytes(
                    serialise({protocol.STATUS: protocol.ServerStatus.WAITING, protocol.MESSAGE: message})
                )


class PolicyManager:
    """Manages the lifecycle of the one policy ``source`` currently has loaded.

    Ensures only one policy is loaded at a time. Waits for all active sessions
    to finish before switching policies.
    """

    def __init__(self, source: ModelSource):
        self._source = source
        self.current_checkpoint_id: str | None = None
        self.current_policy: Policy | None = None
        self.active_sessions: int = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def get_policy(self, checkpoint_id: str, websocket: WebSocket | None = None) -> Policy:
        await _acquire_with_keepalives(self._lock, websocket, 'Waiting for the model slot')
        try:
            if self.current_checkpoint_id != checkpoint_id:
                logger.info(f'Switching policy from {self.current_checkpoint_id} to {checkpoint_id}')

                while self.active_sessions > 0:
                    message = f'Waiting for {self.active_sessions} active session(s) to finish...'
                    logger.info(message)
                    if websocket:
                        await websocket.send_bytes(
                            serialise({protocol.STATUS: protocol.ServerStatus.WAITING, protocol.MESSAGE: message})
                        )

                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                    except TimeoutError:
                        continue

                if self.current_policy:
                    logger.info('Unloading current policy')
                    self.current_policy.close()
                    # Empty the slot first: a failed load must not leave the closed policy under the old id.
                    self.current_policy = None
                    self.current_checkpoint_id = None

                if websocket:
                    await websocket.send_bytes(
                        serialise({
                            protocol.STATUS: protocol.ServerStatus.LOADING,
                            protocol.MESSAGE: f'Loading checkpoint {checkpoint_id}...',
                        })
                    )

                logger.info(f'Loading policy {checkpoint_id}')
                on_progress = self._progress_callback(websocket)
                self.current_policy = await asyncio.to_thread(self._source.load, checkpoint_id, on_progress)
                self.current_checkpoint_id = checkpoint_id

            assert self.current_policy is not None
            if websocket:
                self.active_sessions += 1
            return self.current_policy
        finally:
            self._lock.release()

    @staticmethod
    def _progress_callback(websocket: WebSocket | None) -> Callable[[str], None] | None:
        """Sync callback for the loader thread, marshaling ``loading`` messages onto the event loop.

        Blocks the loader until each message is on the wire, so one emitted at the very end of a load
        cannot overtake the ``ready`` that follows it and be read as the first inference result.
        """
        if websocket is None:
            return None
        loop = asyncio.get_running_loop()

        def on_progress(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(
                websocket.send_bytes(
                    serialise({protocol.STATUS: protocol.ServerStatus.LOADING, protocol.MESSAGE: msg})
                ),
                loop,
            ).result()

        return on_progress

    async def release_session(self):
        async with self._lock:
            self.active_sessions -= 1
            if self.active_sessions == 0:
                self._condition.notify_all()

    def close(self):
        """Close the loaded policy. Runs outside the event loop, at server shutdown."""
        if self.current_policy is not None:
            self.current_policy.close()
            self.current_policy = None
            self.current_checkpoint_id = None


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


def _declared_stack(local: Layer | None) -> dict[str, Any]:
    """The rig-side spec a served pipeline must publish."""
    if local is None:
        raise ValueError(
            'Nothing sits left of the `remote` marker, so the pipeline declares no rig-side stack. Put the '
            'layers the rig runs there, starting with a scheduler such as ChunkedSchedule'
        )
    return local.to_spec()


class PolicyServer:
    """Serves a policy pipeline: one layer chain with a ``remote`` marker, closed by a ``ModelSource``
    (see ``positronic.policy.spec``).

    The half right of the marker wraps the model here; the half left of it is published as the
    ``local_stack`` spec in the ``ready`` handshake for the rig to build, alongside the marker's own
    wire settings. The source is the only model loader and is fixed at launch.

    When ``pipeline`` is a ``cfn.Config``, query params on the session websocket URL become dotted
    overrides into the pipeline config (e.g. ``?codec.fps=10``), applied and instantiated per session.
    Values must be JSON literals (unparseable values pass through as strings) and are applied with
    ``Config.override_data``, so a param can tune an argument but never name a Python object to
    import; params that change the model source are rejected too. A server built from an
    already-instantiated ``Pipeline`` rejects all session params.

    The WebSocket session flow is:
        accept → session params → resolve → load via manager → remote-half wrap → reset → inference loop

    On startup (before accepting connections): resolve(None) → load.

    The default checkpoint is resolved once, at startup, and pinned for every request that names no
    explicit one — a running server never switches to a newer checkpoint that lands later. A request
    for /api/v1/session/{model_id} still loads that one on demand.
    """

    def __init__(
        self,
        pipeline: cfn.Config | Pipeline,
        host: str = '0.0.0.0',
        port: int = 8000,
        recording_dir: str | None = None,
        idle_timeout_min: float | None = None,
        auth_token: str | None = None,
    ):
        self._pipeline_cfg = pipeline if isinstance(pipeline, cfn.Config) else None
        self._pipeline = pipeline.instantiate() if isinstance(pipeline, cfn.Config) else pipeline
        assert isinstance(self._pipeline, Pipeline), (
            f'PolicyServer serves a policy pipeline closed by a model source, got {type(self._pipeline).__name__}'
        )
        local, _, _ = split(self._pipeline)
        # A local half that is missing or cannot be rendered fails at startup, not at a client's connect.
        # The spec itself is built per session, which params may have changed.
        _declared_stack(local)
        self._source = self._pipeline.source
        self._manager = PolicyManager(self._source)
        self.host = host
        self.port = port
        self.metadata: dict[str, Any] = {offboard_keys.HOST: host, offboard_keys.PORT: port}
        # Synced once; each session builds its own ``Recorder`` so concurrent streams never mix.
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None

        self.idle_timeout_min = idle_timeout_min
        self._active_sessions = 0
        self._last_activity = time.monotonic()
        # Backend calls run in a worker thread, so the event loop keeps servicing other connections, but are
        # serialized here: sessions may share one backend client, which concurrent calls would corrupt.
        self._infer_lock = asyncio.Lock()

        self._default_id: str | None = None

        # ``None`` serves open, so a broken secret must not reach that path by accident. Empty would read
        # as open; anything an ``Authorization`` header cannot carry — a newline off the end of a file, a
        # non-ASCII byte — gates the server against everybody, because no client can send the value back.
        if auth_token is not None and not (auth_token and all('!' <= c <= '~' for c in auth_token)):
            raise ValueError('auth_token must be non-empty printable ASCII without spaces; pass None to serve open')
        self._auth_token = auth_token

        self.app = FastAPI()
        http_auth, ws_auth = [Depends(self._require_http_auth)], [Depends(self._require_ws_auth)]
        self.app.get('/api/v1/models', dependencies=http_auth)(self.get_models)
        self.app.websocket('/api/v1/session', dependencies=ws_auth)(self.default_session)
        # ``:path`` so an id that is itself a path (a HuggingFace repo, say) opens under the name
        # ``/api/v1/models`` advertises.
        self.app.websocket('/api/v1/session/{model_id:path}', dependencies=ws_auth)(self.model_session)

    def _authorized(self, authorization: str | None) -> bool:
        if self._auth_token is None:
            return True
        if authorization is None:
            return False
        # Compared as bytes because a header carries any byte the peer sends: ``compare_digest`` raises on
        # a non-ASCII ``str``, which would answer a malformed header with a 500 instead of a refusal.
        return hmac.compare_digest(authorization.encode(), bearer(self._auth_token).encode())

    def _require_http_auth(self, authorization: str | None = Header(default=None, alias=AUTH_HEADER)) -> None:
        if not self._authorized(authorization):
            raise HTTPException(status_code=401, detail='Invalid or missing bearer token')

    async def _require_ws_auth(self, websocket: WebSocket) -> None:
        """Rejects before ``accept()``, so an unauthorized peer never reaches the session handshake."""
        if not self._authorized(websocket.headers.get(AUTH_HEADER)):
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    async def get_models(self) -> dict:
        return {'models': self._source.get_models()}

    def _session_pipeline(self, params: dict[str, Any]) -> Pipeline:
        """The launch pipeline, or a per-session variant with ``params`` applied as config overrides."""
        if not params:
            return self._pipeline
        if self._pipeline_cfg is None:
            raise ValueError(
                'Session params require a config-launched pipeline; this server was launched from an '
                'instantiated Pipeline'
            )
        # ``override_data``: values came off the wire, so a string stays a string and never names a
        # Python object to import.
        pipeline = self._pipeline_cfg.override_data(**params).instantiate()
        if pipeline.source != self._source:
            raise ValueError('Session params must not change the model source; it is fixed at launch')
        return pipeline

    async def default_session(self, websocket: WebSocket):
        """Serves the model pinned at startup. Naming a model is the path's job, so every query param here
        is a pipeline override."""
        await self._serve_session(websocket, None)

    async def model_session(self, websocket: WebSocket, model_id: str):
        await self._serve_session(websocket, model_id)

    async def _serve_session(self, websocket: WebSocket, model_id: str | None):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {model_id or "default"}')

        self._active_sessions += 1
        self._last_activity = time.monotonic()
        policy: Policy | None = None
        session = None
        try:
            pipeline = self._session_pipeline(_session_params(websocket.query_params))
            local, border, remote_half = split(pipeline)
            local_spec = _declared_stack(local)

            rid = self._source.resolve(model_id) if model_id is not None else self._default_id
            assert rid is not None
            policy = await self._manager.get_policy(rid, websocket)
            # A request has no control loop to answer ``None`` to. This goes innermost, so every layer
            # above it sees one call per answer rather than one per call the answer took.
            answered = blocking(policy)
            if self._recording_dir is not None:
                # Tap both sides: 'raw' is the wire boundary, 'inference' the encoded obs and model output.
                rec = Recorder(self._recording_dir)
                if remote_half is not None:
                    served = (rec.tap('raw') | remote_half | rec.tap('inference')).wrap(answered)
                else:
                    served = rec.tap('inference').wrap(answered)
            else:
                served = remote_half.wrap(answered) if remote_half is not None else answered
            # ``new_session`` resets the shared backend client, so it must not interleave with an in-flight
            # inference. Keepalives here: queuing behind a peer would otherwise trip the handshake timeout.
            await _acquire_with_keepalives(self._infer_lock, websocket, 'Waiting for inference slot')
            try:
                session = await asyncio.to_thread(served.new_session)
            finally:
                self._infer_lock.release()
            assert session is not None
            # Later entries win: per-episode session facts over static ones, the server's own last.
            meta = {
                **self.metadata,
                **self._source.meta(rid),
                offboard_keys.CHECKPOINT_ID: rid,
                **session.meta,
                offboard_keys.LOCAL_STACK: local_spec,
                offboard_keys.COMPRESS_IMAGES: border.compress_images,
                offboard_keys.POSITRONIC_VERSION: _pkg_version('positronic'),
            }
            await websocket.send_bytes(serialise({protocol.STATUS: protocol.ServerStatus.READY, protocol.META: meta}))

            try:
                while True:
                    message = await websocket.receive_bytes()
                    self._last_activity = time.monotonic()
                    try:
                        raw_obs = deserialise(message)
                        # Plain acquire, not the keepalive helper: the client is awaiting a ``result`` and
                        # would mis-parse a ``waiting`` message. Its ``infer_timeout`` bounds the wait.
                        async with self._infer_lock:
                            # The server's clock is not the rig's.
                            actions = await asyncio.to_thread(session, raw_obs, time.time_ns())
                        await websocket.send_bytes(serialise({protocol.RESULT: actions}))
                    except Exception as e:
                        logger.error(f'Error processing message: {e}', exc_info=True)
                        await websocket.send_bytes(serialise({protocol.ERROR: str(e)}))
            except WebSocketDisconnect:
                logger.info('Client disconnected')

        except Exception as e:
            logger.error(f'Failed session: {e}', exc_info=True)
            try:
                await websocket.send_bytes(
                    serialise({protocol.STATUS: protocol.ServerStatus.ERROR, protocol.ERROR: str(e)})
                )
                await websocket.close(code=1008, reason=str(e)[:100])
            except Exception:
                logger.debug('Failed to send error to client', exc_info=True)
        finally:
            self._active_sessions = max(0, self._active_sessions - 1)
            self._last_activity = time.monotonic()
            try:
                if session is not None:
                    # Both ends of a session's life touch the backend — close does a reset round-trip — so
                    # it takes the inference lock like ``new_session`` and runs off the event loop. The
                    # nesting keeps a failure here from swallowing the manager release.
                    async with self._infer_lock:
                        await asyncio.to_thread(session.close)
            finally:
                if policy is not None:
                    await self._manager.release_session()

    async def _startup(self):
        self._default_id = self._source.resolve(None)
        logger.info(f'Pinned default checkpoint at startup: {self._default_id}')
        await self._manager.get_policy(self._default_id)

    async def _idle_watchdog(self, server: uvicorn.Server):
        assert self.idle_timeout_min is not None
        timeout_s = self.idle_timeout_min * 60
        poll = min(timeout_s, 30)
        while not server.should_exit:
            await asyncio.sleep(poll)
            if self._active_sessions > 0:
                continue
            idle = time.monotonic() - self._last_activity
            if idle >= timeout_s:
                logger.warning(f'No activity for {idle:.0f}s (idle timeout {timeout_s:.0f}s); shutting down server')
                server.should_exit = True
                return

    def serve(self):
        async def _run():
            await self._startup()
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
            server = uvicorn.Server(config)
            self._last_activity = time.monotonic()
            watchdog = None
            if self.idle_timeout_min and self.idle_timeout_min > 0:
                watchdog = asyncio.create_task(self._idle_watchdog(server))
            try:
                await server.serve()
            finally:
                if watchdog is not None:
                    watchdog.cancel()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')
        finally:
            self._manager.close()


@cfn.config(host='0.0.0.0', port=8000, recording_dir=None, idle_timeout_min=None)
def serve(pipeline: cfn.Config, host: str, port: int, recording_dir: str | None, idle_timeout_min: float | None):
    """The CLI entry point every vendor server exposes: bind ``pipeline``, and the commands are configs of this.

    Only the socket and the recording taps are flags of their own; everything the served model is —
    codec, source, checkpoint directory — is reached through the pipeline itself
    (``--pipeline.source.checkpoints_dir=...``), so each of those values has exactly one name.

    The bearer token gating the server comes from ``AUTH_TOKEN_ENV`` rather than a flag, which would put
    a secret in the process arguments; unset serves open.
    """
    PolicyServer(
        pipeline,
        host=host,
        port=port,
        recording_dir=recording_dir,
        idle_timeout_min=idle_timeout_min,
        auth_token=os.environ.get(AUTH_TOKEN_ENV),
    ).serve()
