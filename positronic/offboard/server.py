"""The inference server: serves a policy pipeline (see ``positronic.policy.spec``) over the offboard protocol."""

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from importlib.metadata import version as _pkg_version
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.datastructures import QueryParams

from positronic.policy import Codec, Policy, Recorder
from positronic.policy.spec import SEQ, ModelSource, Pipeline, split
from positronic.utils.serialization import deserialise, serialise

logger = logging.getLogger(__name__)


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
                await websocket.send_bytes(serialise({'status': 'waiting', 'message': message}))


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
                        await websocket.send_bytes(serialise({'status': 'waiting', 'message': message}))

                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                    except TimeoutError:
                        continue

                if self.current_policy:
                    logger.info('Unloading current policy')
                    self.current_policy.close()
                    # Empty the slot before loading: a failed load must not leave the closed policy
                    # cached under the old id, where the next session would get it back.
                    self.current_policy = None
                    self.current_checkpoint_id = None

                if websocket:
                    await websocket.send_bytes(
                        serialise({'status': 'loading', 'message': f'Loading checkpoint {checkpoint_id}...'})
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
        """Sync callback for the loader thread, marshaling ``loading`` frames onto the event loop.

        Blocks the loader until each frame is on the wire, so a message emitted at the very end of a
        load cannot overtake the ``ready`` that follows it and be read as the first inference result.
        """
        if websocket is None:
            return None
        loop = asyncio.get_running_loop()

        def on_progress(msg: str) -> None:
            asyncio.run_coroutine_threadsafe(
                websocket.send_bytes(serialise({'status': 'loading', 'message': msg})), loop
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


class PolicyServer:
    """Serves a policy pipeline: one wrapper chain with a ``remote`` marker, closed by a ``ModelSource``
    (see ``positronic.policy.spec``).

    The half right of the marker wraps the model here; the half left of it is published as the
    ``local_stack`` spec in the ``ready`` handshake for the rig to build. The source is the only
    model loader and is fixed at launch.

    When ``pipeline`` is a ``cfn.Config``, query params on the session websocket URL become dotted
    overrides into the pipeline config (e.g. ``?codec.fps=10``), applied and instantiated per session.
    Values must be JSON literals (unparseable values pass through as strings) and are applied with
    ``Config.override_data``, so a param can tune an argument but never name a Python object to
    import; params that change the model source are rejected too. A server built from an
    already-instantiated ``Pipeline`` rejects all session params.

    The WebSocket session flow is:
        accept → session params → resolve → load via manager → remote-half wrap → reset → inference loop

    On startup (before accepting connections): resolve(None) → load → warmup.

    The default checkpoint is resolved exactly once, at startup: whatever ``source.resolve(None)``
    picks (the latest checkpoint, or the configured one) is pinned and reused for every request
    that does not name an explicit checkpoint. A running server therefore never auto-switches to a
    newer checkpoint that lands in checkpoints_dir after startup. Clients can still load a
    specific checkpoint on demand via /api/v1/session/{model_id}.
    """

    def __init__(
        self,
        pipeline: cfn.Config | Pipeline,
        host: str = '0.0.0.0',
        port: int = 8000,
        recording_dir: str | None = None,
        idle_timeout_min: float | None = None,
    ):
        self._pipeline_cfg = pipeline if isinstance(pipeline, cfn.Config) else None
        self._pipeline = pipeline.instantiate() if isinstance(pipeline, cfn.Config) else pipeline
        assert isinstance(self._pipeline, Pipeline), (
            f'PolicyServer serves a policy pipeline closed by a model source, got {type(self._pipeline).__name__}'
        )
        local, self._remote = split(self._pipeline)
        # Resolved eagerly so a pipeline whose local half is not deliverable fails at startup,
        # not at a client's connect. An empty half is an explicit "no glue needed" declaration.
        self._local_spec = local.to_spec() if local is not None else {SEQ: []}
        self._source = self._pipeline.source
        self._manager = PolicyManager(self._source)
        self.host = host
        self.port = port
        self.metadata: dict[str, Any] = {'host': host, 'port': port}
        # Synced once; a fresh ``Recorder`` is created per websocket session so
        # concurrent sessions never share a stream or recorder state.
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None

        self.idle_timeout_min = idle_timeout_min
        self._active_sessions = 0
        self._last_activity = time.monotonic()
        # Backend calls — inference and the per-session reset in ``new_session`` — run in a worker thread
        # (so the event loop keeps servicing other connections and keepalives) but are serialized under this lock:
        # sessions may share one backend client/connection, which concurrent calls would corrupt.
        self._infer_lock = asyncio.Lock()

        # The default checkpoint, resolved once at startup (see class docstring). Pinned
        # here so a request without an explicit checkpoint never re-resolves the latest.
        self._default_id: str | None = None

        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.default_session)
        # ``:path`` so an id that is itself a path — a HuggingFace repo, say — opens under the same
        # name ``/api/v1/models`` advertises.
        self.app.websocket('/api/v1/session/{model_id:path}')(self.model_session)

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
        # ``override_data``: the values came off the wire, so a string must stay a string and never
        # name a Python object to import.
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
            local, remote_half = split(pipeline)
            local_spec = local.to_spec() if local is not None else {SEQ: []}

            # No explicit checkpoint requested -> serve the one pinned at startup
            # (resolved once) rather than re-resolving the latest on every request.
            rid = self._source.resolve(model_id) if model_id is not None else self._default_id
            assert rid is not None
            policy = await self._manager.get_policy(rid, websocket)
            if self._recording_dir is not None:
                # Tap both sides of the remote half so one recording holds the obs/action at the
                # wire boundary ('raw') and the encoded obs / raw model output ('inference').
                rec = Recorder(self._recording_dir)
                if remote_half is not None:
                    served = (rec.tap('raw') | remote_half | rec.tap('inference')).wrap(policy)
                else:
                    served = rec.tap('inference').wrap(policy)
            else:
                served = remote_half.wrap(policy) if remote_half is not None else policy
            # ``new_session`` resets the backend client, which sessions may share; hold the inference lock
            # so the reset can't interleave with an in-flight ``session(raw_obs)`` running in another worker thread.
            # Acquire it with keepalives so queuing behind a peer's slow inference doesn't trip the handshake timeout.
            await _acquire_with_keepalives(self._infer_lock, websocket, 'Waiting for inference slot')
            try:
                session = await asyncio.to_thread(served.new_session)
            finally:
                self._infer_lock.release()
            assert session is not None
            # ``served.meta`` is the static baseline; ``session.meta`` overlays per-episode
            # specifics and wins on conflict. The local-stack declaration and the server's
            # positronic version are server-authoritative, so they merge last.
            meta = {
                **self.metadata,
                **self._source.meta(rid),
                'checkpoint_id': rid,
                **served.meta,
                **session.meta,
                'local_stack': local_spec,
                'positronic_version': _pkg_version('positronic'),
            }
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))

            try:
                while True:
                    message = await websocket.receive_bytes()
                    self._last_activity = time.monotonic()
                    try:
                        raw_obs = deserialise(message)
                        # Plain acquire, not the keepalive helper: past the handshake the client awaits a
                        # ``result`` and would mis-parse a ``waiting`` keepalive; the wait is bounded client-side
                        # by ``infer_timeout``.
                        async with self._infer_lock:
                            actions = await asyncio.to_thread(session, raw_obs)
                        await websocket.send_bytes(serialise({'result': actions}))
                    except Exception as e:
                        logger.error(f'Error processing message: {e}', exc_info=True)
                        await websocket.send_bytes(serialise({'error': str(e)}))
            except WebSocketDisconnect:
                logger.info('Client disconnected')

        except Exception as e:
            logger.error(f'Failed session: {e}', exc_info=True)
            try:
                await websocket.send_bytes(serialise({'status': 'error', 'error': str(e)}))
                await websocket.close(code=1008, reason=str(e)[:100])
            except Exception:
                logger.debug('Failed to send error to client', exc_info=True)
        finally:
            self._active_sessions = max(0, self._active_sessions - 1)
            self._last_activity = time.monotonic()
            try:
                if session is not None:
                    # Session close may do backend I/O (e.g. a reset round-trip over the vendor's own
                    # socket) — keep it off the event loop, and never let it swallow the manager release.
                    await asyncio.to_thread(session.close)
            finally:
                if policy is not None:
                    await self._manager.release_session()

    async def _warmup(self, policy: Policy):
        """Run one warmup inference through the launch codec's ``dummy_encoded()``. Non-fatal on failure."""
        if not isinstance(self._remote, Codec):
            return
        session = None
        try:
            logger.info('Running warmup inference...')
            session = policy.new_session()
            await asyncio.to_thread(session, self._remote.dummy_encoded())
            logger.info('Warmup inference complete')
        except Exception:
            logger.warning('Warmup inference failed (non-fatal)', exc_info=True)
        finally:
            if session is not None:
                session.close()

    async def _startup(self):
        # Pin whatever resolves now (latest, or the configured checkpoint) as the
        # default for subsequent requests, so the latest is resolved exactly once.
        self._default_id = self._source.resolve(None)
        logger.info(f'Pinned default checkpoint at startup: {self._default_id}')
        policy = await self._manager.get_policy(self._default_id)
        await self._warmup(policy)

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
