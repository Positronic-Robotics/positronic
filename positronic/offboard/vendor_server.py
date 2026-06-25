"""Base class for vendor inference servers (GR00T, OpenPI, DreamZero, LeRobot)."""

import asyncio
import hmac
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pos3
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, WebSocketException, status

from positronic.policy import Codec, Policy, Recorder
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.serialization import deserialise, serialise

logger = logging.getLogger(__name__)


class PolicyManager:
    """Manages the lifecycle of a single active policy.

    Ensures only one policy is loaded at a time. Waits for all active sessions
    to finish before switching policies.
    """

    def __init__(self, loader: Callable[[str], Policy]):
        self.loader = loader
        self.current_checkpoint_id: str | None = None
        self.current_policy: Policy | None = None
        self.active_sessions: int = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def get_policy(self, checkpoint_id: str, websocket: WebSocket | None = None) -> Policy:
        async with self._lock:
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

                if websocket:
                    await websocket.send_bytes(
                        serialise({'status': 'loading', 'message': f'Loading checkpoint {checkpoint_id}...'})
                    )

                logger.info(f'Loading policy {checkpoint_id}')
                self.current_policy = await asyncio.to_thread(self.loader, checkpoint_id)
                self.current_checkpoint_id = checkpoint_id

            if websocket:
                self.active_sessions += 1
            return self.current_policy

    async def release_session(self):
        async with self._lock:
            self.active_sessions -= 1
            if self.active_sessions == 0:
                self._condition.notify_all()


def resolve_checkpoint(checkpoints_dir: str, configured: str | None, requested: str | None) -> str:
    """Resolve a checkpoint ID from an explicit request, a configured default, or latest available."""
    if requested:
        available = list_checkpoints(checkpoints_dir)
        if requested not in available:
            raise ValueError(f'Checkpoint not found: {requested}. Available: {available}')
        return requested

    if configured:
        checkpoint_id = str(configured).strip('/')
        available = list_checkpoints(checkpoints_dir)
        if checkpoint_id not in available:
            raise ValueError(f'Configured checkpoint not found: {checkpoint_id}. Available: {available}')
        logger.info(f'Using configured checkpoint: {checkpoint_id}')
        return checkpoint_id

    checkpoint_id = get_latest_checkpoint(checkpoints_dir)
    logger.info(f'Using latest checkpoint: {checkpoint_id}')
    return checkpoint_id


class VendorServer(ABC):
    """Base class for vendor inference servers.

    Provides the FastAPI app, WebSocket inference loop, codec wrapping,
    startup/warmup lifecycle, and serve entrypoint. Subclasses implement
    three hooks:

        resolve_model(model_id, websocket) → (handle, extra_meta)
        create_policy(handle) → Policy
        get_models() → dict

    Optional overrides: warmup(), shutdown_model(), release_policy().

    The WebSocket session flow is:
        accept → resolve_model → create_policy → codec.wrap → reset → inference loop

    On startup (before accepting connections):
        resolve_model(None) → create_policy → reset → warmup

    The default checkpoint is resolved exactly once, at startup: whatever
    resolve_model(None) picks (the latest checkpoint, or the configured one) is
    pinned and reused for every request that does not name an explicit checkpoint.
    A running server therefore never auto-switches to a newer checkpoint that lands
    in checkpoints_dir after startup. Clients can still load a specific checkpoint on
    demand via /api/v1/session/{model_id}.
    """

    def __init__(
        self,
        codec: Codec | None,
        host: str = '0.0.0.0',
        port: int = 8000,
        recording_dir: str | None = None,
        idle_timeout_min: float | None = None,
    ):
        self.codec = codec
        self.host = host
        self.port = port
        # Synced once; a fresh ``Recorder`` is created per websocket session so
        # concurrent sessions never share a stream or recorder state.
        self._recording_dir = pos3.sync(recording_dir) if recording_dir else None

        self.idle_timeout_min = idle_timeout_min
        self._active_sessions = 0
        self._last_activity = time.monotonic()

        # The default checkpoint, resolved once at startup (see class docstring). Pinned
        # here so a request without an explicit checkpoint never re-resolves the latest.
        self._pinned_default_model: str | None = None

        self.metadata: dict[str, Any] = {}

        # Bearer token gating every endpoint, injected as the container's AUTH_TOKEN env var by
        # serve.sh. Absent -> the server is open. Validated in-process because Nebius `--auth token`
        # gates HTTP but does not proxy the inference WebSocket.
        self._auth_token = os.environ.get('AUTH_TOKEN') or None

        self.app = FastAPI()
        http_auth = [Depends(self._require_http_auth)]
        ws_auth = [Depends(self._require_ws_auth)]
        self.app.get('/api/v1/models', dependencies=http_auth)(self.get_models)
        self.app.websocket('/api/v1/session', dependencies=ws_auth)(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{model_id}', dependencies=ws_auth)(self.websocket_endpoint)

    @abstractmethod
    async def resolve_model(self, model_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        """Ensure model/subprocess is running. Return (handle, extra_metadata)."""

    @abstractmethod
    def create_policy(self, model_handle: Any) -> Policy:
        """Wrap vendor client into a Policy. No codec — base handles that."""

    @abstractmethod
    async def get_models(self) -> dict:
        """Return available models."""

    async def warmup(self, policy: Policy):
        """Run one warmup inference. Default uses codec.dummy_encoded(). Non-fatal on failure."""
        if not self.codec:
            return
        session = None
        try:
            logger.info('Running warmup inference...')
            session = policy.new_session()
            await asyncio.to_thread(session, self.codec.dummy_encoded())
            logger.info('Warmup inference complete')
        except Exception:
            logger.warning('Warmup inference failed (non-fatal)', exc_info=True)
        finally:
            if session is not None:
                session.close()

    def shutdown_model(self):  # noqa: B027
        """Called on server shutdown. Default: no-op."""

    async def release_policy(self, model_handle: Any):  # noqa: B027
        """Called when a WebSocket session ends. Default: no-op."""

    @staticmethod
    def _progress_sender(websocket: WebSocket | None):
        async def send_progress(msg: str):
            if websocket is not None:
                await websocket.send_bytes(serialise({'status': 'loading', 'message': msg}))

        return send_progress

    def _authorized(self, authorization: str | None) -> bool:
        if self._auth_token is None:
            return True
        return authorization is not None and hmac.compare_digest(authorization, f'Bearer {self._auth_token}')

    def _require_http_auth(self, authorization: str | None = Header(default=None)):
        if not self._authorized(authorization):
            raise HTTPException(status_code=401, detail='Invalid or missing bearer token')

    async def _require_ws_auth(self, websocket: WebSocket):
        if not self._authorized(websocket.headers.get('authorization')):
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    async def websocket_endpoint(self, websocket: WebSocket, model_id: str | None = None):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {model_id or "default"}')

        self._active_sessions += 1
        self._last_activity = time.monotonic()
        model_handle = None
        session = None
        try:
            # No explicit checkpoint requested -> serve the one pinned at startup
            # (resolved once) rather than re-resolving the latest on every request.
            requested_model = model_id if model_id is not None else self._pinned_default_model
            model_handle, extra_meta = await self.resolve_model(requested_model, websocket)
            base_policy = self.create_policy(model_handle)
            if self._recording_dir is not None:
                # Tap both sides of the codec so one recording holds the obs/action at the
                # wire boundary ('raw') and the encoded obs / raw model output ('inference').
                rec = Recorder(self._recording_dir)
                if self.codec:
                    policy = (rec.tap('raw') | self.codec | rec.tap('inference')).wrap(base_policy)
                else:
                    policy = rec.tap('inference').wrap(base_policy)
            else:
                policy = self.codec.wrap(base_policy) if self.codec else base_policy
            session = policy.new_session()
            # ``policy.meta`` is the static baseline; ``session.meta`` overlays
            # per-episode specifics and wins on conflict.
            meta = {**self.metadata, **extra_meta, **policy.meta, **session.meta}
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))

            try:
                while True:
                    message = await websocket.receive_bytes()
                    self._last_activity = time.monotonic()
                    try:
                        raw_obs = deserialise(message)
                        actions = session(raw_obs)
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
            if session is not None:
                session.close()
            if model_handle is not None:
                await self.release_policy(model_handle)

    async def _startup(self):
        model_handle, meta = await self.resolve_model(None, websocket=None)
        # Pin whatever was resolved now (latest, or the configured checkpoint) as the
        # default for subsequent requests, so the latest is resolved exactly once.
        self._pinned_default_model = meta.get('checkpoint_id')
        if self._pinned_default_model is not None:
            logger.info(f'Pinned default checkpoint at startup: {self._pinned_default_model}')
        policy = self.create_policy(model_handle)
        await self.warmup(policy)

    async def _idle_watchdog(self, server: uvicorn.Server):
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
            self.shutdown_model()
