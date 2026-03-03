"""Base class for vendor inference servers (GR00T, OpenPI, DreamZero, LeRobot)."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from positronic.policy import Codec, Policy, RecordingCodec
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

    async def get_policy(self, checkpoint_id: str, websocket: WebSocket) -> Policy:
        async with self._lock:
            if self.current_checkpoint_id != checkpoint_id:
                logger.info(f'Switching policy from {self.current_checkpoint_id} to {checkpoint_id}')

                while self.active_sessions > 0:
                    message = f'Waiting for {self.active_sessions} active session(s) to finish...'
                    logger.info(message)
                    await websocket.send_bytes(serialise({'status': 'waiting', 'message': message}))

                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                    except TimeoutError:
                        continue

                if self.current_policy:
                    logger.info('Unloading current policy')
                    self.current_policy.close()

                await websocket.send_bytes(
                    serialise({'status': 'loading', 'message': f'Loading checkpoint {checkpoint_id}...'})
                )

                logger.info(f'Loading policy {checkpoint_id}')
                self.current_policy = self.loader(checkpoint_id)
                self.current_checkpoint_id = checkpoint_id

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
    """

    def __init__(self, codec: Codec | None, host: str = '0.0.0.0', port: int = 8000, recording_dir: str | None = None):
        self.codec = codec
        self.host = host
        self.port = port
        if recording_dir:
            self.codec = RecordingCodec(self.codec, pos3.sync(recording_dir))

        self.metadata: dict[str, Any] = {}

        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{model_id}')(self.websocket_endpoint)

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
        try:
            logger.info('Running warmup inference...')
            await asyncio.to_thread(policy.select_action, self.codec.dummy_encoded())
            logger.info('Warmup inference complete')
        except Exception:
            logger.warning('Warmup inference failed (non-fatal)', exc_info=True)

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

    async def websocket_endpoint(self, websocket: WebSocket, model_id: str | None = None):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {model_id or "default"}')

        model_handle = None
        try:
            model_handle, extra_meta = await self.resolve_model(model_id, websocket)
            base_policy = self.create_policy(model_handle)
            policy = self.codec.wrap(base_policy) if self.codec else base_policy
            policy.reset()
            meta = {**self.metadata, **extra_meta, **policy.meta}
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))

            try:
                while True:
                    message = await websocket.receive_bytes()
                    try:
                        raw_obs = deserialise(message)
                        actions = policy.select_action(raw_obs)
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
            if model_handle is not None:
                await self.release_policy(model_handle)

    async def _startup(self):
        model_handle, _meta = await self.resolve_model(None, websocket=None)
        policy = self.create_policy(model_handle)
        policy.reset()
        await self.warmup(policy)

    def serve(self):
        async def _run():
            await self._startup()
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
            await uvicorn.Server(config).serve()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')
        finally:
            self.shutdown_model()
