import asyncio
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.policy import Codec, Policy, RecordingCodec
from positronic.policy.lerobot import LerobotPolicy
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.backbone import register_all

register_all()

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Select the best available torch device unless one is provided."""
    if torch.cuda.is_available():
        return 'cuda'

    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None:
        is_available = getattr(mps_backend, 'is_available', None)
        is_built = getattr(mps_backend, 'is_built', None)
        if callable(is_available) and is_available():
            if not callable(is_built) or is_built():
                return 'mps'

    return 'cpu'


class _PolicyManager:
    """
    Manages the lifecycle of a single active policy.
    Ensures that only one policy is loaded at a time.
    Waits for all active sessions to finish before switching policies.
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

                # Send waiting status while sessions are active
                while self.active_sessions > 0:
                    message = f'Waiting for {self.active_sessions} active session(s) to finish...'
                    logger.info(message)
                    await websocket.send_bytes(serialise({'status': 'waiting', 'message': message}))

                    try:
                        # Wait with timeout so we can send periodic updates
                        await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                    except TimeoutError:
                        # Timeout is expected - send another update
                        continue

                if self.current_policy:
                    logger.info('Unloading current policy')
                    self.current_policy.close()

                # Send loading status before blocking load operation
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


class InferenceServer:
    """LeRobot inference server with singleton policy manager.

    This server loads policies synchronously (in-process), which means checkpoint
    loading should be reasonably fast (<20s) to avoid WebSocket keepalive timeouts.

    For very large checkpoints or slow S3 downloads, consider:
    - Pre-downloading checkpoints to local storage
    - Using a subprocess-based server with periodic status updates
    - Implementing async checkpoint download (see positronic.offboard.server_utils)

    The server enforces a single active policy at a time, queueing new requests
    until the current policy is unloaded.
    """

    def __init__(
        self,
        policy_factory: Callable[[str], PreTrainedPolicy],
        codec: Codec,
        checkpoints_dir: str | Path,
        checkpoint: str | None = None,
        host: str = '0.0.0.0',
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
        device: str | None = None,
        recording_dir: str | None = None,
    ):
        self.policy_factory = policy_factory
        self.codec = codec
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.device = device or _detect_device()
        if recording_dir:
            self.codec = RecordingCodec(self.codec, pos3.sync(recording_dir))

        self.metadata = metadata or {}
        self.metadata.update(
            host=host,
            port=port,
            device=self.device,
            experiment_name=str(checkpoints_dir).rstrip('/').split('/')[-1] or '',
        )

        # Initialize Policy Manager (loads base policy without codec wrapping)
        self.policy_manager = _PolicyManager(self._load_policy)

        # Initialize FastAPI
        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{checkpoint_id}')(self.websocket_endpoint)

    def _load_policy(self, checkpoint_id: str) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')

        base_meta = {'checkpoint_id': checkpoint_id, 'checkpoint_path': checkpoint_path, **self.metadata}
        policy = self.policy_factory(checkpoint_path)
        if hasattr(policy, 'metadata') and policy.metadata:
            base_meta.update(policy.metadata)

        return LerobotPolicy(policy, self.device, extra_meta=base_meta)

    async def get_models(self):
        try:
            return {'models': list_checkpoints(self.checkpoints_dir)}
        except Exception:
            logger.exception('Failed to list checkpoints.')
            return {'models': []}

    async def _resolve_checkpoint_id(self, websocket: WebSocket, checkpoint_id: str | None) -> str | None:
        if checkpoint_id:
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                error_msg = f'Checkpoint not found: {checkpoint_id}. Available: {available}'
                logger.error(error_msg)
                await websocket.send_bytes(serialise({'status': 'error', 'error': error_msg}))
                await websocket.close(code=1008, reason='Checkpoint not found')
                return None

            return checkpoint_id

        if self.checkpoint:
            checkpoint_id = str(self.checkpoint).strip('/')
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                error_msg = f'Configured checkpoint not found: {checkpoint_id}. Available: {available}'
                logger.error(error_msg)
                await websocket.send_bytes(serialise({'status': 'error', 'error': error_msg}))
                await websocket.close(code=1008, reason='Configured checkpoint not found')
                return None

            logger.info(f'Using configured checkpoint: {checkpoint_id}')
            return checkpoint_id

        try:
            checkpoint_id = get_latest_checkpoint(self.checkpoints_dir)
            logger.info(f'Using latest checkpoint: {checkpoint_id}')
            return checkpoint_id
        except Exception:
            error_msg = f'No checkpoints found in {self.checkpoints_dir}'
            logger.exception(error_msg)
            await websocket.send_bytes(serialise({'status': 'error', 'error': error_msg}))
            await websocket.close(code=1008, reason='No checkpoints available')
            return None

    async def websocket_endpoint(self, websocket: WebSocket, checkpoint_id: str | None = None):
        await websocket.accept()

        checkpoint_id = await self._resolve_checkpoint_id(websocket, checkpoint_id)
        if not checkpoint_id:
            return

        logger.info(f'Connected to {websocket.client} requesting {checkpoint_id}')

        try:
            base_policy = await self.policy_manager.get_policy(checkpoint_id, websocket)
            try:
                policy = self.codec.wrap(base_policy)
                policy.reset()
                await websocket.send_bytes(serialise({'status': 'ready', 'meta': policy.meta}))
                try:
                    while True:
                        message = await websocket.receive_bytes()
                        try:
                            obs = deserialise(message)
                            action = policy.select_action(obs)
                            await websocket.send_bytes(serialise({'result': action}))
                        except Exception as e:
                            logger.error(f'Error processing message: {e}')
                            await websocket.send_bytes(serialise({'error': str(e)}))
                except WebSocketDisconnect:
                    logger.info('Client disconnected')
            finally:
                await self.policy_manager.release_session()
        except (WebSocketDisconnect, Exception) as e:
            logger.info(f'Connection closed: {e}')
            logger.debug(traceback.format_exc())

    def serve(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
        server = uvicorn.Server(config)
        return server.serve()


def act(checkpoint_path: str) -> PreTrainedPolicy:
    policy = ACTPolicy.from_pretrained(checkpoint_path, strict=True)
    policy.metadata = {'type': 'act', 'checkpoint_path': checkpoint_path}
    return policy


@cfn.config(policy_factory=act, codec=lerobot_codecs.ee, checkpoint=None, port=8000, host='0.0.0.0', recording_dir=None)
def main(
    policy_factory: Callable[[str], PreTrainedPolicy],
    checkpoints_dir: str,
    checkpoint: str | None,
    codec,
    port: int,
    host: str,
    recording_dir: str | None,
):
    """
    Starts the inference server with the given policy.
    """
    checkpoints_dir = str(pos3.download(checkpoints_dir))
    server = InferenceServer(
        policy_factory, codec, checkpoints_dir, checkpoint, host=host, port=port, recording_dir=recording_dir
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info('Server stopped by user')


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(main)
