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

from positronic.cfg.policy import action as act_cfg
from positronic.cfg.policy import observation as obs_cfg
from positronic.offboard.serialisation import deserialise, serialise
from positronic.policy import DecodedEncodedPolicy, Policy
from positronic.policy.lerobot import LerobotPolicy
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging

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

    async def get_policy(self, checkpoint_id: str) -> Policy:
        async with self._lock:
            if self.current_checkpoint_id != checkpoint_id:
                logger.info(f'Switching policy from {self.current_checkpoint_id} to {checkpoint_id}')

                while self.active_sessions > 0:  # Wait for all active sessions to finish
                    logger.info(f'Waiting for {self.active_sessions} sessions to finish...')
                    await self._condition.wait()

                if self.current_policy:
                    logger.info('Unloading current policy')
                    self.current_policy.close()

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
    def __init__(
        self,
        policy_factory: Callable[[str], PreTrainedPolicy],
        observation_encoder,
        action_decoder,
        checkpoints_dir: str | Path,
        checkpoint: str | None = None,
        host: str = '0.0.0.0',
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
        device: str | None = None,
    ):
        self.policy_factory = policy_factory
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.device = device or _detect_device()

        self.metadata = metadata or {}
        self.metadata.update(host=host, port=port, device=self.device)

        # Initialize Policy Manager
        self.policy_manager = _PolicyManager(self._load_policy)

        # Initialize FastAPI
        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{checkpoint_id}')(self.websocket_endpoint)

    def _load_policy(self, checkpoint_id: str) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')

        base_meta = {'checkpoint_path': checkpoint_path, **self.metadata}
        policy = self.policy_factory(checkpoint_path)
        if hasattr(policy, 'metadata') and policy.metadata:
            base_meta.update(policy.metadata)

        base = LerobotPolicy(policy, self.device)
        return DecodedEncodedPolicy(
            base, encoder=self.observation_encoder.encode, decoder=self.action_decoder.decode, extra_meta=base_meta
        )

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
                logger.error('Checkpoint not found: %s', checkpoint_id)
                await websocket.send_bytes(serialise({'error': 'Checkpoint not found'}))
                await websocket.close(code=1008, reason='Checkpoint not found')
                return None

            return checkpoint_id

        if self.checkpoint:
            checkpoint_id = str(self.checkpoint).strip('/')
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                logger.error('Configured checkpoint not found: %s', checkpoint_id)
                await websocket.send_bytes(serialise({'error': 'Configured checkpoint not found'}))
                await websocket.close(code=1008, reason='Configured checkpoint not found')
                return None

            logger.info(f'Using configured checkpoint: {checkpoint_id}')
            return checkpoint_id

        try:
            checkpoint_id = get_latest_checkpoint(self.checkpoints_dir)
            logger.info(f'Using latest checkpoint: {checkpoint_id}')
            return checkpoint_id
        except Exception:
            logger.exception('Failed to get latest checkpoint.')
            await websocket.close(code=1008, reason='No checkpoints available')
            return None

    async def websocket_endpoint(self, websocket: WebSocket, checkpoint_id: str | None = None):
        await websocket.accept()

        checkpoint_id = await self._resolve_checkpoint_id(websocket, checkpoint_id)
        if not checkpoint_id:
            return

        logger.info(f'Connected to {websocket.client} requesting {checkpoint_id}')

        try:
            # Request policy from manager (may wait for other sessions)
            policy = await self.policy_manager.get_policy(checkpoint_id)

            try:
                policy.reset()
                await websocket.send_bytes(serialise({'meta': policy.meta}))  # Send Metadata

                # Inference Loop
                try:
                    while True:
                        message = await websocket.receive_bytes()
                        try:
                            obs = deserialise(message)
                            action = policy.select_action(obs)
                            await websocket.send_bytes(serialise({'result': action}))
                        except Exception as e:
                            logger.error(f'Error processing message: {e}')
                            error_response = {'error': str(e)}
                            await websocket.send_bytes(serialise(error_response))
                except WebSocketDisconnect:
                    logger.info('Client disconnected')

            finally:
                # Release session when done
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


@cfn.config(
    policy_factory=act,
    observation_encoder=obs_cfg.eepose,
    action_decoder=act_cfg.absolute_position,
    checkpoint=None,
    port=8000,
    host='0.0.0.0',
)
def main(
    policy_factory: Callable[[str], PreTrainedPolicy],
    checkpoints_dir: str,
    checkpoint: str | None,
    observation_encoder,
    action_decoder,
    port: int,
    host: str,
):
    """
    Starts the inference server with the given policy.
    """
    checkpoints_dir = str(pos3.download(checkpoints_dir))
    server = InferenceServer(
        policy_factory, observation_encoder, action_decoder, checkpoints_dir, checkpoint, host=host, port=port
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info('Server stopped by user')


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(main)
