import asyncio
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.policy import Codec, Policy, RecordingCodec
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.policy import LerobotPolicy, _detect_device, make_processors

logger = logging.getLogger(__name__)


class _PolicyManager:
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


class InferenceServer:
    """LeRobot 0.4.x inference server with singleton policy manager.

    Auto-detects policy type from checkpoint config and creates the appropriate
    preprocessor/postprocessor. Works with SmolVLA, ACT, Diffusion, or any
    lerobot 0.4.x policy.
    """

    def __init__(
        self,
        policy_factory: Callable[[str], PreTrainedPolicy],
        codec: Codec | None,
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

        self.policy_manager = _PolicyManager(self._load_policy)

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

        preprocessor, postprocessor = make_processors(policy.config, checkpoint_path)
        return LerobotPolicy(policy, preprocessor, postprocessor, self.device, extra_meta=base_meta)

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
                policy = self.codec.wrap(base_policy) if self.codec else base_policy
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


def _default_policy_factory(checkpoint_path: str) -> PreTrainedPolicy:
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    policy_cls = get_policy_class(config.type)
    return policy_cls.from_pretrained(checkpoint_path)


@cfn.config(
    policy_factory=_default_policy_factory,
    codec=lerobot_codecs.ee,
    checkpoint=None,
    port=8000,
    host='0.0.0.0',
    recording_dir=None,
)
def main(
    policy_factory: Callable[[str], PreTrainedPolicy],
    checkpoints_dir: str,
    checkpoint: str | None,
    codec,
    port: int,
    host: str,
    recording_dir: str | None,
):
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
