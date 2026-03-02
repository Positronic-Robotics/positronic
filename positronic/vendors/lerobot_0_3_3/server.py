import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import torch
import uvicorn
from fastapi import WebSocket
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.offboard.vendor_server import VendorServer
from positronic.policy import Codec, Policy
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import serialise
from positronic.vendors.lerobot_0_3_3 import codecs as lerobot_codecs
from positronic.vendors.lerobot_0_3_3.backbone import register_all
from positronic.vendors.lerobot_0_3_3.policy import LerobotPolicy

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


class InferenceServer(VendorServer):
    """LeRobot inference server with singleton policy manager.

    This server loads policies synchronously (in-process), which means checkpoint
    loading should be reasonably fast (<20s) to avoid WebSocket keepalive timeouts.

    The server enforces a single active policy at a time, queueing new requests
    until the current policy is unloaded.
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
        super().__init__(codec=codec, host=host, port=port, recording_dir=recording_dir)
        self.policy_factory = policy_factory
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.checkpoint = checkpoint
        self.device = device or _detect_device()

        self.metadata = metadata or {}
        self.metadata.update(
            host=host,
            port=port,
            device=self.device,
            experiment_name=str(checkpoints_dir).rstrip('/').split('/')[-1] or '',
        )

        self.policy_manager = _PolicyManager(self._load_policy)

    def _load_policy(self, checkpoint_id: str) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')

        base_meta = {'checkpoint_id': checkpoint_id, 'checkpoint_path': checkpoint_path, **self.metadata}
        policy = self.policy_factory(checkpoint_path)
        if hasattr(policy, 'metadata') and policy.metadata:
            base_meta.update(policy.metadata)

        return LerobotPolicy(policy, self.device, extra_meta=base_meta)

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        if checkpoint_id:
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                raise ValueError(f'Checkpoint not found: {checkpoint_id}. Available: {available}')
            return checkpoint_id

        if self.checkpoint:
            checkpoint_id = str(self.checkpoint).strip('/')
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                raise ValueError(f'Configured checkpoint not found: {checkpoint_id}. Available: {available}')
            logger.info(f'Using configured checkpoint: {checkpoint_id}')
            return checkpoint_id

        checkpoint_id = get_latest_checkpoint(self.checkpoints_dir)
        logger.info(f'Using latest checkpoint: {checkpoint_id}')
        return checkpoint_id

    async def resolve_model(self, checkpoint_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        resolved_id = self._resolve_checkpoint_id(checkpoint_id)
        policy = await self.policy_manager.get_policy(resolved_id, websocket)
        return policy, {'checkpoint_id': resolved_id}

    def create_policy(self, model_handle: Any) -> Policy:
        return model_handle

    async def get_models(self) -> dict:
        try:
            return {'models': list_checkpoints(self.checkpoints_dir)}
        except Exception:
            logger.exception('Failed to list checkpoints.')
            return {'models': []}

    async def release_policy(self, model_handle):
        await self.policy_manager.release_session()

    async def _startup(self):
        pass

    def serve(self):
        async def _run():
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
            await uvicorn.Server(config).serve()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')


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
    InferenceServer(
        policy_factory, codec, checkpoints_dir, checkpoint, host=host, port=port, recording_dir=recording_dir
    ).serve()


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(main)
