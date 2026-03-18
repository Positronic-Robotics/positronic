import logging
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
from fastapi import WebSocket

from positronic.offboard.vendor_server import PolicyManager, VendorServer, resolve_checkpoint
from positronic.policy import Codec, Policy
from positronic.utils.checkpoints import list_checkpoints
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.policy import LerobotPolicy, _detect_device

logger = logging.getLogger(__name__)


class InferenceServer(VendorServer):
    """LeRobot 0.4.x inference server with singleton policy manager.

    Auto-detects policy type from checkpoint config and creates the appropriate
    preprocessor/postprocessor. Works with SmolVLA, ACT, Diffusion, or any
    lerobot 0.4.x policy.
    """

    def __init__(
        self,
        codec: Codec | None,
        checkpoints_dir: str | Path,
        checkpoint: str | None = None,
        host: str = '0.0.0.0',
        port: int = 8000,
        device: str | None = None,
        recording_dir: str | None = None,
    ):
        super().__init__(codec=codec, host=host, port=port, recording_dir=recording_dir)
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.checkpoint = checkpoint
        self.device = device or _detect_device()

        self.metadata = {
            'host': host,
            'port': port,
            'device': self.device,
            'experiment_name': str(checkpoints_dir).rstrip('/').split('/')[-1] or '',
        }

        self.policy_manager = PolicyManager(self._load_policy)

    def _load_policy(self, checkpoint_id: str) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        meta = {'checkpoint_id': checkpoint_id, 'checkpoint_path': checkpoint_path, **self.metadata}
        return LerobotPolicy(checkpoint_path, self.device, extra_meta=meta)

    async def resolve_model(self, model_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        resolved_id = resolve_checkpoint(self.checkpoints_dir, self.checkpoint, model_id)
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


@cfn.config(codec=lerobot_codecs.ee, checkpoint=None, port=8000, host='0.0.0.0', recording_dir=None)
def main(checkpoints_dir: str, checkpoint: str | None, codec, port: int, host: str, recording_dir: str | None):
    checkpoints_dir = str(pos3.download(checkpoints_dir))
    InferenceServer(codec, checkpoints_dir, checkpoint, host=host, port=port, recording_dir=recording_dir).serve()


phail = main.override(
    checkpoints_dir='s3://checkpoints/phail_unified/smolvla/170316_ee/',
    recording_dir='s3://inference/phail_unified/server_recordings/smolvla/170316_ee/',
)


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({'serve': main, 'phail': phail})
