import asyncio
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import torch
import websockets
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from websockets.asyncio.server import serve

from positronic.cfg.policy import action as act_cfg
from positronic.cfg.policy import observation as obs_cfg
from positronic.offboard.serialisation import deserialise, serialise
from positronic.policy import DecodedEncodedPolicy, Policy
from positronic.policy.lerobot import LerobotPolicy
from positronic.utils import get_latest_checkpoint
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
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.device = device or _detect_device()

        self.metadata = metadata or {}
        self.metadata.update(host=host, port=port, device=self.device)

        checkpoint_path = self._checkpoint_path()
        self.policy = self._policy(checkpoint_path)

    def _policy(self, checkpoint_path: str) -> Policy:
        base_meta = {'checkpoint_path': checkpoint_path}
        # Preserve caller-provided metadata (host/port/device/etc.) in the policy meta.
        base_meta.update(self.metadata)
        policy = self.policy_factory(checkpoint_path)
        if hasattr(policy, 'metadata'):
            base_meta.update(policy.metadata)
        base = LerobotPolicy(policy, self.device)

        return DecodedEncodedPolicy(
            base, encoder=self.observation_encoder.encode, decoder=self.action_decoder.decode, extra_meta=base_meta
        )

    def _checkpoint_path(self) -> str:
        checkpoints_dir = str(self.checkpoints_dir).rstrip('/') + '/checkpoints/'
        if self.checkpoint is None:
            checkpoint = get_latest_checkpoint(checkpoints_dir)
        else:
            checkpoint = str(self.checkpoint).strip('/')

        return checkpoints_dir.rstrip('/') + '/' + checkpoint + '/pretrained_model/'

    async def _handler(self, websocket):
        peer = websocket.remote_address
        logger.info(f'Connected to {peer}')

        try:
            self.policy.reset()

            # Send Metadata
            await websocket.send(serialise({'meta': self.policy.meta}))

            # Inference Loop
            async for message in websocket:
                try:
                    obs = deserialise(message)
                    action = self.policy.select_action(obs)
                    await websocket.send(serialise({'result': action}))

                except Exception as e:
                    logger.error(f'Error processing message from {peer}: {e}')
                    logger.debug(traceback.format_exc())
                    error_response = {'error': str(e)}
                    await websocket.send(serialise(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f'Connection closed for {peer}')
        except Exception as e:
            logger.error(f'Unexpected error for {peer}: {e}')
            logger.debug(traceback.format_exc())

    async def serve(self):
        async with serve(self._handler, self.host, self.port):
            logger.info(f'Server started on ws://{self.host}:{self.port}')
            await asyncio.get_running_loop().create_future()  # Run forever


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
