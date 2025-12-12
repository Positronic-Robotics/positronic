import asyncio
import logging
import traceback
from typing import Any

import configuronic as cfn
import numpy as np
import ormsgpack
import pos3
import torch
import websockets
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from websockets.asyncio.server import serve

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
        port: int,
        policy: PreTrainedPolicy,
        n_action_chunk: int | None = None,
        metadata: dict[str, Any] | None = None,
        host: str = '0.0.0.0',
        device: str | None = None,
    ):
        self.policy = policy
        self.n_action_chunk = n_action_chunk
        self.metadata = metadata or {}
        self.host = host
        self.port = port
        self.device = device or _detect_device()

        self.metadata.update({'host': host, 'port': port, 'device': device})

    async def _handler(self, websocket):
        peer = websocket.remote_address
        logger.info(f'Connected to {peer}')

        try:
            self.policy.reset()

            # Send Metadata
            await websocket.send(ormsgpack.packb({'meta': self.metadata}, option=ormsgpack.OPT_SERIALIZE_NUMPY))

            # Inference Loop
            async for message in websocket:
                try:
                    obs = ormsgpack.unpackb(message)
                    for key, val in obs.items():
                        if isinstance(val, np.ndarray):
                            if key.startswith('observation.images.'):
                                val = np.transpose(val.astype(np.float32) / 255.0, (2, 0, 1))
                            val = val[np.newaxis, ...]
                            obs[key] = torch.from_numpy(val).to(self.device)
                        else:
                            obs[key] = torch.as_tensor(val).to(self.device)

                    action = self.policy.predict_action_chunk(obs)[:, : self.n_action_chunk]
                    action = action.squeeze(0).cpu().numpy()
                    await websocket.send(ormsgpack.packb({'result': action}, option=ormsgpack.OPT_SERIALIZE_NUMPY))

                except Exception as e:
                    logger.error(f'Error processing message from {peer}: {e}')
                    logger.debug(traceback.format_exc())
                    error_response = {'error': str(e)}
                    await websocket.send(ormsgpack.packb(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f'Connection closed for {peer}')
        except Exception as e:
            logger.error(f'Unexpected error for {peer}: {e}')
            logger.debug(traceback.format_exc())

    async def serve(self):
        async with serve(self._handler, self.host, self.port):
            logger.info(f'Server started on ws://{self.host}:{self.port}')
            await asyncio.get_running_loop().create_future()  # Run forever


@cfn.config()
def act(checkpoint_path: str):
    path = pos3.download(checkpoint_path)
    policy = ACTPolicy.from_pretrained(path, strict=True)
    policy.metadata = {'type': 'act', 'checkpoint_path': checkpoint_path}
    return policy


@cfn.config(policy=act, port=8000, host='0.0.0.0', n_action_chunk=None)
def main(policy: PreTrainedPolicy, n_action_chunk: int | None, port: int, host: str):
    """
    Starts the inference server with the given policy.
    """
    server = InferenceServer(port, policy, n_action_chunk, policy.metadata, host)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info('Server stopped by user')


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(main)
