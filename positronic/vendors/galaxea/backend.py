"""Full-chunk G0.5 backend for internal, non-commercial evaluation only; see README.md and NOTICE.

Run as galaxea.backend with Galaxea's Python 3.10 environment; see server.py for module paths.
"""

import argparse
import logging
import threading
from pathlib import Path

import torch
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.websocket import packb, unpackb
from scripts.serve_policy import build_obs_dict, setup
from websockets.sync.server import serve

from . import protocol

logger = logging.getLogger(__name__)

_ABSENT_KEYS = '_absent_keys'
_COT_TEXT = '_cot_text'


class ChunkBackend:
    """Serialize access to the shared model; every request recomputes a complete trajectory."""

    def __init__(self, inferencer: PolicyInferencer, processor):
        self._inferencer = inferencer
        self._processor = processor
        self._lock = threading.Lock()

    def infer(self, obs: dict) -> dict:
        with self._lock:
            prediction = self._inferencer.infer([build_obs_dict(obs, self._processor)])[0]
        absent = prediction.pop(_ABSENT_KEYS, set())
        prediction.pop(_COT_TEXT, None)
        actions = {}
        for name, value in prediction.items():
            if name in absent:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError(f'Expected {name} as a (1, T, D) tensor')
            actions[name] = value[0].float().cpu().numpy()
        return protocol.chunk_response(actions)

    def handle(self, connection):
        connection.send(packb({protocol.PROTOCOL: protocol.FULL_CHUNK_V1}))
        for message in connection:
            try:
                response = self.infer(unpackb(message))
            except Exception as exc:
                logger.exception('G0.5 inference failed')
                response = {protocol.ERROR: str(exc)}
            connection.send(packb(response))


_EVAL_EMBODIMENT = 'eval_embodiment'
_DISCRETE_ACTION = 'model.model_arch.discrete_action'
_CONTINUOUS_ACTION = 'model.model_arch.continuous_action'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    checkpoint = args.checkpoint.absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    overrides = [
        f'{_EVAL_EMBODIMENT}={protocol.DROID_FRANKA}',
        f'{_DISCRETE_ACTION}=true',
        f'{_CONTINUOUS_ACTION}=false',
    ]
    cfg = load_config_from_run_dir(find_run_dir(str(checkpoint)), str(checkpoint), overrides)
    filter_embodiment(cfg, protocol.DROID_FRANKA)
    model, processor = setup(cfg, device=args.device)
    backend = ChunkBackend(PolicyInferencer(model, processor, device=args.device), processor)
    logger.info('G0.5 backend: internal, non-commercial evaluation only')
    with serve(backend.handle, args.host, args.port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
