from collections import deque
from collections.abc import Mapping

import numpy as np
import torch

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError as e:
    OPENPI_CLIENT_URL = 'https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client'
    raise ImportError(f'openpi_client not installed, install from {OPENPI_CLIENT_URL}') from e


def _prepare_observations(observation: Mapping[str, torch.Tensor]) -> Mapping[str, np.ndarray]:
    openpi_observation = {
        'observation/image': observation['observation.images.side'][0].cpu().numpy(),
        'observation/wrist_image': observation['observation.images.image'][0].cpu().numpy(),
        'observation/state': observation['observation.state'].cpu().numpy()[0],
    }

    if 'task' in observation:
        openpi_observation['prompt'] = observation['task']

    return openpi_observation


class PI0RemotePolicy:
    def __init__(self, host: str, port: int, n_action_steps: int | None = None):
        self.client = WebsocketClientPolicy(host, port)
        self.action_queue = deque()
        self.n_action_steps = n_action_steps

    def select_action(self, observation: Mapping[str, torch.Tensor]) -> np.ndarray:
        observation = _prepare_observations(observation)

        if len(self.action_queue) == 0:
            action_chunk = self.client.infer(observation)['actions']
            if self.n_action_steps is not None:
                action_chunk = action_chunk[: self.n_action_steps]
            self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()

        return torch.tensor(action)

    def reset(self):
        self.action_queue.clear()
        self.client.reset()

    def to(self, device: str):
        return self
