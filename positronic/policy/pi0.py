from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import Policy

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError as e:
    OPENPI_CLIENT_URL = 'https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client'
    raise ImportError(f'openpi_client not installed, install from {OPENPI_CLIENT_URL}') from e


def basic_pi0_request(observation: dict[str, Any]) -> dict[str, Any]:
    res = {
        'observation/image': observation['observation.images.side'][0],
        'observation/wrist_image': observation['observation.images.image'][0],
        'observation/state': observation['observation.state'][0],
    }
    if 'task' in observation:
        res['prompt'] = observation['task']
    return res


def droid_request(observation: dict[str, Any]) -> dict[str, Any]:
    res = {
        'observation/joint_position': observation['observation.state'][0][:7],
        'observation/gripper_position': observation['observation.state'][0][7:8],
        'observation/exterior_image_1_left': observation['observation.images.exterior'][0],
        'observation/wrist_image_left': observation['observation.images.wrist'][0],
    }
    if 'task' in observation:
        res['prompt'] = observation['task']
    return res


class OpenPIRemotePolicy(Policy):
    def __init__(self, host: str, port: int, n_action_steps: int | None = None, obs_tf=basic_pi0_request):
        self.client = WebsocketClientPolicy(host, port)
        self.action_queue = deque()
        self.n_action_steps = n_action_steps
        self.obs_tf = obs_tf

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        if len(self.action_queue) == 0:
            request = self.obs_tf(observation)
            action_chunk = self.client.infer(request)['actions']
            if self.n_action_steps is not None:
                action_chunk = action_chunk[: self.n_action_steps]
            self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()
        return np.array(action)

    def reset(self):
        self.action_queue.clear()
        self.client.reset()
