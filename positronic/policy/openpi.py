import time
from collections import deque
from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from positronic.utils import flatten_dict

from .base import Policy


class OpenPIRemotePolicy(Policy):
    def __init__(self, host: str, port: int, n_action_steps: int | None = None):
        self.client = WebsocketClientPolicy(host, port)
        self.action_queue = deque()
        self.n_action_steps = n_action_steps
        self.start_time = None
        self.last_log_time = None

    def select_action(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self.start_time is None:
            self.start_time = time.monotonic()

        obs = inputs.copy()
        if len(self.action_queue) == 0:
            if 'task' in obs:
                obs['prompt'] = obs['task']
                del obs['task']

            action_chunk = self.client.infer(obs)['actions']
            if self.n_action_steps is not None:
                action_chunk = action_chunk[: self.n_action_steps]

            self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()
        self.last_log_time = time.monotonic()

        return {'action': np.array(action)}

    def reset(self):
        self.action_queue.clear()
        self.client.reset()

    @property
    def meta(self) -> dict[str, Any]:
        result = {'type': 'openpi', 'server': self.client.get_server_metadata()}
        if self.n_action_steps is not None:
            result['n_action_steps'] = self.n_action_steps
        return flatten_dict(result)
