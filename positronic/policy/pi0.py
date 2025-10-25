import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import rerun as rr

from . import Policy

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError as e:
    OPENPI_CLIENT_URL = 'https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client'
    raise ImportError(f'openpi_client not installed, install from {OPENPI_CLIENT_URL}') from e


class OpenPIRemotePolicy(Policy):
    def __init__(self, host: str, port: int, n_action_steps: int | None = None):
        self.client = WebsocketClientPolicy(host, port)
        self.action_queue = deque()
        self.n_action_steps = n_action_steps
        self.start_time = None
        self.last_log_time = None

    def select_action(self, obs: Mapping[str, Any]) -> np.ndarray:
        if self.start_time is None:
            self.start_time = time.monotonic()

        if len(self.action_queue) == 0:
            rr.set_time_seconds('offset_time', time.monotonic() - self.start_time)
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    rr.log(key, rr.Image(value))
                elif isinstance(value, np.ndarray) and value.ndim == 1:
                    for i in range(value.shape[0]):
                        rr.log(f'{key}/{i}', rr.Scalar(value[i]))
                elif isinstance(value, float | int):
                    rr.log(key, rr.Scalar(value))

            if 'task' in obs:
                obs['prompt'] = obs['task']
                del obs['task']

            action_chunk = self.client.infer(obs)['actions']
            if self.n_action_steps is not None:
                action_chunk = action_chunk[: self.n_action_steps]

            self.action_queue.extend(action_chunk)

        action = self.action_queue.popleft()
        rr.set_time_seconds('offset_time', time.monotonic() - self.start_time)
        for i, action_value in enumerate(action):
            rr.log(f'raw_action/{i}', rr.Scalar(action_value))

        if self.last_log_time is not None:
            rr.log('delay', rr.Scalar(time.monotonic() - self.last_log_time))
        self.last_log_time = time.monotonic()

        return np.array(action)

    def reset(self):
        self.action_queue.clear()
        self.client.reset()
