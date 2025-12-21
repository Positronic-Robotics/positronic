from typing import Any

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from positronic.utils import flatten_dict

from .base import Policy


class OpenPIRemotePolicy(Policy):
    def __init__(self, host: str, port: int, n_action_steps: int | None = None):
        self.client = WebsocketClientPolicy(host, port)
        self.n_action_steps = n_action_steps

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        obs_internal = obs.copy()
        if 'task' in obs_internal:
            obs_internal['prompt'] = obs_internal.pop('task')

        action_chunk = self.client.infer(obs_internal)['actions']
        if self.n_action_steps is not None:
            action_chunk = action_chunk[: self.n_action_steps]

        return [{'action': np.array(action)} for action in action_chunk]

    def reset(self):
        self.client.reset()

    @property
    def meta(self) -> dict[str, Any]:
        result = {'type': 'openpi', 'server': self.client.get_server_metadata()}
        if self.n_action_steps is not None:
            result['n_action_steps'] = self.n_action_steps
        return flatten_dict(result)
