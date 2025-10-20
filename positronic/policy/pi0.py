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


def basic_pi0_request(observation: dict[str, Any]) -> dict[str, Any]:
    try:
        res = {
            'observation/image': observation['observation.images.side'],
            'observation/wrist_image': observation['observation.images.image'],
            'observation/state': observation['observation.state'],
        }
        if 'task' in observation:
            res['prompt'] = observation['task']
    except KeyError as e:
        raise KeyError(f'Missing key, available keys: {observation.keys()}') from e
    return res


def droid_request(observation: dict[str, Any]) -> dict[str, Any]:
    res = {
        'observation/joint_position': observation['observation.state'][:7],
        'observation/gripper_position': observation['observation.state'][7:8],
        'observation/exterior_image_1_left': observation['observation.images.exterior'],
        'observation/wrist_image_left': observation['observation.images.wrist'],
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
        self.start_time = None
        self.last_log_time = None

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.start_time is None:
            self.start_time = time.monotonic()

        if len(self.action_queue) == 0:
            request = self.obs_tf(observation)
            rr.set_time_seconds('offset_time', time.monotonic() - self.start_time)
            rr.log('observation/exterior_image_1_left', rr.Image(request['observation/exterior_image_1_left']))
            rr.log('observation/wrist_image_left', rr.Image(request['observation/wrist_image_left']))
            jp = request['observation/joint_position']
            for i in range(jp.shape[0]):
                rr.log(f'observation/joint_position/{i}', rr.Scalar(jp[i]))
            rr.log('observation/gripper_position', rr.Scalar(request['observation/gripper_position']))

            action_chunk = self.client.infer(request)['actions']
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
