from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .base import Policy


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


class LerobotPolicy(Policy):
    def __init__(self, policy: PreTrainedPolicy, device: str | None = None, extra_meta: dict[str, Any] | None = None):
        self._device = device or _detect_device()
        self._policy = policy.to(self._device)
        self.extra_meta = extra_meta or {}

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        obs_int = {}
        for key, val in obs.items():
            if key == 'task':
                obs_int[key] = val
            elif isinstance(val, np.ndarray):
                if key.startswith('observation.images.'):
                    val = np.transpose(val.astype(np.float32) / 255.0, (2, 0, 1))
                val = val[np.newaxis, ...]
                obs_int[key] = torch.from_numpy(val).to(self._device)
            else:
                obs_int[key] = torch.as_tensor(val).to(self._device)

        action = self._policy.predict_action_chunk(obs_int)
        action = action.squeeze(0).cpu().numpy()
        return [{'action': a} for a in action]

    def reset(self):
        self._policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        return self.extra_meta.copy()

    def close(self):
        if self._policy is not None:
            self._policy.to('cpu')
            del self._policy
            self._policy = None
