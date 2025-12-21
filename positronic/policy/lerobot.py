from collections.abc import Callable
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
    def __init__(
        self,
        policy_factory: Callable[[], PreTrainedPolicy],
        device: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ):
        self.factory = policy_factory
        self.original = None
        self.target_device = device
        self.n_action_chunk = None

        # We initialize on CPU to ensure the policy is pickleable when passed to a subprocess.
        # The model will be moved to the target device (e.g. MPS/CUDA) lazily on the first inference call.
        self.device = 'cpu'
        self.extra_meta = extra_meta or {}

    @property
    def _policy(self) -> PreTrainedPolicy:
        if self.original is None:
            self.original = self.factory()
            self.target_device = self.target_device or _detect_device()
            self.original.to(self.target_device)
        return self.original

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        policy = self._policy

        obs_int = {}
        for key, val in obs.items():
            if key == 'task':
                obs_int[key] = val
            elif isinstance(val, np.ndarray):
                if key.startswith('observation.images.'):
                    val = np.transpose(val.astype(np.float32) / 255.0, (2, 0, 1))
                val = val[np.newaxis, ...]
                obs_int[key] = torch.from_numpy(val).to(self.target_device)
            else:
                obs_int[key] = torch.as_tensor(val).to(self.target_device)

        action = policy.predict_action_chunk(obs_int)[:, : self.n_action_chunk]
        action = action.squeeze(0).cpu().numpy()
        return [{'action': a} for a in action]

    def reset(self):
        self._policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        return self.extra_meta.copy()
