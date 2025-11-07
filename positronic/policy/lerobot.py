from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from . import Policy


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
    def __init__(self, original: PreTrainedPolicy, device: str | None = None, extra_meta: dict[str, Any] | None = None):
        self.original = original
        self.device = device or _detect_device()
        self.original.to(self.device)
        self.extra_meta = extra_meta or {}

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        obs = {}
        for key, val in observation.items():
            if key == 'task':
                obs[key] = val
            elif isinstance(val, np.ndarray):
                if key.startswith('observation.images.'):
                    val = np.transpose(val.astype(np.float32) / 255.0, (2, 0, 1))
                val = val[np.newaxis, ...]
                obs[key] = torch.from_numpy(val).to(self.device)
            else:
                obs[key] = torch.as_tensor(val).to(self.device)

        return self.original.select_action(obs).squeeze(0).cpu().numpy()

    def reset(self):
        self.original.reset()

    @property
    def meta(self) -> dict[str, Any]:
        return self.extra_meta.copy()
