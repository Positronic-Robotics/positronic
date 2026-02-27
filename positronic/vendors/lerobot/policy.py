from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.policy import Policy


def _detect_device() -> str:
    """Select the best available torch device."""
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


def make_processors(config, pretrained_path):
    """Load preprocessor/postprocessor pipelines from a pretrained checkpoint."""
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(config, pretrained_path=pretrained_path)


class LerobotPolicy(Policy):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor=None,
        postprocessor=None,
        device: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ):
        self._device = device or _detect_device()
        self._policy = policy.to(self._device)
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.extra_meta = extra_meta or {}

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        obs_int = {}
        for key, val in obs.items():
            if key == 'task':
                obs_int[key] = val
            elif isinstance(val, np.ndarray):
                if key.startswith('observation.images.'):
                    val = torch.from_numpy(np.transpose(val, (2, 0, 1)).copy()).float() / 255.0
                else:
                    val = torch.from_numpy(val).float()
                obs_int[key] = val
            else:
                obs_int[key] = torch.as_tensor(val)

        if self._preprocessor is not None:
            obs_int = self._preprocessor(obs_int)

        action = self._policy.select_action(obs_int)

        if self._postprocessor is not None:
            action = self._postprocessor(action)

        action = action.cpu().numpy().squeeze(0)
        if action.ndim == 1:
            return [{'action': action}]
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
            if self._device.startswith('cuda'):
                torch.cuda.empty_cache()
