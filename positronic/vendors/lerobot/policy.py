from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from positronic.policy import Policy, Session
from positronic.policy import keys as policy_keys
from positronic.policy.observation import TASK_FIELD


def _detect_device() -> str:
    """Select the best available torch device.

    Duplicated across lerobot vendors because torch is not a base dependency.
    """
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


class _LerobotSession(Session):
    def __init__(self, policy, preprocessor, postprocessor, device: str, meta: dict[str, Any]):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._meta = meta

    def __call__(self, obs: dict[str, Any], time_ns: int) -> list[dict[str, Any]]:
        obs_int = {}
        for key, val in obs.items():
            if key == TASK_FIELD:
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

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta


def warm_observation(config: PreTrainedConfig) -> dict[str, Any]:
    """Zero-filled inputs matching the features ``config`` declares.

    Taken from the policy that was built rather than read from the checkpoint a second time. Visual features
    are declared channels-first and arrive here channels-last, the way a session takes them.
    """
    if not config.input_features:
        raise ValueError('The policy declares no input features, so there is nothing to warm it with')
    obs: dict[str, Any] = {TASK_FIELD: ''}
    for name, feature in config.input_features.items():
        if feature.type is FeatureType.VISUAL:
            channels, height, width = feature.shape
            obs[name] = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            obs[name] = np.zeros(feature.shape, dtype=np.float32)
    return obs


class LerobotPolicy(Policy):
    def __init__(self, checkpoint_path: str, device: str | None = None, extra_meta: dict[str, Any] | None = None):
        self._device = device or _detect_device()
        config = PreTrainedConfig.from_pretrained(checkpoint_path)
        policy_cls = get_policy_class(config.type)
        self._policy = policy_cls.from_pretrained(checkpoint_path).to(self._device)
        self._preprocessor, self._postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint_path)
        self._meta = {**(extra_meta or {}), policy_keys.TYPE: config.type}

    @property
    def config(self) -> PreTrainedConfig:
        """The checkpoint's own declaration of what this policy takes."""
        return self._policy.config

    def new_session(self, context=None, rt=None):
        self._policy.reset()
        return _LerobotSession(self._policy, self._preprocessor, self._postprocessor, self._device, self._meta)

    def close(self):
        if self._policy is not None:
            del self._policy
            self._policy = None
            if self._device.startswith('cuda'):
                torch.cuda.empty_cache()
