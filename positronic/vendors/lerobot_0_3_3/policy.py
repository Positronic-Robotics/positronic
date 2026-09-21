from typing import Any

import configuronic as cfn
import numpy as np
import pos3
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.offboard.spec import Model
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import ACTION
from positronic.policy.observation import TASK_FIELD
from positronic.utils.checkpoints import resolve_checkpoint
from positronic.vendors.lerobot_0_3_3.backbone import register_all


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


def warm_observation(config: PreTrainedConfig) -> dict[str, Any]:
    """Zero-filled inputs matching the features ``config`` declares.

    Taken from the policy that was built rather than from the checkpoint directory, so a factory is free to
    load one however it likes. Visual features are declared channels-first and arrive here channels-last, the
    way a session takes them.
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


def _infer(policy: PreTrainedPolicy, device: str, obs: Obs) -> list[dict[str, Any]]:
    """One model call: an observation in, an action chunk out."""
    obs_int = {}
    for key, val in obs.items():
        if key == TASK_FIELD:
            obs_int[key] = val
        elif isinstance(val, np.ndarray):
            if key.startswith('observation.images.'):
                val = np.transpose(val.astype(np.float32) / 255.0, (2, 0, 1))
            val = val[np.newaxis, ...]
            obs_int[key] = torch.from_numpy(val).to(device)
        else:
            obs_int[key] = torch.as_tensor(val).to(device)

    action = policy.predict_action_chunk(obs_int)
    action = action.squeeze(0).cpu().numpy()
    return [{ACTION: a} for a in action]


class LerobotModel(Model):
    """A loaded LeRobot model accepting encoded observations and returning action chunks."""

    def __init__(self, policy: PreTrainedPolicy, device: str | None = None, extra_meta: dict[str, Any] | None = None):
        self._device = device or _detect_device()
        self._policy = policy.to(self._device)
        self._meta = extra_meta or {}

    def __call__(self, obs: Obs) -> list[dict[str, Any]]:
        return _infer(self._policy, self._device, obs)

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self) -> None:
        del self._policy
        if self._device.startswith('cuda'):
            torch.cuda.empty_cache()


@cfn.config(checkpoint=None)
def act(checkpoints_dir: str, checkpoint: str | None, n_action_steps: int | None = None, device: str | None = None):
    register_all()

    checkpoints_dir = checkpoints_dir.rstrip('/') + f'/{CHECKPOINTS_DIR}'
    checkpoint = resolve_checkpoint(checkpoints_dir, checkpoint, None)
    checkpoint_dir = f'{checkpoints_dir}/{checkpoint}/{PRETRAINED_MODEL_DIR}/'
    policy = ACTPolicy.from_pretrained(pos3.download(checkpoint_dir), strict=True)
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return LerobotModel(
        policy, device, extra_meta={policy_keys.TYPE: 'act', policy_keys.CHECKPOINT_PATH: checkpoint_dir}
    )


# TODO: Bind local model ownership to processor runs before adding an in-process ACT policy config.
