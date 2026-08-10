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

from positronic import keys
from positronic.cfg import codecs
from positronic.policy import Codec, Policy, Session
from positronic.policy.observation import TASK_FIELD
from positronic.policy.spec import PolicySource, inline
from positronic.policy.wrappers import ChunkedSchedule
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


class _LerobotSession(Session):
    def __init__(self, policy, device: str, meta: dict[str, Any]):
        self._policy = policy
        self._device = device
        self._meta = meta

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        obs_int = {}
        for key, val in obs.items():
            if key == TASK_FIELD:
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

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta


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


class LerobotPolicy(Policy):
    def __init__(self, policy: PreTrainedPolicy, device: str | None = None, extra_meta: dict[str, Any] | None = None):
        self._device = device or _detect_device()
        self._policy = policy.to(self._device)
        self._meta = extra_meta or {}

    def new_session(self, context=None, *, now=None, gate=None):
        self._policy.reset()
        return _LerobotSession(self._policy, self._device, self._meta)

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta.copy()

    def close(self):
        if self._policy is not None:
            del self._policy
            self._policy = None
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

    return LerobotPolicy(policy, device, extra_meta={keys.TYPE: 'act', keys.CHECKPOINT_PATH: checkpoint_dir})


@cfn.config(
    base=act, codec=codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action, horizon=1.0)
)
def act_absolute(base: Policy, codec: Codec):
    """ACT with the absolute-position codec, composed in-process."""
    return inline(ChunkedSchedule() | codec | PolicySource(base))
