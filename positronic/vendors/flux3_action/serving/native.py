"""Adapt DROID observations and physical action prefixes to FluxActionPolicy."""

from pathlib import Path

import numpy as np
import torch
from examples.droid_ee50 import ee50
from flux_action.policy import OBS_STATE, FluxActionPolicy
from flux_action.serving import robolab
from huggingface_hub import hf_hub_download

from . import ee, keys


def _ee50_stats(checkpoint: str, revision: str | None, subfolder: str | None) -> ee50.EE50Stats:
    """The EE-50 statistics a package trained with, from its directory or its Hugging Face repository."""
    local = Path(checkpoint, subfolder or '')
    if local.is_dir():
        return ee50.load_stats(local / keys.EE50_STATS_FILE)
    return ee50.load_stats(hf_hub_download(checkpoint, keys.EE50_STATS_FILE, revision=revision, subfolder=subfolder))


class Policy:
    def __init__(
        self,
        checkpoint: str,
        revision: str | None,
        subfolder: str | None,
        *,
        device: str = 'cuda',
        compile_dit: bool = True,
        offload_text_encoder: bool = False,
        rtc: bool = False,
    ):
        target_device = torch.device(device)
        policy = FluxActionPolicy.from_pretrained(checkpoint, revision=revision, subfolder=subfolder, device='cpu')
        config = policy.config
        self.ee = config.action_modality == keys.EE50_MODALITY
        if (
            config.camera_layout != 'droid'
            or config.inference_profile != 'default'
            or config.action_dim != (50 if self.ee else 8)
        ):
            raise ValueError(
                'FLUX serving requires a single-observation DROID checkpoint with eight joint action dimensions '
                'or fifty EE-50 dimensions'
            )
        if self.ee and rtc:
            raise ValueError('EE-50 checkpoints serve without RTC')
        self._ee50 = _ee50_stats(checkpoint, revision, subfolder) if self.ee else None
        if rtc and (
            config.rtc_max_prefix <= 0
            or config.quantization is not None
            or config.action_parameterization not in ('absolute', 'joint_relative')
        ):
            raise ValueError(
                'RTC requires a BF16 checkpoint trained with rtc_max_prefix and absolute or joint_relative actions'
            )
        # CUDA graph capture must be initialized outside inference mode before compiled DiT warm-up.
        self._first_graph = torch.cuda.CUDAGraph() if target_device.type == 'cuda' else None
        if self._first_graph is not None:
            with torch.cuda.graph(self._first_graph):
                torch.zeros(1, device=target_device)
        self._policy = robolab.prepare_serving_policy(
            policy,
            device=target_device,
            dtype='bfloat16',
            compile_dit=compile_dit,
            offload_text_encoder=offload_text_encoder,
            warmup=0,
        )

    @property
    def config(self):
        return self._policy.config

    def predict(self, observation: dict, committed_actions: np.ndarray | None, *, seed: int) -> np.ndarray:
        device = self._policy.device
        if self.ee:
            anchor = ee.pose_to_droid_cartesian(observation[keys.EE_POSE_FIELD])
            grip = float(np.asarray(observation[keys.GRIPPER_POSITION], dtype=np.float64).reshape(-1)[0])
            state = ee50.droid_state_token(grip, self._ee50)
        else:
            state = robolab.observation_state(observation, self.config.action_dim)
        batch = {OBS_STATE: torch.from_numpy(state)[None].to(device), keys.TASK: [observation[keys.PROMPT]]}
        for name, source in zip(self.config.camera_order, keys.CAMERAS, strict=True):
            image = np.asarray(observation[source])
            if image.dtype != np.uint8 or image.shape != (keys.VIEW_SIZE[1], keys.VIEW_SIZE[0], 3):
                raise ValueError(f'{source} must be a uint8 RGB image of shape (360, 640, 3)')
            batch[name] = torch.from_numpy(np.require(image, requirements=['C', 'W'])).permute(2, 0, 1)[None].to(device)
        committed = None if committed_actions is None else torch.from_numpy(committed_actions)[None].to(device)
        self.config.inference_seed = seed
        actions = self._policy.predict_action_chunk(batch, committed_actions=committed)
        result = actions[0].detach().float().cpu().numpy()
        if self.ee:
            # absolute flange targets [x, y, z, qw, qx, qy, qz, gripper], anchored at the observed pose
            result = ee.droid_commands_to_actions(ee50.decode_droid_chunk(result, anchor, self._ee50))
        result = result.astype(np.float32)
        if result.shape != (self.config.chunk_size, 8) or not np.isfinite(result).all():
            raise ValueError('FLUX returned an invalid action chunk')
        return result

    def close(self) -> None:
        del self._policy
