"""Single-GPU DreamZero inference without the Roboarena wrapper's recording buffers."""

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

from .keys import ACTION_GRIP, ACTION_JOINTS


class Policy:
    def __init__(self, model_source: str, revision: str | None):
        os.environ['ENABLE_DIT_CACHE'] = 'true'
        os.environ['ATTENTION_BACKEND'] = 'TE'
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        from groot.vla.data.schema import EmbodimentTag  # noqa: PLC0415
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: PLC0415

        path = Path(model_source)
        if not path.is_dir():
            path = Path(snapshot_download(model_source, revision=revision))
        torch.cuda.set_device(0)
        self._rendezvous = tempfile.TemporaryDirectory()
        dist.init_process_group('nccl', init_method=f'file://{self._rendezvous.name}/init', rank=0, world_size=1)
        mesh = init_device_mesh('cuda', (1,), mesh_dim_names=('ip',))
        self._policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag('oxe_droid'), model_path=str(path), device='cuda', device_mesh=mesh
        )
        limit = self._policy.trained_model.action_head.model.local_attn_size
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('DreamZero must declare a finite positive native attention window')

    @torch.inference_mode()
    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        result, _ = self._policy.lazy_joint_forward_causal(Batch(obs=observation))
        actions = {}
        for name, size in ((ACTION_JOINTS, 7), (ACTION_GRIP, 1)):
            value = result.act[name]
            if isinstance(value, torch.Tensor):
                value = value.float().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if name == ACTION_GRIP and value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2 or value.shape[1] != size:
                raise ValueError(f'Native {name} must have shape (time, {size}), got {value.shape}')
            actions[name] = value
        if not len(actions[ACTION_JOINTS]) or len(actions[ACTION_JOINTS]) != len(actions[ACTION_GRIP]):
            raise ValueError('Native joint and gripper predictions must have the same nonempty horizon')
        if any(not np.isfinite(values).all() for values in actions.values()):
            raise ValueError('Native predictions must be finite')
        return actions

    def reset(self) -> None:
        head = self._policy.trained_model.action_head
        head.current_start_frame = 0
        head.language = None
        head.kv_cache1 = head.kv_cache_neg = None
        head.crossattn_cache = head.crossattn_cache_neg = None
        head.clip_feas = head.ys = None

    def close(self) -> None:
        self.reset()
        del self._policy
        dist.destroy_process_group()
        self._rendezvous.cleanup()
        torch.cuda.empty_cache()
