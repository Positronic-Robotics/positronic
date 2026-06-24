from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from positronic.policy import Policy, Session


class _MolmoAct2Session(Session):
    def __init__(self, model, processor, norm_tag: str, num_steps: int, meta: dict[str, Any]):
        self._model = model
        self._processor = processor
        self._norm_tag = norm_tag
        self._num_steps = num_steps
        self._meta = meta

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        out = self._model.predict_action(
            processor=self._processor,
            images=obs['images'],
            task=obs.get('task', ''),
            state=np.asarray(obs['state'], dtype=np.float32),
            norm_tag=self._norm_tag,
            inference_action_mode='continuous',
            enable_depth_reasoning=False,
            num_steps=self._num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )
        actions = out.actions[0].float().cpu().numpy()
        return [{'action': action} for action in actions]

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta


class MolmoAct2Policy(Policy):
    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = 'auto',
        norm_tag: str = 'franka_droid',
        num_steps: int = 10,
        extra_meta: dict[str, Any] | None = None,
    ):
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self._norm_tag = norm_tag
        self._num_steps = num_steps
        self._meta = {**(extra_meta or {}), 'type': 'molmoact2', 'norm_tag': norm_tag}

    def new_session(self, context=None) -> Session:
        return _MolmoAct2Session(self._model, self._processor, self._norm_tag, self._num_steps, self._meta)

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta.copy()

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
