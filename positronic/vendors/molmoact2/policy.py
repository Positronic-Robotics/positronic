from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from positronic.offboard import keys as offboard_keys
from positronic.offboard.spec import Model
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import ACTION
from positronic.vendors import molmoact2

# The three views of the DROID action space this vendor serves.
_NUM_VIEWS = 3


def warm_observation() -> dict[str, Any]:
    """Zero-filled inputs one inference can run on, so the model's first-call cost is paid before it serves."""
    return {
        molmoact2.IMAGES: [np.zeros((*molmoact2.IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(_NUM_VIEWS)],
        molmoact2.STATE: np.zeros(molmoact2.STATE_DIM, dtype=np.float32),
        molmoact2.TASK: '',
    }


class MolmoAct2Model(Model):
    def __init__(self, model_id: str, *, device_map: str = 'auto', norm_tag: str = 'franka_droid', num_steps: int = 10):
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self._norm_tag = norm_tag
        self._num_steps = num_steps
        self._meta = {
            policy_keys.TYPE: 'molmoact2',
            'norm_tag': norm_tag,
            'hf_repo': model_id,
            'model_id': model_id.split('/')[-1],
            offboard_keys.CHECKPOINT_ID: model_id.split('/')[-1],
        }

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        # predict_action is decorated @torch.no_grad() and manages its own precision: the model loads
        # in bfloat16 and runs bf16 throughout (its autocast path only guards fp32 inputs), so an
        # external torch.inference_mode() / torch.autocast wrap or a detach() would all be redundant.
        out = self._model.predict_action(
            processor=self._processor,
            images=obs[molmoact2.IMAGES],
            task=obs.get(molmoact2.TASK, ''),
            state=np.asarray(obs[molmoact2.STATE], dtype=np.float32),
            norm_tag=self._norm_tag,
            inference_action_mode='continuous',
            enable_depth_reasoning=False,
            num_steps=self._num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )
        actions = out.actions[0].float().cpu().numpy()
        return [{ACTION: action} for action in actions]

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
