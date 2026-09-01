from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from positronic.policy import Policy, Session
from positronic.policy.base import TYPE
from positronic.vendors import molmoact2

# The three views and the 8-D ``[joint_positions(7), grip(1)]`` state of the DROID action space this vendor
# serves, at the 378x378 the model tiles every image to.
_NUM_VIEWS = 3
_IMAGE_SIZE = (378, 378)
_STATE_DIM = 8


def warm_observation() -> dict[str, Any]:
    """Zero-filled inputs one inference can run on, so the model's first-call cost is paid before it serves."""
    return {
        molmoact2.IMAGES: [np.zeros((*_IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(_NUM_VIEWS)],
        molmoact2.STATE: np.zeros(_STATE_DIM, dtype=np.float32),
        molmoact2.TASK: '',
    }


class _MolmoAct2Session(Session):
    def __init__(self, model, processor, norm_tag: str, num_steps: int, meta: dict[str, Any]):
        self._model = model
        self._processor = processor
        self._norm_tag = norm_tag
        self._num_steps = num_steps
        self._meta = meta

    def __call__(self, obs: dict[str, Any], time_ns: int) -> list[dict[str, Any]]:
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
        return [{'action': action} for action in actions]

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta


class MolmoAct2Policy(Policy):
    def __init__(self, model_id: str, *, device_map: str = 'auto', norm_tag: str = 'franka_droid', num_steps: int = 10):
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self._norm_tag = norm_tag
        self._num_steps = num_steps
        self._meta = {TYPE: 'molmoact2', 'norm_tag': norm_tag}

    def new_session(self, context=None, rt=None) -> Session:
        return _MolmoAct2Session(self._model, self._processor, self._norm_tag, self._num_steps, self._meta)

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
