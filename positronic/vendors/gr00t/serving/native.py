"""GR00T inference with checkpoint-native normalization of RTC action inputs."""

from typing import Any

import numpy as np
import torch
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from . import ACTION, STATE


class Policy(Gr00tPolicy):
    """Run the native processor and model with optional physical action targets."""

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processed = []
        for obs in self._unbatch_observation(observation):
            step = self._to_vla_step_data(obs)
            if options is not None:
                step.actions = options[ACTION]
            processed.append(self.processor([{'type': MessageType.EPISODE_STEP.value, 'content': step}]))
        inputs = _rec_to_dtype(self.collate_fn(processed), dtype=torch.bfloat16)
        with torch.inference_mode():
            prediction = self.model.get_action(**inputs, options=options)['action_pred'].float()
        decoded = self.processor.decode_action(prediction.cpu().numpy(), self.embodiment_tag, observation[STATE])
        return {name: values.astype(np.float32) for name, values in decoded.items()}, {}
