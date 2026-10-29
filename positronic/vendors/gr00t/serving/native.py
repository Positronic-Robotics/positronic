"""GR00T inference with checkpoint-native normalization of RTC action inputs."""

from copy import copy
from typing import Any

import numpy as np
import torch
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from . import ACTION, STATE, keys


class Policy(Gr00tPolicy):
    """Run the native processor and model with optional physical action targets."""

    def _encode_action_prefix(self, actions: dict[str, np.ndarray], state: dict[str, np.ndarray]) -> torch.Tensor:
        # Planned targets must survive normalization even outside the checkpoint's training range.
        processor = copy(self.processor.state_action_processor)
        processor.clip_outliers = False
        normalized = processor.apply_action(actions, self.embodiment_tag.value, state)
        names = self.processor.modality_configs[self.embodiment_tag.value][ACTION].modality_keys
        values = np.concatenate([normalized[name] for name in names], axis=-1)
        padded = torch.zeros((self.processor.max_action_horizon, self.processor.max_action_dim))
        padded[: values.shape[0], : values.shape[1]] = torch.from_numpy(values)
        return padded

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processed = []
        for obs in self._unbatch_observation(observation):
            step = self._to_vla_step_data(obs)
            inputs = self.processor([{'type': MessageType.EPISODE_STEP.value, 'content': step}])
            if options is not None:
                inputs[ACTION] = self._encode_action_prefix(options[ACTION], obs[STATE])
            processed.append(inputs)
        inputs = _rec_to_dtype(self.collate_fn(processed), dtype=torch.bfloat16)
        with torch.inference_mode():
            prediction = self.model.get_action(**inputs, options=options)['action_pred'].float()
        decoded = self.processor.decode_action(prediction.cpu().numpy(), self.embodiment_tag, observation[STATE])
        if options is not None:
            # Frozen physical targets are exact; native decoding clips and model inputs use bfloat16.
            frozen = options[keys.RTC.FROZEN_STEPS]
            for name, values in options[ACTION].items():
                decoded[name][:, :frozen] = values[:frozen]
        return {name: values.astype(np.float32) for name, values in decoded.items()}, {}
