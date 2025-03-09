from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class ImageEncodingConfig:
    key: str = "image"
    output_key: Optional[str] = None  # If not None, the encoding will have different key
    resize: Optional[List[int]] = None


@dataclass
class StateEncodingConfig:
    state_output_key: str
    images: List[ImageEncodingConfig] = field(default_factory=list)
    state: List[str] = field(default_factory=list)


class StateEncoder:
    def __init__(self, state_output_key: str, images: List[ImageEncodingConfig], state: List[str]):
        self.state_output_key = state_output_key
        self.images = images
        self.state = state

    def encode_episode(self, episode_data):
        """Encodes data for training (i.e. to_lerobot.py). Every episode is a dict of tensors."""
        obs = {}
        for cfg in self.images:
            image = episode_data[cfg.key]
            image = image.permute(0, 3, 2, 1)  # BHWC -> BCWH
            if cfg.resize is not None:
                image = F.interpolate(image, size=tuple(cfg.resize), mode='nearest')

            output_key = cfg.output_key if cfg.output_key is not None else cfg.key
            obs[output_key] = image.permute(0, 1, 3, 2)  # BCWH -> BCHW

        obs[self.state_output_key] = torch.cat(
            [episode_data[k].unsqueeze(1) if episode_data[k].dim() == 1 else episode_data[k]
             for k in self.state],
            dim=1
        )
        return obs

    def encode(self, images, inputs):
        """Encodes data for inference."""
        obs = {}
        for cfg in self.images:
            image = torch.tensor(images[cfg.key], dtype=torch.float32).permute(2, 1, 0).unsqueeze(0) / 255
            if cfg.resize is not None:
                image = F.interpolate(image, size=tuple(cfg.resize), mode='nearest')
            output_key = cfg.output_key if cfg.output_key is not None else cfg.key
            obs[output_key] = image.permute(0, 1, 3, 2)

        data = {}
        for key in self.state:
            tensor = torch.tensor(inputs[key], dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            data[key] = tensor

        obs[self.state_output_key] = torch.cat(
            [data[k] for k in self.state], dim=0
        ).unsqueeze(0).type(torch.float32)
        return obs
