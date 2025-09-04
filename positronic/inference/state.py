from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class ImageEncodingConfig:
    key: str = "image"
    output_key: Optional[str] = None  # If not None, the encoding will have different key
    resize: Optional[List[int]] = None
    offset: Optional[int] = None


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
        self.frame_queues = {}
        for cfg in images:
            self.frame_queues[cfg.key] = deque(maxlen=cfg.offset)

    def encode_episode(self, episode_data):
        """Encodes data for training (i.e. to_lerobot.py). Every episode is a dict of tensors."""
        obs = {}
        for cfg in self.images:
            image = episode_data[cfg.key]
            image = image.permute(0, 3, 2, 1)  # BHWC -> BCWH
            if cfg.resize is not None:
                image = F.interpolate(image, size=tuple(cfg.resize), mode='nearest')
            if cfg.offset is not None:
                image = torch.cat([torch.zeros_like(image[:cfg.offset]), image[:-cfg.offset]], dim=0)

            output_key = cfg.output_key if cfg.output_key is not None else cfg.key
            obs[output_key] = image.permute(0, 3, 2, 1)  # BCWH -> BHWC

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
            if cfg.offset is not None:
                image = self._get_from_frame_queue(cfg)
            else:
                image = torch.tensor(images[cfg.key]).permute(2, 1, 0).unsqueeze(0)
                if cfg.resize is not None:
                    image = F.interpolate(image, size=tuple(cfg.resize), mode='nearest')
                image = image.float() / 255
                self.frame_queues[cfg.key].append(image)
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

    def _get_from_frame_queue(self, cfg: ImageEncodingConfig):
        if cfg.key in self.frame_queues and len(self.frame_queues[cfg.key]) == cfg.offset:
            return self.frame_queues[cfg.key][0]
        else:
            return torch.zeros(1, 3, *cfg.resize, dtype=torch.float32)

    def get_features(self):
        features = {}
        for cfg in self.images:
            features[cfg.output_key] = {
                "dtype": "video",
                "shape": (*cfg.resize[::-1], 3),
                "names": ["height", "width", "channel"],
            }

        features[self.state_output_key] = {
            "dtype": "float64",
            "shape": (len(self.state),),
            "names": self.state,
        }
        return features
