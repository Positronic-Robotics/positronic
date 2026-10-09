from pathlib import Path
from typing import Any

from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic.dataset.transforms.episode import EpisodeTransform, Identity

from .base import Codec

GR00T_MODALITY_PATH = Path('meta/modality.json')
GR00T_MODALITY = 'gr00t_modality'
LEROBOT_FEATURES = 'lerobot_features'
ACTION = 'action'


def lerobot_vector(dim: int, names: list[str] | None = None) -> dict[str, Any]:
    """LeRobot feature descriptor for a float32 vector."""
    f: dict[str, Any] = {'shape': (dim,), 'dtype': 'float32'}
    if names:
        f['names'] = names
    return f


def lerobot_image(width: int, height: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an RGB image."""
    return {'shape': (height, width, 3), 'names': ['height', 'width', 'channel'], 'dtype': 'video'}


def lerobot_action(dim: int) -> dict[str, Any]:
    """LeRobot feature descriptor for an action vector."""
    return lerobot_vector(dim, ['actions'])


class Metadata(Codec):
    """Attach metadata to a codec and its training transform without changing the data."""

    WIRE_NAME = 'metadata'

    def __init__(self, values: dict[str, Any]):
        self._values = dict(values)

    def encode(self, data):
        return data

    def decode(self, data):
        return data

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._values)

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity(meta=self.meta)

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'values': self.meta}}
