import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import keys as obs_keys
from positronic.dataset.transforms import Elementwise
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Group, Identity

from .base import Codec


class BinarizeGripTraining(Codec):
    """Binarize grip signals in training data.

    Overrides the specified episode signals with thresholded values (> threshold → 1.0,
    else 0.0) so the model learns to predict binary grip. Compose to the left of
    obs/action codecs::

        BinarizeGripTraining(('grip', 'target_grip')) | BinarizeGripInference() | obs & action
    """

    WIRE_NAME = 'binarize_grip_training'

    def __init__(self, keys: tuple[str, ...], threshold: float = 0.5):
        self._keys = keys
        self._threshold = threshold

    def encode(self, data):
        return data

    def _decode_single(self, data: dict) -> dict:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        threshold = self._threshold

        def _binarize_signal(key):
            def _derive(episode):
                return Elementwise(
                    episode[key], lambda v: (np.asarray(v, dtype=np.float32) > threshold).astype(np.float32)
                )

            return _derive

        transforms = {k: _binarize_signal(k) for k in self._keys}
        return Group(Derive(meta=None, **transforms), Identity())

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'keys': list(self._keys), 'threshold': self._threshold},
        }


class BinarizeGripInference(Codec):
    """Threshold grip in decoded actions at inference time.

    Compose to the left of action codecs so it runs after action decoding::

        BinarizeGripInference() | obs & action
    """

    WIRE_NAME = 'binarize_grip_inference'

    def __init__(self, threshold: float = 0.5, key: str = obs_keys.TARGET_GRIP):
        self._threshold = threshold
        self._key = key

    def encode(self, data):
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        if self._key in data:
            data[self._key] = 1.0 if data[self._key] > self._threshold else 0.0
        return data

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'threshold': self._threshold, 'key': self._key},
        }


class FlipGrip(Codec):
    """Serve a checkpoint that speaks the inverted grip convention (1 = open) on the canonical
    1 = closed wire: flips ``grip`` entering the model and ``target_grip`` leaving it.

    HACK: exists only to keep checkpoints trained on inverted-grip sim data alive.
    TODO: Drop it (and its ``flip_grip`` compose hook) when no served checkpoint needs it.

    Compose to the left of obs/action codecs::

        FlipGrip() | obs & action
    """

    WIRE_NAME = 'flip_grip'

    def encode(self, data):
        # ``data`` belongs to the caller, so the flip goes on a copy.
        if obs_keys.GRIP in data:
            data = {**data, obs_keys.GRIP: 1.0 - data[obs_keys.GRIP]}
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        if obs_keys.TARGET_GRIP in data:
            data[obs_keys.TARGET_GRIP] = 1.0 - data[obs_keys.TARGET_GRIP]
        return data

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}
