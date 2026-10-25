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


# The grip channels of a single-arm embodiment and of each arm of a two-arm one.
_ARMS = (None, *obs_keys.BIMANUAL_ARMS)
_GRIPS = tuple(obs_keys.arm_channel(obs_keys.GRIP, arm) for arm in _ARMS)
_TARGET_GRIPS = tuple(obs_keys.arm_channel(obs_keys.TARGET_GRIP, arm) for arm in _ARMS)


class FlipGrip(Codec):
    """Serve a model whose gripper speaks an aperture (1 = open) on positronic's 1 = closed wire: flips every
    arm's ``grip`` entering the model and every arm's ``target_grip`` leaving it.

    Compose to the left of obs/action codecs::

        FlipGrip() | obs & action
    """

    WIRE_NAME = 'flip_grip'

    def encode(self, data):
        # ``data`` belongs to the caller, so the flip goes on a copy.
        flipped = {key: 1.0 - data[key] for key in _GRIPS if key in data}
        return {**data, **flipped} if flipped else data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        for key in _TARGET_GRIPS:
            if key in data:
                data[key] = 1.0 - data[key]
        return data

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}
