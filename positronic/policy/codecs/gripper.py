from collections.abc import Sequence

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
    """Serve a model whose gripper speaks an aperture (1 = open) on positronic's 1 = closed wire: flips ``grip``
    entering the model and ``target_grip`` leaving it.

    With ``arms`` it flips ``grip.{arm}`` and ``target_grip.{arm}`` of each arm named, and publishes version 2.
    Without, it flips the one-arm channels and publishes version 1. Compose to the left of obs/action codecs::

        FlipGrip() | obs & action
    """

    WIRE_NAME = 'flip_grip'
    WIRE_VERSION = 2
    ONE_ARM_VERSION = 1

    def __init__(self, arms: Sequence[str] | None = None):
        self._arms = None if arms is None else tuple(arms)
        names = (None,) if self._arms is None else self._arms
        self._grips = tuple(obs_keys.arm_channel(obs_keys.GRIP, arm) for arm in names)
        self._target_grips = tuple(obs_keys.arm_channel(obs_keys.TARGET_GRIP, arm) for arm in names)

    def encode(self, data):
        # ``data`` belongs to the caller, so the flip goes on a copy.
        flipped = {key: 1.0 - data[key] for key in self._grips if key in data}
        return {**data, **flipped} if flipped else data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Identity()

    def _decode_single(self, data: dict) -> dict:
        for key in self._target_grips:
            if key in data:
                data[key] = 1.0 - data[key]
        return data

    def to_spec(self):
        if self._arms is None:
            return {NAME: self.WIRE_NAME, VERSION: self.ONE_ARM_VERSION}
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'arms': list(self._arms)}}
