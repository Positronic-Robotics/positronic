"""Episode transforms for model training."""

from collections.abc import Sequence
from functools import partial

import numpy as np

from positronic import geom, keys
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, FromValue, Group, map_signals
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import assert_default_frame, change_frame, ee_frame
from positronic.drivers.roboarm.models import DEFAULT_FRAME
from positronic.policy.codecs.metadata import LEROBOT_FEATURES, lerobot_image

_QUAT = geom.Rotation.Representation.QUAT


def _image_column(source: str, size: tuple[int, int], resample: image.Resample, episode: Episode):
    return image.resize_with_pad(*size, signal=episode[source], method=resample)


def image_observation(
    images: dict[str, tuple[str, tuple[int, int]]], resample: image.Resample = image.Resample.BILINEAR
) -> EpisodeTransform:
    """Derive named padded image columns and their training feature descriptions."""
    return Derive(
        meta={LEROBOT_FEATURES: {name: lerobot_image(*size) for name, (_, size) in images.items()}},
        **{name: partial(_image_column, source, size, resample) for name, (source, size) in images.items()},
    )


class ChangeEEFrame(EpisodeTransform):
    """Move recorded poses from the default frame and record the resulting frame."""

    def __init__(
        self,
        transform: geom.Transform3D,
        fields: Sequence[str] = (keys.EE_POSE, keys.TARGET_EE_POSE, keys.ROBOT_COMMAND),
    ):
        self._transform = transform
        self._fields = tuple(fields)

    def __call__(self, episode: Episode) -> Episode:
        assert_default_frame(episode)
        existing = ee_frame(episode)
        if not np.allclose(existing.as_matrix, np.eye(4)):
            raise ValueError(
                f'Episode poses already sit at {existing.as_vector(_QUAT).tolist()} relative to '
                f'{DEFAULT_FRAME!r}; the frame transform requires poses in the default frame'
            )
        convert = map_signals(
            partial(change_frame, transform=self._transform), [key for key in self._fields if key in episode]
        )
        frame = Derive(meta=None, **{roboarm_keys.EE_FRAME: FromValue(self._transform.as_vector(_QUAT))})
        return Group(frame, convert)(episode)

    @property
    def meta(self):
        return {roboarm_keys.EE_FRAME: self._transform.as_vector(_QUAT).tolist()}
