"""Episode transforms for model training."""

from collections.abc import Sequence
from functools import partial

import numpy as np

from positronic import geom, keys
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, FromValue, Group, map_signals
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import change_frame, ee_frames
from positronic.drivers.roboarm.models import EEFrame, ee_frames_entry

_QUAT = geom.Rotation.Representation.QUAT


class ChangeEEFrame(EpisodeTransform):
    """Move recorded poses into ``frame``, and state where they now sit in ``EE_FRAMES``.

    Every field it moves must be recorded at ``frame.site`` with no offset, since the offset is measured from there.
    """

    def __init__(self, frame: EEFrame, fields: Sequence[str] = (keys.EE_POSE, keys.TARGET_EE_POSE, keys.ROBOT_COMMAND)):
        self._frame = frame
        self._fields = tuple(fields)

    def __call__(self, episode: Episode) -> Episode:
        frames = ee_frames(episode)
        moved = [key for key in self._fields if key in episode]
        for key in moved:
            if key not in frames:
                raise ValueError(f'{key!r} states no {roboarm_keys.EE_FRAMES} entry, so its site is unknown')
            recorded = frames[key]
            if recorded.site != self._frame.site or not np.allclose(recorded.offset.as_matrix, np.eye(4)):
                raise ValueError(
                    f'{key!r} sits at {recorded.offset.as_vector(_QUAT).tolist()} from {recorded.site!r}; the frame '
                    f'transform requires poses at {self._frame.site!r} with no offset'
                )
        convert = map_signals(partial(change_frame, transform=self._frame.offset), moved)
        stated = ee_frames_entry({**frames, **dict.fromkeys(moved, self._frame)})
        frame = Derive(meta=None, **{name: FromValue(value) for name, value in stated.items()})
        return Group(frame, convert)(episode)

    @property
    def meta(self):
        return ee_frames_entry(dict.fromkeys(self._fields, self._frame))
