"""Lance codecs.

Mirror the existing LeRobot codecs but add static scalar fields that the Lance
converter will write as scalar columns in the wide per-episode table (uuid,
trajectory_length, current_task, language_instruction1).
"""

import uuid as _uuid

import configuronic as cfn

from positronic.cfg import codecs as base
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get
from positronic.policy.codec import Codec


def _random_uuid(_episode: Episode) -> str:
    return str(_uuid.uuid4())


class _LanceScalars(Codec):
    """Adds static scalars derived from the raw episode for Lance table columns."""

    def __init__(self, fps: float):
        self._fps = fps

    def _trajectory_length(self, episode: Episode) -> int:
        # Count matches np.arange(start, last+1, 1e9/fps) used by the converter loop
        return int((episode.last_ts - episode.start_ts) * self._fps // int(1e9)) + 1

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive(
            uuid=_random_uuid,
            trajectory_length=self._trajectory_length,
            current_task=Get('task', ''),
            language_instruction1=Get('task', ''),
        )


@cfn.config(fps=15.0, horizon=1.0, binarize_grip=None)
def _compose(obs, action, fps: float, horizon: float | None, binarize_grip):
    return base.compose(obs=obs, action=action, fps=fps, horizon=horizon, binarize_grip=binarize_grip) & _LanceScalars(
        fps=fps
    )


ee = _compose.override(obs=base.eepose_obs.override(image_size=(512, 512)), action=base.absolute_pos_action)
