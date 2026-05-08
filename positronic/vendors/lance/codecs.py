"""Lance codecs.

Mirror the existing LeRobot codecs but add static scalar fields that the Lance
converter writes as scalar columns in the wide per-episode table.
"""

import uuid as _uuid

import configuronic as cfn

from positronic.cfg import codecs as base
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get
from positronic.policy.codec import Codec


def _random_uuid(_episode: Episode) -> str:
    return str(_uuid.uuid4())


def _trajectory_length_for(fps: float):
    def _fn(episode: Episode) -> int:
        return int((episode.last_ts - episode.start_ts) * fps // int(1e9)) + 1

    return _fn


class _StaticScalars(Codec):
    """Adds arbitrary static scalars derived from the raw episode."""

    def __init__(self, **derivations):
        self._derivations = derivations

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive(**self._derivations)


@cfn.config(fps=15.0, horizon=1.0, binarize_grip=None, uuid=False)
def _compose(obs, action, fps: float, horizon: float | None, binarize_grip, uuid: bool):
    derivations = {
        'trajectory_length': _trajectory_length_for(fps),
        'current_task': Get('task', ''),
        'language_instruction1': Get('task', ''),
    }
    if uuid:
        derivations['uuid'] = _random_uuid
    inner = base.compose(obs=obs, action=action, fps=fps, horizon=horizon, binarize_grip=binarize_grip)
    return inner & _StaticScalars(**derivations)


ee = _compose.override(obs=base.eepose_obs.override(image_size=(512, 512)), action=base.absolute_pos_action)
