from typing import Any

from positronic.utils import merge_dicts

from ..dataset import Dataset
from ..episode import Episode
from .episode import EpisodeTransform, TransformedEpisode


class TransformedDataset(Dataset):
    """Transform a dataset into a new view of the dataset."""

    def __init__(self, dataset: Dataset, *transforms: EpisodeTransform, extra_meta: dict[str, Any] = None):
        self._dataset = dataset
        self._transforms = transforms
        self._extra_meta = extra_meta or {}
        self._episode_cache: dict[int, Episode] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        if index not in self._episode_cache:
            self._episode_cache[index] = TransformedEpisode(self._dataset[index], *self._transforms)
        return self._episode_cache[index]

    @property
    def meta(self) -> dict[str, Any]:
        result = self._dataset.meta.copy() | self._extra_meta
        for tf in self._transforms:
            result = merge_dicts(result, tf.meta)
        return result
