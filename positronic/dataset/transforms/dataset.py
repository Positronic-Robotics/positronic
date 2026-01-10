import weakref
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
        # WeakValueDictionary allows episodes to be garbage collected when no longer referenced.
        # Without this, cached episodes would keep all loaded signal data (parquet) in memory.
        self._episode_cache: weakref.WeakValueDictionary[int, Episode] = weakref.WeakValueDictionary()

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        ep = self._episode_cache.get(index)
        if ep is None:
            ep = TransformedEpisode(self._dataset[index], *self._transforms)
            self._episode_cache[index] = ep
        return ep

    @property
    def meta(self) -> dict[str, Any]:
        result = self._dataset.meta.copy() | self._extra_meta
        for tf in self._transforms:
            result = merge_dicts(result, tf.meta)
        return result
