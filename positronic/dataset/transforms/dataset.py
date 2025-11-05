from typing import Any

from positronic.utils import merge_dicts

from ..dataset import Dataset
from ..episode import Episode
from .episode import EpisodeTransform, TransformedEpisode


class TransformedDataset(Dataset):
    """Transform a dataset into a new view of the dataset."""

    def __init__(self, dataset: Dataset, *transforms: EpisodeTransform):
        self._dataset = dataset
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        return TransformedEpisode(self._dataset[index], *self._transforms)

    @property
    def meta(self) -> dict[str, Any]:
        result = self._dataset.meta.copy()
        for tf in self._transforms:
            result = merge_dicts(result, tf.meta)
        return result
