from typing import Any

from ..dataset import Dataset
from ..episode import Episode
from .episode import EpisodeTransform, TransformedEpisode


def _merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            dst[key] = _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


class TransformedDataset(Dataset):
    """Transform a dataset into a new view of the dataset."""

    def __init__(self, dataset: Dataset, *transforms: EpisodeTransform, pass_through: bool | list[str] = False):
        self._dataset = dataset
        self._transforms = transforms
        self._pass_through = pass_through
        if not isinstance(pass_through, bool):
            self._pass_through = tuple(pass_through)

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        episode = self._dataset[index]
        return TransformedEpisode(episode, *self._transforms, pass_through=self._pass_through)

    @property
    def meta(self) -> dict[str, Any]:
        result = self._dataset.meta.copy()
        for tf in self._transforms:
            result = _merge(result, tf.meta)
        return result
