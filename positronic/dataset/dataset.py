import collections.abc
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from .episode import Episode, EpisodeWriter
from .signal import IndicesLike


class DatasetWriter(AbstractContextManager, ABC):
    """Abstract factory for creating new Episodes within a dataset."""

    @abstractmethod
    def new_episode(self) -> EpisodeWriter:
        """Allocate and return a writer for a new episode."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize resources on context-manager exit."""
        ...


class Dataset(ABC, collections.abc.Sequence[Episode]):
    """Ordered collection of Episodes with sequence-style access."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _get_episode(self, index: int) -> Episode:
        """Return the episode at a single, already-normalized index."""
        pass

    def __getitem__(self, index_or_slice: int | IndicesLike) -> Episode | list[Episode]:
        """Return an Episode or list of Episodes after normalizing integer-based indices."""
        length = len(self)

        def normalize_index(idx: int) -> int:
            if idx < 0:
                idx += length
            if not (0 <= idx < length):
                raise IndexError('Index out of range')
            return idx

        if isinstance(index_or_slice, slice):
            start, stop, step = index_or_slice.indices(length)
            return [self._get_episode(i) for i in range(start, stop, step)]

        if isinstance(index_or_slice, np.ndarray):
            idxs = np.asarray(index_or_slice)
            if np.issubdtype(idxs.dtype, np.bool_):
                raise TypeError('Boolean indexing is not supported')
            return [self._get_episode(normalize_index(int(i))) for i in idxs]

        if isinstance(index_or_slice, collections.abc.Sequence) and not isinstance(index_or_slice, str | bytes):
            idxs = np.asarray(index_or_slice)
            if np.issubdtype(idxs.dtype, np.bool_):
                raise TypeError('Boolean indexing is not supported')
            return [self._get_episode(normalize_index(int(i))) for i in idxs]

        return self._get_episode(normalize_index(int(index_or_slice)))

    @property
    def meta(self) -> dict[str, Any]:
        """Return dataset metadata."""
        return {}

    def __add__(self, other: 'Dataset') -> 'Dataset':
        if not isinstance(other, Dataset):
            return NotImplemented
        return ConcatDataset(self, other)

    def __radd__(self, other: 'Dataset | int') -> 'Dataset':
        # Support sum([...]) by treating 0 as the additive identity.
        if other == 0:
            return self
        if isinstance(other, Dataset):
            return ConcatDataset(other, self)
        return NotImplemented


class CachedDataset(Dataset):
    """Dataset wrapper that caches episodes for repeated access."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._cache: dict[int, Episode] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        ep = self._cache.get(index)
        if ep is None:
            ep = self._dataset[index]
            self._cache[index] = ep
        return ep

    @property
    def meta(self) -> dict[str, Any]:
        return self._dataset.meta


class ConcatDataset(Dataset):
    """Concatenate two datasets together."""

    def __init__(self, *datasets: Dataset):
        if len(datasets) == 0:
            raise ValueError('ConcatDataset requires at least one dataset')
        flattened: list[Dataset] = []
        for dataset in datasets:
            if isinstance(dataset, ConcatDataset):
                flattened.extend(dataset._datasets)
            else:
                flattened.append(dataset)
        self._datasets: tuple[Dataset, ...] = tuple(flattened)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    def _get_episode(self, index: int) -> Episode:
        original_index = index
        for ds in self._datasets:
            if index < len(ds):
                return ds[index]
            index -= len(ds)
        # We should never reach here because of the upfront bounds check.
        raise IndexError(f'Index {original_index} out of range for ConcatDataset of length {len(self)}')


class FilterDataset(Dataset):
    """View of a dataset containing only episodes matching a predicate."""

    def __init__(self, dataset: Dataset, predicate: collections.abc.Callable[[Episode], bool]):
        self._dataset = dataset
        self._predicate = predicate
        self._indices: list[int] | None = None

    def _ensure_indices(self) -> list[int]:
        if self._indices is None:
            self._indices = [i for i in range(len(self._dataset)) if self._predicate(self._dataset[i])]
        return self._indices

    def __len__(self) -> int:
        return len(self._ensure_indices())

    def _get_episode(self, index: int) -> Episode:
        return self._dataset[self._ensure_indices()[index]]

    @property
    def meta(self) -> dict[str, Any]:
        return self._dataset.meta
