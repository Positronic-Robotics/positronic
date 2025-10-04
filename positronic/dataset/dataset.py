import collections.abc
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

import numpy as np

from .episode import Episode, EpisodeWriter
from .signal import IndicesLike, SignalMeta


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
    @abstractmethod
    def signals_meta(self) -> dict[str, SignalMeta]:
        """Return signal metadata keyed by signal name for the dataset schema."""
        pass
