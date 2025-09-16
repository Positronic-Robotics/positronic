from abc import ABC, abstractmethod
import collections.abc
from typing import Sequence

import numpy as np

from .episode import Episode, EpisodeWriter


class DatasetWriter(ABC):
    """Abstract factory for creating new Episodes within a dataset."""

    @abstractmethod
    def new_episode(self) -> EpisodeWriter:
        """Allocate and return a writer for a new episode."""
        pass


class Dataset(ABC, collections.abc.Sequence[Episode]):
    """Ordered collection of Episodes with sequence-style access."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Episode:
        pass
