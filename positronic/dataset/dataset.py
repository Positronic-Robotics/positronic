import collections.abc
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Sequence

import numpy as np

from .episode import Episode, EpisodeWriter
from .signal import SignalMeta


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
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Episode:
        pass

    @property
    @abstractmethod
    def signals_meta(self) -> dict[str, SignalMeta]:
        pass
