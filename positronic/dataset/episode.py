from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar

import numpy as np

from .signal import Signal

EPISODE_SCHEMA_VERSION = 1
T = TypeVar('T')
SIGNAL_FACTORY_T = Callable[[], Signal[Any]]


class _EpisodeTimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'Episode') -> None:
        self.episode = episode

    def __getitem__(self, index_or_slice):
        match index_or_slice:
            case int() | np.integer() | float() | np.floating() as ts:
                # For a single timestamp, return static items and only the values for signals
                sampled = {key: sig.time[ts][0] for key, sig in self.episode.signals.items()}
                return {**self.episode.static, **sampled}
            case slice() as sl if sl.step is None:
                raise KeyError('Episode.time[start:stop] is not supported; use a step or explicit timestamps')
            case slice() | list() | tuple() | np.ndarray() as req:
                # For slice or sequence of timestamps, return a dict:
                # - static items as-is
                # - dynamic signals mapped to sequences of values sampled at requested timestamps
                # If slice with step but no stop provided, default stop to episode.last_ts (+1 for end-exclusive)
                if isinstance(req, slice) and req.step is not None and req.stop is None:
                    req = slice(req.start, self.episode.last_ts + 1, req.step)
                result: dict[str, Any] = self.episode.static.copy()
                for key, sig in self.episode.signals.items():
                    view = sig.time[req]
                    # Extract the full sequence of values corresponding to the time selection
                    result[key] = view._values_at(slice(None))
                return {**self.episode.static, **result}
            case _:
                raise TypeError(f'Invalid index type: {type(index_or_slice)}')


class Episode(ABC, Mapping[str, Any]):
    """Abstract base class for an Episode (core concept).

    Subclasses must implement the following methods:
    - __getitem__ - return a Signal or static value by name
    - __iter__ - return an iterator over the keys
    - __len__ - return the number of keys
    """

    # TODO: Replace meta with static data starting with 'meta.' prefix.
    @property
    @abstractmethod
    def meta(self) -> dict:
        pass

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        out: dict[str, Signal[Any]] = {}
        for k in self:
            v = self[k]
            if isinstance(v, Signal):
                out[k] = v
        return out

    @property
    def static(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in self:
            v = self[k]
            if not isinstance(v, Signal):
                out[k] = v
        return out

    @property
    def start_ts(self):
        values = [sig.start_ts for sig in self.signals.values()]
        if not values:
            raise ValueError('Episode has no signals')
        return max(values)

    @property
    def last_ts(self):
        values = [sig.last_ts for sig in self.signals.values()]
        if not values:
            raise ValueError('Episode has no signals')
        return max(values)

    @property
    def duration_ns(self):
        if not self.signals:
            return 0
        return self.last_ts - self.start_ts

    @property
    def time(self):
        return _EpisodeTimeIndexer(self)


class EpisodeContainer(Episode):
    """In-memory view over an Episode's items."""

    def __init__(
        self, signals: dict[str, Signal[Any]], static: dict[str, Any] | None = None, meta: dict[str, Any] | None = None
    ) -> None:
        self._signals = signals
        self._static = static or {}
        self._meta = meta or {}

    def keys(self) -> list[str]:
        return [*self._signals.keys(), *self._static.keys()]

    def __iter__(self) -> Iterator[str]:
        yield from self._signals.keys()
        yield from self._static.keys()

    def __len__(self) -> int:
        return len(self._signals) + len(self._static)

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name in self._signals:
            return self._signals[name]
        if name in self._static:
            return self._static[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        return dict(self._signals)

    @property
    def static(self) -> dict[str, Any]:
        return dict(self._static)


class EpisodeWriter(AbstractContextManager, ABC, Generic[T]):
    """Abstract interface for recording an episode's dynamic and static data."""

    @abstractmethod
    def append(self, signal_name: str, data: T, ts_ns: int, extra_ts: dict[str, int] | None = None) -> None:
        """Append a sample for the named signal."""
        pass

    @abstractmethod
    def set_static(self, name: str, data: Any) -> None:
        """Record a static (per-episode) item by key."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize resources on context-manager exit."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """Abort the write and discard any partially written data."""
        pass
