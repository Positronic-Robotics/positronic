"""Named timestamp values, collections, and sampling."""

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Generic, NamedTuple, TypeVar, overload

import numpy as np

from pimm.time import Time, validate_timelines
from pimm.time import validate_timeline as validate_timeline
from positronic.utils.lazy import LazySequence

HARNESS_WORLD = 'harness.world'
HARNESS_WALL = 'harness.wall'

Coordinate = TypeVar('Coordinate', int, Time)


class TimeBounds(NamedTuple, Generic[Coordinate]):
    """Inclusive timestamp endpoints on one timeline or a named set of timelines."""

    start: Coordinate
    finish: Coordinate


def validate_queries(queries: Sequence[Time]) -> None:
    for i, query in enumerate(queries):
        if not isinstance(query, Time):
            raise TypeError('A timestamp batch must contain Time values')
        if i and not queries[i - 1] < query:
            raise ValueError('Timestamp queries must be strictly increasing')


def search_timestamps(columns: Mapping[str, np.ndarray], queries: Sequence[Time]) -> np.ndarray:
    """Find the last row satisfying all bounds without narrowing Python integer queries."""
    if not len(queries):
        return np.empty(0, dtype=np.int64)
    names = queries[0].timelines
    for query in queries:
        if set(query) != set(names):
            raise ValueError('Queries must use the same timeline names')
    indices = np.full(len(queries), len(columns[names[0]]) - 1, dtype=np.int64)
    low, high = -(2**63), 2**63 - 1
    for name in names:
        values = [query[name] for query in queries]
        bounded = np.array([min(high, max(low, value)) for value in values], dtype=np.int64)
        found = np.searchsorted(columns[name], bounded, side='right') - 1
        found[np.array([value < low for value in values])] = -1
        indices = np.minimum(indices, found)
    return indices


class TimeArray(Sequence[Time]):
    """Immutable numeric timestamp columns with shared timeline names and lazy row values."""

    def __init__(self, timelines: tuple[str, ...], values: np.ndarray):
        validate_timelines(timelines, allow_empty=len(values) == 0)
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(timelines):
            raise ValueError('Timestamp rows must match the timeline names')
        if array.size and not np.issubdtype(array.dtype, np.integer):
            raise TypeError('Timestamp coordinates must be integers')
        if array.size and (int(array.min()) < -(2**63) or int(array.max()) >= 2**63):
            raise OverflowError('Stored timestamps must fit int64')
        self._timelines = timelines
        self._columns = MappingProxyType({name: i for i, name in enumerate(timelines)})
        # Bytes own the storage, so neither rows nor their NumPy bases can enable writes.
        self._values = np.frombuffer(array.astype(np.int64).tobytes(), dtype=np.int64).reshape(array.shape)

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._timelines

    @classmethod
    def from_times(cls, times: Sequence[Time], timelines: tuple[str, ...]) -> 'TimeArray':
        if isinstance(times, cls):
            return times.select(timelines)
        return cls(
            timelines,
            np.array([[ts[name] for name in timelines] for ts in times], dtype=np.int64).reshape(
                len(times), len(timelines)
            ),
        )

    def __len__(self) -> int:
        return len(self._values)

    class _Row(Mapping[str, int]):
        def __init__(self, parent: 'TimeArray', row: np.ndarray):
            self._parent = parent
            self._row = row

        def __getitem__(self, name: str) -> int:
            return int(self._row[self._parent._columns[name]])

        def __iter__(self) -> Iterator[str]:
            return iter(self._parent.timelines)

        def __len__(self) -> int:
            return len(self._parent.timelines)

    @overload
    def __getitem__(self, key: int) -> Time: ...

    @overload
    def __getitem__(self, key: slice) -> 'TimeArray': ...

    def __getitem__(self, key: int | slice) -> 'Time | TimeArray':
        if isinstance(key, slice):
            return self.take(key)
        result = Time.__new__(Time)
        result._coordinates = self._Row(self, self._values[key])
        return result

    def take(self, indices: slice | Sequence[int] | np.ndarray) -> 'TimeArray':
        if not isinstance(indices, slice):
            indices = np.asarray(indices, dtype=np.int64)
        return TimeArray(self.timelines, self._values[indices])

    def select(self, timelines: tuple[str, ...]) -> 'TimeArray':
        validate_timelines(timelines)
        if timelines == self.timelines:
            return self
        return TimeArray(timelines, self._values[:, [self._columns[name] for name in timelines]])

    def search(self, queries: Sequence[Time]) -> np.ndarray:
        columns = {name: self._values[:, i] for name, i in self._columns.items()}
        return search_timestamps(columns, queries)

    def validate_order(self, *, strict: bool = True) -> None:
        if len(self) < 2:
            return
        previous, following = self._values[:-1], self._values[1:]
        if np.any(following < previous) or (strict and np.any(np.all(following == previous, axis=1))):
            raise ValueError('Timestamps must be non-decreasing in every timeline and strictly increasing in one')


class TimeGrid(Sequence[Time]):
    """Lazy, componentwise sampling grid with an explicit named increment."""

    def __init__(self, start: Time, stop: Time, step: Time, *, inclusive: bool = False):
        start._validate_timelines(stop)
        start._validate_timelines(step)
        if any(value < 0 for value in step.values()) or not any(step.values()):
            raise ValueError('A time step must be non-negative and advance at least one timeline')
        self._start, self._step = start, step
        self._length = 0
        if start <= stop:
            self._length = min((stop[name] - start[name]) // delta for name, delta in step.items() if delta) + 1
            if not inclusive and self[self._length - 1] == stop:
                self._length -= 1

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, key: int) -> Time: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[Time]: ...

    def __getitem__(self, key: int | slice) -> Time | Sequence[Time]:
        if isinstance(key, slice):
            return LazySequence(range(*key.indices(len(self))), self.__getitem__)
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        return Time(**{name: value + key * self._step[name] for name, value in self._start.items()})
