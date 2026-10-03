from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, final, overload, runtime_checkable

import numpy as np

from positronic.utils.lazy import LazySequence

from .time import Time, TimeBounds, TimeGrid, as_time, validate_queries, validate_timelines

RECORDED_TIME = 'recorded'
TIMELINE_METADATA_KEY = b'positronic.timeline'
TIMELINES_KEY = 'timelines'


T = TypeVar('T')

IndicesLike: TypeAlias = slice | Sequence[int] | np.ndarray


def _infer_item_dtype_shape(item: Any) -> tuple[Any, Any]:
    """Infer dtype and shape for a single signal element.

    Rules:
    - numpy.ndarray: dtype = array.dtype, shape = array.shape
    - numeric scalars (Python int/float or numpy integer/floating scalars): dtype = type(item), shape = ()
    - tuple: dtype, shape are tuples of per-element dtype/shape
    - other: dtype = type(item), shape = None
    """
    if isinstance(item, np.ndarray):
        return item.dtype, item.shape
    if isinstance(item, np.integer | np.floating | int | float):
        return type(item), ()
    if isinstance(item, tuple):
        dts_shapes = tuple(_infer_item_dtype_shape(x) for x in item)
        dts = tuple(ds[0] for ds in dts_shapes)
        shapes = tuple(ds[1] for ds in dts_shapes)
        return dts, shapes
    return type(item), None


@runtime_checkable
class TimeIndexerLike(Protocol, Generic[T]):
    @overload
    def __getitem__(self, key: Time | Mapping[str, int]) -> tuple[T, Time]: ...

    @overload
    def __getitem__(self, key: slice | Sequence[Time]) -> 'Signal[T]': ...


class Kind(Enum):
    NUMERIC = 'numeric'
    IMAGE = 'image'


@dataclass
class SignalMeta:
    """Metadata of a signal's values."""

    dtype: Any
    shape: Any
    kind: Kind = Kind.NUMERIC


class Signal(Sequence[tuple[T, Time]], ABC, Generic[T]):
    """An ordered record stream on a fixed set of named integer timelines.

    Coordinates never decrease, and each record advances at least one timeline.
    Backends provide batched value/timestamp reads and can optimize timestamp search.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def timelines(self) -> tuple[str, ...]: ...

    @abstractmethod
    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        """Read selected coordinates at record indices without loading values."""
        ...

    @abstractmethod
    def _values_at(self, indices: IndicesLike) -> Sequence[T]: ...

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        """Last record satisfying every requested upper bound, or -1 if none does."""
        result = []
        for query in queries:
            self._validate_selection(query.timelines)
            lo, hi = 0, len(self)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._ts_at([mid], query.timelines)[0] <= query:
                    lo = mid + 1
                else:
                    hi = mid
            result.append(lo - 1)
        return result

    def _validate_selection(self, timelines: tuple[str, ...]) -> None:
        validate_timelines(timelines)
        for name in timelines:
            if name not in self.timelines:
                raise KeyError(name)

    @overload
    def bounds(self, timelines: str) -> TimeBounds[int]: ...

    @overload
    def bounds(self, timelines: tuple[str, ...]) -> TimeBounds[Time]: ...

    @final
    def bounds(self, timelines: str | tuple[str, ...]) -> TimeBounds[int] | TimeBounds[Time]:
        """Inclusive endpoints: integers for a name, Time values for a tuple of names."""
        names = (timelines,) if isinstance(timelines, str) else timelines
        if not len(self):
            raise ValueError('Signal is empty')
        self._validate_selection(names)
        bounds = self._bounds(names)
        if isinstance(timelines, str):
            return TimeBounds(bounds.start[timelines], bounds.finish[timelines])
        return bounds

    def _bounds(self, timelines: tuple[str, ...]) -> TimeBounds[Time]:
        bounds = self._ts_at([0, len(self) - 1], timelines)
        return TimeBounds(bounds[0], bounds[1])

    @cached_property
    def meta(self) -> SignalMeta:
        if not len(self):
            raise ValueError('Signal is empty')
        dtype, shape = _infer_item_dtype_shape(self._values_at([0])[0])
        kind = (
            Kind.IMAGE
            if (dtype == np.uint8 and isinstance(shape, tuple) and len(shape) == 3 and shape[2] == 3)
            else Kind.NUMERIC
        )
        return SignalMeta(dtype=dtype, shape=shape, kind=kind)

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self):
        return self.meta.shape

    @property
    def kind(self) -> Kind:
        return self.meta.kind

    @property
    def time(self) -> TimeIndexerLike[T]:
        return _SignalViewTime(self)

    @final
    def values(self) -> Sequence[T]:
        return self._values_at(slice(None))

    @overload
    def timestamps(self, timelines: str) -> Sequence[int]: ...

    @overload
    def timestamps(self, timelines: tuple[str, ...]) -> Sequence[Time]: ...

    @final
    def timestamps(self, timelines: str | tuple[str, ...]) -> Sequence[int] | Sequence[Time]:
        """Selected coordinates: integers for a name, Time values for a tuple of names."""
        names = (timelines,) if isinstance(timelines, str) else timelines
        self._validate_selection(names)
        times = self._ts_at(slice(None), names)
        if isinstance(timelines, str):
            return LazySequence(times, lambda ts: ts[timelines])
        return times

    @overload
    def __getitem__(self, key: int) -> tuple[T, Time]: ...

    @overload
    def __getitem__(self, key: IndicesLike) -> 'Signal[T]': ...

    @final
    def __getitem__(self, key: int | IndicesLike) -> 'tuple[T, Time] | Signal[T]':
        if isinstance(key, int | np.integer):
            idx = int(key)
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(idx)
            return self._values_at([idx])[0], self._ts_at([idx], self.timelines)[0]
        if isinstance(key, slice):
            if key.step is not None and key.step <= 0:
                raise ValueError('Slice step must be positive')
            return _SignalView(self, range(*key.indices(len(self))))
        if isinstance(key, np.ndarray | Sequence) and not isinstance(key, str | bytes):
            indices = np.array(key, copy=True)
            if indices.ndim != 1:
                raise ValueError('Signal indices must be one-dimensional')
            if indices.size == 0:
                return _SignalView(self, range(0))
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError('Signal indices must be integers')
            indices[indices < 0] += len(self)
            if np.any(indices < 0) or np.any(indices >= len(self)):
                raise IndexError('Signal index out of range')
            if np.any(indices[1:] <= indices[:-1]):
                raise ValueError('Record indices must be strictly increasing')
            return _SignalView(self, indices)
        raise TypeError(f'Unsupported index type: {type(key)}')


class _SignalViewTime(Generic[T]):
    def __init__(self, signal: Signal[T]):
        self._signal = signal

    @overload
    def __getitem__(self, key: Time | Mapping[str, int]) -> tuple[T, Time]: ...

    @overload
    def __getitem__(self, key: slice | Sequence[Time]) -> Signal[T]: ...

    def __getitem__(self, key):
        if isinstance(key, Mapping):
            query = as_time(key)
            self._signal._validate_selection(query.timelines)
            idx = int(self._signal._search_ts([query])[0])
            if idx < 0:
                raise KeyError(f'No record at or before {query}')
            return self._signal[idx]
        if isinstance(key, slice):
            return self._slice(key)
        if not isinstance(key, Sequence):
            raise TypeError('Expected named coordinates, a time slice, or a sequence of Time')
        validate_queries(key)
        if not len(key):
            return _SignalView(self._signal, range(0))
        self._signal._validate_selection(key[0].timelines)
        indices = self._signal._search_ts(key)
        if any(i < 0 for i in indices):
            raise KeyError('No record at or before some requested timestamps')
        return _SignalView(self._signal, indices, timestamps=key)

    def _slice(self, selection: slice) -> Signal[T]:
        start = as_time(selection.start) if selection.start is not None else None
        stop = as_time(selection.stop) if selection.stop is not None else None
        step = as_time(selection.step) if selection.step is not None else None
        supplied = [time for time in (start, stop, step) if time is not None]
        if not supplied:
            raise ValueError('A time slice requires named endpoints')
        names = supplied[0].timelines
        self._signal._validate_selection(names)
        for time in supplied[1:]:
            supplied[0]._validate_timelines(time)
        if step is not None:
            if start is None:
                raise ValueError('Slice start is required when step is provided')
            # Validate the increment even for an empty source.
            TimeGrid(start, start, step)
            if not len(self._signal):
                return _SignalView(self._signal, range(0))
            if int(self._signal._search_ts([start])[0]) < 0:
                raise KeyError(f'No record at or before {start}')
            bound = stop if stop is not None else self._signal.bounds(names).finish
            return self[TimeGrid(start, bound, step, inclusive=stop is None)]
        if not len(self._signal) or (start is not None and stop is not None and not start < stop):
            return _SignalView(self._signal, range(0))
        first = max(0, int(self._signal._search_ts([start])[0])) if start is not None else 0
        start = start if start is not None else self._signal.bounds(names).start
        end = len(self._signal)
        if stop is not None:
            end = int(self._signal._search_ts([stop])[0]) + 1
            while end > 0 and self._signal._ts_at([end - 1], names)[0] == stop:
                end -= 1
        carried = self._signal._ts_at([first], names)[0]
        view = _SignalView(self._signal, range(first, max(first, end)), start=start if carried < start else None)
        if len(view) > 1 and not view._ts_at([0], view.timelines)[0] < view._ts_at([1], view.timelines)[0]:
            raise ValueError('The carried window timestamp is incompatible with the following record')
        return view


class _SignalView(Signal[T]):
    def __init__(
        self,
        signal: Signal[T],
        indices: Sequence[int] | np.ndarray,
        *,
        timestamps: Sequence[Time] | None = None,
        start: Time | None = None,
    ):
        self._signal = signal
        self._indices = indices
        self._timestamps = timestamps
        self._start = start

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._signal.timelines

    @cached_property
    def meta(self) -> SignalMeta:
        if not len(self):
            raise ValueError('Signal is empty')
        return self._signal.meta

    def __len__(self) -> int:
        return len(self._indices)

    def _mapped(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        if isinstance(indices, slice):
            return self._indices[indices]
        return [self._indices[int(i)] for i in indices]

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        source = self._signal._ts_at(self._mapped(indices), timelines)
        if self._timestamps is None and self._start is None:
            return source
        positions = range(*indices.indices(len(self))) if isinstance(indices, slice) else indices

        def overlay(position: int) -> Time:
            i = int(positions[position])
            query = self._timestamps[i] if self._timestamps is not None else self._start if i == 0 else None
            original = source[position]
            return Time(**{
                name: query[name] if query is not None and name in query else original[name] for name in timelines
            })

        return LazySequence(range(len(positions)), overlay)

    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        return self._signal._values_at(self._mapped(indices))

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        if self._timestamps is None and self._start is None:
            return np.searchsorted(self._indices, self._signal._search_ts(queries), side='right') - 1
        return super()._search_ts(queries)


class SignalWriter(AbstractContextManager, ABC, Generic[T]):
    """Append-only writer whose first successful record fixes the timeline set."""

    def __init__(self):
        self._last_time: Time | None = None

    def _validate_timestamps(self, timestamps: Time) -> None:
        if not isinstance(timestamps, Time):
            raise TypeError('Expected Time')
        if any(value < -(2**63) or value >= 2**63 for value in timestamps.values()):
            raise OverflowError('Stored timestamps must fit int64')
        if self._last_time is not None:
            if set(timestamps) != set(self._last_time):
                raise ValueError('Timeline names must be consistent across all appends')
            if not self._last_time < timestamps:
                raise ValueError('Timestamp is not increasing: no coordinate may decrease, and one must advance')

    @abstractmethod
    def append(self, data: T, timestamps: Time) -> None: ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...


@runtime_checkable
class SupportsEncodedRepresentation(Protocol):
    """Protocol for signals with a raw/encoded representation distinct from decoded values.

    Signals that use lossy encoding (e.g., video, compressed audio) can implement this
    protocol to expose their raw encoded data for efficient transfer without re-encoding.
    This is modality-agnostic - any signal type with lossy encoding can implement it.
    """

    @property
    def encoding_format(self) -> str:
        """Format identifier for the encoded representation.

        Returns a versioned string like 'positronic.video.v1' that identifies
        both the type of encoding and its wire format version.
        """
        ...

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        """Stream all encoded data as opaque bytes.

        The format of the bytes is defined by `encoding_format`. Receivers must
        parse the stream according to the format version.

        Yields:
            Chunks of raw encoded bytes.
        """
        ...
