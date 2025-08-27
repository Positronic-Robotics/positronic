from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
import collections.abc
from typing import Any, Generic, Protocol, Sequence, Tuple, TypeVar, runtime_checkable
from collections.abc import Sequence as SequenceABC

import numpy as np

T = TypeVar('T')


@runtime_checkable
class TimeIndexerLike(Protocol, Generic[T]):

    def __getitem__(self, key: int | slice | Sequence[int] | np.ndarray) -> Tuple[T, int] | "Signal[T]":
        ...


class Signal(Sequence[Tuple[T, int]], ABC, Generic[T]):
    """Strictly typed, stepwise function of time.

    A Signal is an ordered sequence of (value, ts_ns) where timestamps are
    strictly increasing. The value at time t is defined as the last value at
    or before t; the signal is undefined for t < first_timestamp.

    Access patterns:
    - Index-based: integer/slice/array indexing by position. Slices and arrays
      return Signal views; implementations aim to share storage when possible.
    - Time-based: `signal.time[...]` provides timestamp access (snapshots,
      windows, stepped sampling). Stepped time sampling materializes the
      requested timestamps.

    Concrete implementations decide storage/layout (e.g., Parquet for
    scalar/vector, encoded video + Parquet index for images) but must follow
    the semantics above.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of records."""
        raise NotImplementedError

    @abstractmethod
    def _ts_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> int | Sequence[int] | np.ndarray:
        """Return timestamp(s) at given index(es). Input indices are in ascending order and in range [0, len(self)).

        - If input is int, returns int timestamp.
        - If input is list-like, returns an array/sequence of timestamps in the same order.
        """
        raise NotImplementedError

    @abstractmethod
    def _values_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> T | Sequence[T]:
        """Return value(s) at given index(es). Input indices are in ascending order and in range [0, len(self))."""
        raise NotImplementedError

    @abstractmethod
    def _search_ts(self, ts_or_array: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        """Return floor index for given timestamp(s).

        For each timestamp t, returns the largest i such that ts[i] <= t.
        Returns -1 when t precedes the first record. For arrays, returns an
        np.ndarray[int64] with the same shape.
        """
        raise NotImplementedError

    # Public API

    @property
    def start_ts(self) -> int:
        """Returns the timestamp of the first record in the signal."""
        if len(self) == 0:
            raise ValueError("Signal is empty")
        return self._ts_at(0)

    @property
    def last_ts(self) -> int:
        """Returns the timestamp of the last record in the signal."""
        if len(self) == 0:
            raise ValueError("Signal is empty")
        return self._ts_at(len(self) - 1)

    @property
    def time(self) -> TimeIndexerLike[T]:
        """Returns an indexer for accessing Signal data by timestamp.

        Similar to pandas.loc, this property provides a timestamp-based indexer.

        Usage:
            signal.time[ts_ns] -> Tuple[T, int]
                Returns (value, timestamp_ns) for the closest record at or before ts_ns.
                Raises KeyError if there's no record at or before ts_ns.

            signal.time[start_ts_ns:end_ts_ns] -> Signal[T]
                Returns a Signal view containing all records in [start_ts_ns, end_ts_ns)
                (i.e. the end timestamp is not included)

            signal.time[start_ts_ns:end_ts_ns:step_ts_ns] -> Signal[T]
                Returns a Signal sampled at "requested" timestamps t_i = start_ts_ns + i * step_ts_ns,
                for i >= 0 while t_i < end_ts_ns. Each element of the returned Signal is
                (value_at_or_before_t_i, t_i). If t_i is before the first available record,
                that t_i is skipped. A non-positive step_ts_ns yields an empty result. The
                end timestamp is exclusive.

            signal.time[[t1, t2, ...]] -> Signal[T]
                Returns a Signal sampled at the provided timestamps. Each element is
                (value_at_or_before_t_i, t_i). Raises KeyError if any t_i precedes the first record.
        """
        return _SignalViewTime(self)

    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Tuple[T, int] | "Signal[T]":
        """Access the Signal data by index or slice.

        Args:
            index_or_slice: Integer index, slice object, or array-like of indices

        Returns:
            If index: Tuple of (value, timestamp_ns)
            If slice/array-like: Signal[T] view of the original data
        """
        match index_or_slice:
            case int() as idx:
                if idx < 0:
                    idx = len(self) + idx
                if idx < 0 or idx >= len(self):
                    raise IndexError(f"Index {idx} out of range for Signal of length {len(self)}")
                return self._values_at(idx), self._ts_at(idx)
            case slice() as sl:
                if sl.step is not None and sl.step <= 0:
                    raise ValueError("Slice step must be positive")
                return _SignalView(self, range(*sl.indices(len(self))))
            case np.ndarray() | SequenceABC() as idxs:
                arr = np.asarray(idxs)
                if arr.size == 0:
                    return _SignalView(self, arr.astype(np.int64))
                if arr.dtype == np.bool_:
                    raise IndexError("Boolean mask indexing is not supported for Signal")
                if not np.issubdtype(arr.dtype, np.integer):
                    raise TypeError(f"Unsupported index dtype: {arr.dtype}")
                arr[arr < 0] += len(self)
                if (arr < 0).any() or (arr >= len(self)).any():
                    raise IndexError("Index out of range for Signal")
                return _SignalView(self, arr)
            case _:
                raise TypeError(f"Unsupported index type: {type(index_or_slice)}")


class _SignalViewTime(TimeIndexerLike[T], Generic[T]):

    def __init__(self, signal: Signal[T]):
        self._signal = signal

    def __getitem__(self, ts_or_array: int | Sequence[int] | np.ndarray | slice) -> Tuple[T, int] | "Signal[T]":
        """Access the Signal data by timestamp.
        The sequence of timestamps must be non-decreasing, otherwise the behavior is undefined.

        Returns:
            - Tuple[T, int]: When a single timestamp is provided.
            - Signal[T]: When a sequence of timestamps is provided.
        """
        match ts_or_array:
            case int() as ts:
                idx = self._signal._search_ts(ts)
                if idx < 0:
                    raise KeyError(f"Timestamp {ts} precedes the first record")
                return self._signal._values_at(idx), self._signal._ts_at(idx)
            case slice() as sl if sl.step is None:
                if len(self._signal) == 0:  # Empty signal -> empty view
                    return _SignalView(self._signal, range(0, 0))
                start = sl.start if sl.start is not None else self._signal.start_ts
                stop = sl.stop if sl.stop is not None else self._signal.last_ts + 1
                start_id, end_id = self._signal._search_ts([start, stop])
                if stop > self._signal._ts_at(end_id):
                    end_id += 1
                kwargs = {}
                # Inject start when not exact and within bounds
                if start_id > -1 and self._signal._ts_at(start_id) < start:
                    kwargs['start_ts'] = start
                if start_id < 0:
                    start_id = 0
                return _SignalView(self._signal, range(start_id, end_id, 1), **kwargs)
            case slice() as sl if sl.step is not None and sl.step <= 0:
                raise ValueError("Slice step must be positive")
            case slice() as sl if sl.step is not None and sl.start is None:
                raise ValueError("Slice start is required when step is provided")
            case np.ndarray() | SequenceABC() | slice() as tss:
                if isinstance(tss, slice):
                    if len(self._signal) == 0:  # Empty signal -> empty view
                        return _SignalView(self._signal, range(0, 0))
                    start = tss.start if tss.start is not None else self._signal.start_ts
                    stop = tss.stop if tss.stop is not None else self._signal.last_ts + 1
                    if start < self._signal.start_ts:
                        raise KeyError(f"Timestamp {start} precedes the first record")
                    tss = np.arange(start, stop, tss.step)
                idxs = np.asarray(self._signal._search_ts(tss))
                if (idxs < 0).any():  # Any timestamp before first must raise
                    raise KeyError("No record at or before some of the requested timestamps")
                return _SignalView(self._signal, idxs, tss)
            case _:
                raise TypeError(f"Unsupported index type: {type(ts_or_array)}")


class _SignalView(Signal[T], Generic[T]):

    def __init__(self,
                 signal: Signal[T],
                 indices: Sequence[int],
                 timestamps: Sequence[int] | None = None,
                 start_ts: int | None = None):
        self._signal = signal
        self._indices = indices
        assert timestamps is None or start_ts is None, "Only one of timestamps or start_ts can be provided"
        self._timestamps = timestamps
        self._start_ts = start_ts

    def __len__(self) -> int:
        return len(self._indices)

    def _ts_at(self, index_or_indices: int | Sequence[int] | np.ndarray | slice) -> int | Sequence[int] | np.ndarray:
        if self._timestamps is not None:
            return self._timestamps[index_or_indices]
        match index_or_indices:
            case int() | np.integer() as i:
                i = int(i)
                return self._signal._ts_at(self._indices[i]) if self._start_ts is None or i > 0 else self._start_ts
            case slice() | np.ndarray() | SequenceABC() as idxs:
                result = self._signal._ts_at(self._indices[idxs])
                if self._start_ts is not None:
                    if isinstance(idxs, slice):
                        idxs = np.arange(*idxs.indices(len(self)))
                    else:
                        idxs = np.asarray(idxs)
                    result[idxs == 0] = self._start_ts
                return result
            case _:
                raise TypeError(f"Unsupported index type: {type(index_or_indices)}")

    def _values_at(self, index_or_indices: int | Sequence[int] | np.ndarray) -> T | Sequence[T]:
        match index_or_indices:
            case int() | np.integer() | slice() | np.ndarray() | SequenceABC():
                return self._signal._values_at(self._indices[index_or_indices])
            case _:
                raise TypeError(f"Unsupported index type: {type(index_or_indices)}")

    def _search_ts(self, ts_or_array: int | Sequence[int] | np.ndarray) -> int | np.ndarray:
        match ts_or_array:
            case int() | slice() | np.ndarray() | SequenceABC():
                if self._timestamps is None:
                    parent_idx = self._signal._search_ts(ts_or_array)
                    return np.searchsorted(self._indices, parent_idx, side='right') - 1
                else:
                    return np.searchsorted(self._timestamps, ts_or_array, side='right') - 1
            case _:
                raise TypeError(f"Unsupported index type: {type(ts_or_array)}")


class SignalWriter(AbstractContextManager, ABC, Generic[T]):
    """Append-only writer for Signals.

    Writers accept (data, ts_ns) pairs with strictly increasing timestamps and
    consistent shape/dtype per signal. They optimize for low-latency appends
    during recording and finalize persistence on `finish()`. After finishing,
    further `append` calls must fail. Concrete writers choose the backing store
    (e.g., Parquet for scalar/vector; video + Parquet index for images).
    """

    @abstractmethod
    def append(self, data: T, ts_ns: int) -> None:
        """Appends data with timestamp.

        Args:
            data: Data to append (must have consistent shape and dtype)
            ts_ns: Timestamp in nanoseconds (must be strictly increasing)

        Raises:
            RuntimeError: If writer has been finished/closed
            ValueError: If timestamp is not increasing or data shape/dtype doesn't match
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    @abstractmethod
    def abort(self) -> None:
        """Abort this writer and discard partial outputs. All appends after this point will fail.

        Implementations should clean up any artifacts and invalidate the writer.
        """
        pass


class Episode(ABC):
    """Abstract base class for an Episode (core concept).

    Represents a collection of dynamic signals and static episode-level items
    that share a common time axis. Concrete implementations may load from disk
    or present in-memory views, but must provide the same read API.
    """

    @property
    @abstractmethod
    def keys(self):
        """Names of all items (dynamic signals + static items)."""
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> Signal[Any] | Any:
        """Access by name: returns a Signal for dynamic items or the value for static items."""
        pass

    @property
    @abstractmethod
    def meta(self) -> dict:
        """Read-only system metadata for this episode.

        Contains only system-generated fields (e.g., creation timestamp,
        schema version, writer info). Not included in `keys` and not
        accessible via `__getitem__`.
        """
        pass

    @property
    @abstractmethod
    def start_ts(self) -> int:
        """Latest start timestamp across all dynamic signals."""
        pass

    @property
    @abstractmethod
    def last_ts(self) -> int:
        """Latest end timestamp across all dynamic signals."""
        pass

    @property
    @abstractmethod
    def time(self):
        """Episode-wide time accessor.

        - ep.time[ts] -> dict merging static items with sampled values from each signal at-or-before ts.
        - ep.time[start:end] -> Episode view windowed to [start, end).
        - ep.time[start:end:step] -> Episode view sampled at t_i = start + i*step (end-exclusive).
        - ep.time[[t1, t2, ...]] -> Episode view sampled at provided timestamps.
        """
        pass


class EpisodeWriter(AbstractContextManager, ABC, Generic[T]):
    """Abstract base class for writing episodes to a backing store.

    Implementations are context managers. Exiting the context finalizes all
    underlying signal writers and persists static metadata.
    """

    @abstractmethod
    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        """Append dynamic signal data with strictly increasing timestamps per signal.

        Raises if appending to a name that conflicts with an existing static item.
        """
        pass

    @abstractmethod
    def set_static(self, name: str, data: Any) -> None:
        """Set a static (non-time-varying) item for the episode.

        Raises if the name conflicts with an existing dynamic signal or static item.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    @abstractmethod
    def abort(self) -> None:
        """Abort the episode and delete all data. All appends after this point will fail.

        Raises if the episode has already been finalized.
        """
        pass


class DatasetWriter(ABC):
    """Abstract factory for creating new Episodes within a dataset.

    Implementations allocate a new episode slot in the underlying dataset
    (e.g., create a new directory or record) and return an `EpisodeWriter`
    to populate it.
    """

    @abstractmethod
    def new_episode(self) -> EpisodeWriter:
        """Allocate and return a writer for a new episode.

        Returns:
            EpisodeWriter: Context-managed writer used to append dynamic
            signals and set static items.
        """
        pass


class Dataset(ABC, collections.abc.Sequence[Episode]):
    """Ordered collection of Episodes with sequence-style access.

    A Dataset provides read-only, index-based access to episodes. Concrete
    implementations define discovery and storage (e.g., filesystem layout),
    but must provide stable ordering and support integer, slice, and array
    indexing. Slices and index arrays return lists of `Episode` objects.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Episode:
        """Access one or more episodes by position.

        Args:
            index_or_slice: Integer index, slice, or array-like of indices.

        Returns:
            - Episode: when `index_or_slice` is an integer
            - list[Episode]: when a slice or index array is provided

        Raises:
            IndexError: If any index is out of range
            TypeError: If boolean masks are provided or types unsupported
        """
        pass
