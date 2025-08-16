from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
import collections.abc
from typing import Any, Iterator, TypeVar, Generic, Tuple, Sequence, Protocol, runtime_checkable
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
        """Returns the number of records in the signal."""
        pass

    @property
    @abstractmethod
    def start_ts(self) -> int:
        """Returns the timestamp of the first record in the signal."""
        pass

    @property
    @abstractmethod
    def last_ts(self) -> int:
        """Returns the timestamp of the last record in the signal."""
        pass

    @property
    @abstractmethod
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
        pass

    @abstractmethod
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Tuple[T, int] | "Signal[T]":
        """Access the Signal data by index or slice.

        Args:
            index_or_slice: Integer index, slice object, or array-like of indices

        Returns:
            If index: Tuple of (value, timestamp_ns)
            If slice/array-like: Signal[T] view of the original data
        """
        pass


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


class DatasetWriter(ABC):
    @abstractmethod
    def new_episode(self, **metadata: dict[str, Any]) -> EpisodeWriter:
        pass


class Dataset(ABC, collections.abc.Sequence[Episode]):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Episode:
        pass
