from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, Sequence

T = TypeVar('T')


class Signal(ABC, Generic[T]):

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of records in the signal."""
        pass

    @property
    @abstractmethod
    def time(self) -> Sequence[Tuple[T, int]]:
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
        """
        pass

    @abstractmethod
    def __getitem__(self, index_or_slice: int | slice) -> Tuple[T, int] | "Signal[T]":
        """Access the Signal data by index or slice.

        Args:
            index_or_slice: Integer index or slice object

        Returns:
            If index: Tuple of (value, timestamp_ns)
            If slice: Signal[T] view of the original data
        """
        pass


class SignalWriter(ABC, Generic[T]):

    @abstractmethod
    def append(self, data: T, ts_ns: int) -> None:
        """Appends data with timestamp.

        Args:
            data: Data to append (must have consistent shape and dtype)
            ts_ns: Timestamp in nanoseconds (must be strictly increasing)

        Raises:
            RuntimeError: If writer has been finished
            ValueError: If timestamp is not increasing or data shape/dtype doesn't match
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalizes the writing. All following append calls will fail."""
        pass
