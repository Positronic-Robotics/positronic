from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, Sequence

T = TypeVar('T')


class Stream(ABC, Generic[T]):
    @abstractmethod
    def at(self, ts_ns: int) -> Tuple[T, int] | None:
        """Returns the value and timestamp of the closest record at or before the given timestamp.
        
        Args:
            ts_ns: Timestamp in nanoseconds
            
        Returns:
            Tuple of (value, timestamp_ns) or None if not found
        """
        pass
    
    @abstractmethod
    def window(self, start_ts_ns: int, end_ts_ns: int) -> Tuple[Sequence[T], Sequence[int]]:
        """Returns all records in the inclusive range [start_ts_ns, end_ts_ns].
        
        Args:
            start_ts_ns: Start timestamp in nanoseconds
            end_ts_ns: End timestamp in nanoseconds
            
        Returns:
            Tuple of (values, timestamps_ns) sequences
        """
        pass


class StreamWriter(ABC, Generic[T]):
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