from typing import TypeVar
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from .core import Signal, SignalWriter

T = TypeVar('T')


class ArraySignal(Signal[T]):
    """An in-memory, array-backed Signal implementation.

    - Index slicing returns zero-copy views
    - Time window returns zero-copy views
    - Stepped view returns a new array with sampled rows
    """

    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        self._timestamps = timestamps
        self._values = values
        self._time_indexer = TimeIndexer(self)

    def _load_data(self):
        # Already in-memory
        return None

    def __len__(self) -> int:
        return int(self._timestamps.shape[0])

    @property
    def time(self):
        return self._time_indexer

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, int):
            length = len(self)
            idx = index_or_slice
            if idx < 0:
                idx = length + idx
            if idx < 0 or idx >= length:
                raise IndexError(f"Index {index_or_slice} out of range")
            return (self._values[idx], self._timestamps[idx])
        elif isinstance(index_or_slice, slice):
            start, stop, step = index_or_slice.indices(len(self))
            if step != 1:
                raise ValueError("Step slicing not supported for index-based slicing")
            return ArraySignal(self._timestamps[start:stop], self._values[start:stop])
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def _window_view(self, start_ts_ns: int, end_ts_ns: int):
        if len(self) == 0:
            return ArraySignal(self._timestamps[:0], self._values[:0])

        start_idx = int(np.searchsorted(self._timestamps, start_ts_ns, side='left'))
        end_idx = int(np.searchsorted(self._timestamps, end_ts_ns, side='left'))
        return ArraySignal(self._timestamps[start_idx:end_idx], self._values[start_idx:end_idx])

    def _stepped_view(self, start_ts_ns: int, end_ts_ns: int, step_ts_ns: int):
        if len(self) == 0 or step_ts_ns <= 0:
            return ArraySignal(self._timestamps[:0], self._values[:0])

        indices: list[int] = []
        ts = int(start_ts_ns)
        while ts < end_ts_ns:
            if ts >= int(self._timestamps[0]):
                idx = int(np.searchsorted(self._timestamps, ts, side='right')) - 1
                if 0 <= idx < len(self):
                    indices.append(idx)
            ts += step_ts_ns

        if not indices:
            return ArraySignal(self._timestamps[:0], self._values[:0])

        idx_array = np.array(indices, dtype=np.int64)
        return ArraySignal(self._timestamps[idx_array], self._values[idx_array])


class TimeIndexer:
    """Helper class to implement the time property for Signal."""

    def __init__(self, signal):
        self.signal = signal

    def __getitem__(self, key):
        if isinstance(key, int):
            # Ensure data is loaded for SimpleSignal instances
            self.signal._load_data()

            if len(self.signal._timestamps) == 0 or key < self.signal._timestamps[0]:
                raise KeyError(f"No record at or before timestamp {key}")

            idx = np.searchsorted(self.signal._timestamps, key, side='right') - 1
            if idx < 0:
                raise KeyError(f"No record at or before timestamp {key}")

            return (self.signal._values[idx], self.signal._timestamps[idx])
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else np.inf
            step = key.step

            if step is None:
                # Regular window query - return a view
                return self.signal._window_view(start, stop)
            else:
                # Stepped query - create sampled view
                return self.signal._stepped_view(start, stop, step)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")


class SimpleSignal(Signal[T]):
    """Parquet-based implementation for scalar and vector Signals.

    Stores data in a parquet file with 'timestamp' and 'value' columns.
    Provides O(log N) random access using binary search operations.
    Data is lazily loaded into memory and kept as pyarrow arrays.
    """

    def __init__(self, filepath: Path):
        """Initialize Signal reader from a parquet file."""
        self.filepath = filepath
        self._data: pa.Table | None = None
        # Initialize with empty arrays to satisfy type checkers; real data is loaded lazily
        self._timestamps: np.ndarray = np.empty(0, dtype=np.int64)
        self._values: np.ndarray = np.empty(0, dtype=object)
        self._time_indexer = TimeIndexer(self)

    def _load_data(self):
        """Lazily load parquet data into memory as numpy arrays."""
        if self._data is None:
            table = pq.read_table(self.filepath)
            self._timestamps = table['timestamp'].to_numpy()
            self._values = table['value'].to_numpy()
            self._data = table

    def _as_array_signal(self) -> ArraySignal:
        """Create an ArraySignal view over the loaded numpy arrays (zero-copy)."""
        self._load_data()
        return ArraySignal(self._timestamps, self._values)

    def __len__(self) -> int:
        """Returns the number of records in the signal."""
        self._load_data()
        return len(self._timestamps)

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return self._time_indexer

    def __getitem__(self, index_or_slice):
        """Access the Signal data by index or slice."""
        return self._as_array_signal()[index_or_slice]

    def _window_view(self, start_ts_ns: int, end_ts_ns: int):
        """Create a zero-copy Signal view from a time window."""
        return self._as_array_signal()._window_view(start_ts_ns, end_ts_ns)

    def _stepped_view(self, start_ts_ns: int, end_ts_ns: int, step_ts_ns: int):
        """Create a Signal with values sampled at regular intervals."""
        return self._as_array_signal()._stepped_view(start_ts_ns, end_ts_ns, step_ts_ns)



class SimpleSignalWriter(SignalWriter[T]):
    """Parquet-based writer for scalar and vector Signals.

    Writes data in chunks to parquet file for memory efficiency.
    Enforces consistent shape/dtype and strictly increasing timestamps.
    Supports scalars and fixed-size vectors/arrays.
    """

    def __init__(self, filepath: Path, chunk_size: int = 10000):
        """Initialize Signal writer to save data to a parquet file.

        Args:
            filepath: Path to the output parquet file
            chunk_size: Number of records to accumulate before writing a chunk (default 10000)
        """
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._writer = None
        self._timestamps: list[int] = []
        self._values: list[object] = []
        self._finished = False
        self._last_ts = None
        self._expected_shape = None
        self._expected_dtype = None

    def _flush_chunk(self):
        """Write current chunk to parquet file."""
        if len(self._timestamps) == 0:
            return

        timestamps_array = pa.array(self._timestamps, type=pa.int64())
        values_array = pa.array(self._values)
        batch = pa.record_batch([timestamps_array, values_array],
                                names=['timestamp', 'value'])

        if self._writer is None:
            self._writer = pq.ParquetWriter(self.filepath, batch.schema)

        self._writer.write_batch(batch)
        self._timestamps = []
        self._values = []

    def append(self, data: T, ts_ns: int) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished writer")

        if self._last_ts is not None and ts_ns <= self._last_ts:
            raise ValueError(f"Timestamp {ts_ns} is not increasing (last was {self._last_ts})")

        # Normalize the input value without changing the declared generic type T
        value: object = data
        if isinstance(value, pa.Array):  # runtime conversion; keep linter happy via getattr
            to_numpy = getattr(value, "to_numpy", None)
            if callable(to_numpy):
                value = to_numpy()

        if isinstance(value, (list, tuple)):
            value = np.array(value)

        if isinstance(value, np.ndarray):
            if self._expected_shape is None:
                self._expected_shape = value.shape
                self._expected_dtype = value.dtype
            else:
                if value.shape != self._expected_shape:
                    raise ValueError(f"Data shape {value.shape} doesn't match expected shape {self._expected_shape}")
                if value.dtype != self._expected_dtype:
                    raise ValueError(f"Data dtype {value.dtype} doesn't match expected dtype {self._expected_dtype}")
        else:  # Scalar type
            if self._expected_dtype is None:
                self._expected_dtype = type(value)
            else:
                if type(value) is not self._expected_dtype:
                    raise ValueError(f"Data type {type(value)} doesn't match expected type {self._expected_dtype}")

        self._timestamps.append(int(ts_ns))
        self._values.append(value)
        self._last_ts = ts_ns

        if len(self._timestamps) >= self.chunk_size:
            self._flush_chunk()

    def finish(self) -> None:
        """Write remaining data to parquet file and mark writer as finished."""
        if self._finished:
            return

        self._finished = True

        self._flush_chunk()  # Flush any remaining data

        if self._writer:
            self._writer.close()
        elif len(self._timestamps) == 0 and self._writer is None:
            # No data was ever written, create empty file with default schema
            schema = pa.schema([('timestamp', pa.int64()), ('value', pa.int64())])
            table = pa.table({'timestamp': [], 'value': []}, schema=schema)
            pq.write_table(table, self.filepath)
