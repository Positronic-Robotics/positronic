from typing import TypeVar
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from .core import Signal, SignalWriter

T = TypeVar('T')


class _ArraySignal(Signal[T]):
    """An in-memory, array-backed Signal implementation.

    - Index slicing returns zero-copy views
    - Time window returns zero-copy views
    - Stepped view returns a new array with sampled rows
    """

    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        self._timestamps = timestamps
        self._values = values
        self._time_indexer = _TimeIndexer(self)

    def _load_data(self):
        return None  # Already in-memory

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def start_ts(self) -> int:
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[0])

    @property
    def last_ts(self) -> int:
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[-1])

    @property
    def time(self):
        return self._time_indexer

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, int):
            idx = index_or_slice
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {index_or_slice} out of range")
            return (self._values[idx], self._timestamps[idx])
        elif isinstance(index_or_slice, slice):
            start, stop, step = index_or_slice.indices(len(self))
            if step <= 0:
                raise ValueError("Slice step must be positive")
            return _ArraySignal(self._timestamps[start:stop:step], self._values[start:stop:step])
        elif isinstance(index_or_slice, (list, tuple, np.ndarray)):
            # Support fancy indexing by integer arrays/lists and boolean masks
            idx_array = np.asarray(index_or_slice)

            if idx_array.size == 0:
                return _ArraySignal(self._timestamps[:0], self._values[:0])

            if idx_array.dtype == np.bool_:
                raise IndexError("Boolean indexes are not supported")

            if not np.issubdtype(idx_array.dtype, np.integer):
                raise TypeError(f"Invalid index array dtype: {idx_array.dtype}")

            return _ArraySignal(self._timestamps[idx_array], self._values[idx_array])
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def _window_view(self, start_ts_ns: int, end_ts_ns: int) -> "Signal[T]":
        if len(self) == 0:
            return _ArraySignal(self._timestamps[:0], self._values[:0])

        start_idx = int(np.searchsorted(self._timestamps, start_ts_ns, side='left'))
        end_idx = int(np.searchsorted(self._timestamps, end_ts_ns, side='left'))
        return _ArraySignal(self._timestamps[start_idx:end_idx], self._values[start_idx:end_idx])


class _TimeIndexer:
    """Helper class to implement the time property for Signal."""

    def __init__(self, signal):
        self.signal = signal

    def __getitem__(self, key):  # noqa: C901  Function is too complex
        self.signal._load_data()
        if isinstance(key, int):
            idx = np.searchsorted(self.signal._timestamps, key, side='right') - 1
            if idx < 0:
                raise KeyError(f"No record at or before timestamp {key}")
            return (self.signal._values[idx], self.signal._timestamps[idx])
        elif isinstance(key, (list, tuple, np.ndarray)):  # Sample at arbitrary requested timestamps
            req_ts = np.asarray(key)
            if not np.issubdtype(req_ts.dtype, np.integer):
                raise TypeError(f"Invalid timestamp array dtype: {req_ts.dtype}")

            if req_ts.size == 0 or len(self.signal._timestamps) == 0:
                return _ArraySignal(self.signal._timestamps[:0], self.signal._values[:0])

            # For each requested timestamp t, find index of value at or before t
            pos = np.searchsorted(self.signal._timestamps, req_ts, side='right') - 1
            if not np.all(pos >= 0):
                raise KeyError("No record at or before some of the requested timestamps")

            return _ArraySignal(req_ts, self.signal._values[pos])
        elif isinstance(key, slice):
            if key.step is not None:
                if key.step <= 0:
                    raise ValueError("Slice step must be positive")
                if key.start is None:
                    raise ValueError("Slice start is required when step is provided")

            start = key.start if key.start is not None else self.signal._timestamps[0]
            stop = key.stop if key.stop is not None else (self.signal._timestamps[-1] + 1)
            if key.step is not None:
                return self[np.arange(start, stop, key.step, dtype=np.int64)]
            else:
                return self.signal._window_view(start, stop)
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
        self._time_indexer = _TimeIndexer(self)

    def _load_data(self):
        """Lazily load parquet data into memory as numpy arrays."""
        if self._data is None:
            table = pq.read_table(self.filepath)
            self._timestamps = table['timestamp'].to_numpy()
            self._values = table['value'].to_numpy()
            self._data = table

    def _as_array_signal(self) -> _ArraySignal:
        """Create an ArraySignal view over the loaded numpy arrays (zero-copy)."""
        self._load_data()
        return _ArraySignal(self._timestamps, self._values)

    def __len__(self) -> int:
        """Returns the number of records in the signal."""
        self._load_data()
        return len(self._timestamps)

    @property
    def start_ts(self) -> int:
        """Returns the timestamp of the first record in the signal."""
        self._load_data()
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[0])

    @property
    def last_ts(self) -> int:
        """Returns the timestamp of the last record in the signal."""
        self._load_data()
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[-1])

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return self._time_indexer

    def __getitem__(self, index_or_slice):
        """Access the Signal data by index or slice."""
        return self._as_array_signal()[index_or_slice]

    def _window_view(self, start_ts_ns: int, end_ts_ns: int) -> "Signal[T]":
        """Create a zero-copy Signal view from a time window."""
        return self._as_array_signal()._window_view(start_ts_ns, end_ts_ns)


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
        batch = pa.record_batch([timestamps_array, values_array], names=['timestamp', 'value'])

        if self._writer is None:
            self._writer = pq.ParquetWriter(self.filepath, batch.schema)

        self._writer.write_batch(batch)
        self._timestamps = []
        self._values = []

    def append(self, data: T, ts_ns: int) -> None:  # noqa: C901  Function is too complex
        if self._finished:
            raise RuntimeError("Cannot append to a finished writer")

        if self._last_ts is not None and ts_ns <= self._last_ts:
            raise ValueError(f"Timestamp {ts_ns} is not increasing (last was {self._last_ts})")

        # Normalize the input value without changing the declared generic type T
        value: object = data
        if isinstance(value, pa.Array):  # runtime conversion; keep linter happy via getattr
            value = value.to_numpy()
        elif isinstance(value, (list, tuple)):
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

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize the file on context exit (even on exceptions)."""
        if self._finished:
            return
        self._finished = True
        try:
            self._flush_chunk()  # Flush any remaining data
        finally:
            if self._writer:
                self._writer.close()
            else:
                # No data was ever written, create empty file with default schema
                schema = pa.schema([('timestamp', pa.int64()), ('value', pa.int64())])
                table = pa.table({'timestamp': [], 'value': []}, schema=schema)
                pq.write_table(table, self.filepath)
