from typing import TypeVar, Tuple, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from .core import Signal, SignalWriter

T = TypeVar('T')


class SimpleSignalView(Signal[T]):
    """A zero-copy view of a SimpleSignal with sliced data."""

    def __init__(self, parent_signal, start_idx: int, end_idx: int):
        """Create a view over parent signal's data.

        Args:
            parent_signal: The parent SimpleSignal instance
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
        """
        self._parent = parent_signal
        self._start_idx = start_idx
        self._end_idx = end_idx
        # These are numpy array views (zero-copy)
        self._timestamps = parent_signal._timestamps[start_idx:end_idx]
        self._values = parent_signal._values[start_idx:end_idx]
        self._time_indexer = TimeIndexer(self)

    def __len__(self) -> int:
        """Returns the number of records in the signal view."""
        return self._end_idx - self._start_idx

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return self._time_indexer

    def __getitem__(self, index_or_slice):
        """Access the Signal data by index or slice."""
        if isinstance(index_or_slice, int):
            if index_or_slice < 0:
                index_or_slice = len(self) + index_or_slice
            if index_or_slice < 0 or index_or_slice >= len(self):
                raise IndexError(f"Index {index_or_slice} out of range")
            return (self._values[index_or_slice], self._timestamps[index_or_slice])
        elif isinstance(index_or_slice, slice):
            # Create a view of this view
            start, stop, step = index_or_slice.indices(len(self))
            if step != 1:
                raise ValueError("Step slicing not supported for index-based slicing")
            # Adjust indices relative to parent
            new_start = self._start_idx + start
            new_end = self._start_idx + stop
            return SimpleSignalView(self._parent, new_start, new_end)
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def _load_data(self):
        """Convienice method to simplify the implementation of the viewers.
        """
        pass

    def _window_view(self, start_ts_ns: int, end_ts_ns: int):
        """Create a zero-copy Signal view from a time window."""
        if len(self._timestamps) == 0:
            return SimpleSignalView(self._parent, self._start_idx, self._start_idx)

        start_idx = np.searchsorted(self._timestamps, start_ts_ns, side='left')
        end_idx = np.searchsorted(self._timestamps, end_ts_ns, side='left')  # Note: exclusive end

        # Adjust indices relative to parent
        new_start = self._start_idx + start_idx
        new_end = self._start_idx + end_idx
        return SimpleSignalView(self._parent, new_start, new_end)

    def _stepped_view(self, start_ts_ns: int, end_ts_ns: int, step_ts_ns: int):
        """Create a Signal with values sampled at regular intervals.

        Note: This creates a new array with sampled data, not a zero-copy view,
        since we need to interpolate values at specific timestamps.
        """
        if len(self._timestamps) == 0 or step_ts_ns <= 0:
            return SimpleSignalView(self._parent, self._start_idx, self._start_idx)

        # Find indices for sampled timestamps
        indices = []
        ts = start_ts_ns
        while ts < end_ts_ns:
            if ts >= self._timestamps[0]:
                # TODO: This search can be reduced by starting at the previous index
                idx = np.searchsorted(self._timestamps, ts, side='right') - 1
                if idx >= 0 and idx < len(self._timestamps):
                    # Adjust index relative to parent
                    indices.append(self._start_idx + idx)
            ts += step_ts_ns

        if not indices:
            return SimpleSignalView(self._parent, self._start_idx, self._start_idx)

        # Create a new SimpleSignal with sampled data
        # This is not zero-copy since we're sampling at specific intervals
        sampled = SimpleSignal.__new__(SimpleSignal)
        sampled.filepath = None
        sampled._data = None
        sampled._timestamps = self._parent._timestamps[indices]
        sampled._values = self._parent._values[indices]
        sampled._time_indexer = TimeIndexer(sampled)
        return sampled


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
        self._data = None
        self._timestamps = None
        self._values = None
        self._time_indexer = TimeIndexer(self)

    def _load_data(self):
        """Lazily load parquet data into memory as numpy arrays."""
        if self._data is None:
            table = pq.read_table(self.filepath)
            # Direct conversion from Arrow to NumPy - no Polars needed
            self._timestamps = table['timestamp'].to_numpy()
            self._values = table['value'].to_numpy()
            self._data = table

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
        self._load_data()

        if isinstance(index_or_slice, int):
            if index_or_slice < 0:
                index_or_slice = len(self._timestamps) + index_or_slice
            if index_or_slice < 0 or index_or_slice >= len(self._timestamps):
                raise IndexError(f"Index {index_or_slice} out of range")
            return (self._values[index_or_slice], self._timestamps[index_or_slice])
        elif isinstance(index_or_slice, slice):
            # Create a zero-copy view
            start, stop, step = index_or_slice.indices(len(self._timestamps))
            if step != 1:
                raise ValueError("Step slicing not supported for index-based slicing")
            return SimpleSignalView(self, start, stop)
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def _window_view(self, start_ts_ns: int, end_ts_ns: int):
        """Create a zero-copy Signal view from a time window."""
        self._load_data()

        if len(self._timestamps) == 0:
            return SimpleSignalView(self, 0, 0)

        start_idx = np.searchsorted(self._timestamps, start_ts_ns, side='left')
        end_idx = np.searchsorted(self._timestamps, end_ts_ns, side='left')  # Note: exclusive end

        return SimpleSignalView(self, start_idx, end_idx)

    def _stepped_view(self, start_ts_ns: int, end_ts_ns: int, step_ts_ns: int):
        """Create a Signal with values sampled at regular intervals.

        Note: This creates a new array with sampled data, not a zero-copy view,
        since we need to interpolate values at specific timestamps.
        """
        self._load_data()

        if len(self._timestamps) == 0 or step_ts_ns <= 0:
            return SimpleSignalView(self, 0, 0)

        # Find indices for sampled timestamps
        indices = []
        ts = start_ts_ns
        while ts < end_ts_ns:
            if ts >= self._timestamps[0]:
                idx = np.searchsorted(self._timestamps, ts, side='right') - 1
                if idx >= 0 and idx < len(self._timestamps):
                    indices.append(idx)
            ts += step_ts_ns

        if not indices:
            return SimpleSignalView(self, 0, 0)

        # Create a new SimpleSignal with sampled data
        # This is not zero-copy since we're sampling at specific intervals
        sampled = SimpleSignal.__new__(SimpleSignal)
        sampled.filepath = None
        sampled._data = None
        sampled._timestamps = self._timestamps[indices]
        sampled._values = self._values[indices]
        sampled._time_indexer = TimeIndexer(sampled)
        return sampled



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
        self._timestamps = []
        self._values = []
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

        if isinstance(data, pa.Array):
            data = data.to_numpy()

        if isinstance(data, (list, tuple)):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            if self._expected_shape is None:
                self._expected_shape = data.shape
                self._expected_dtype = data.dtype
            else:
                if data.shape != self._expected_shape:
                    raise ValueError(f"Data shape {data.shape} doesn't match expected shape {self._expected_shape}")
                if data.dtype != self._expected_dtype:
                    raise ValueError(f"Data dtype {data.dtype} doesn't match expected dtype {self._expected_dtype}")
        else:  # Scalar type
            if self._expected_dtype is None:
                self._expected_dtype = type(data)
            else:
                if type(data) != self._expected_dtype:
                    raise ValueError(f"Data type {type(data)} doesn't match expected type {self._expected_dtype}")

        self._timestamps.append(ts_ns)
        self._values.append(data)
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
