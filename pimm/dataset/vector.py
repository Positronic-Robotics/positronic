from typing import TypeVar, Tuple, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import numpy as np
from pathlib import Path
from .core import Signal, SignalWriter

T = TypeVar('T')


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

    def _load_data(self):
        """Lazily load parquet data into memory as numpy arrays."""
        if self._data is None:
            table = pq.read_table(self.filepath)
            df = pl.from_arrow(table)
            self._timestamps = df['timestamp'].to_numpy()
            self._values = df['value'].to_numpy()
            self._data = df

    def at(self, ts_ns: int) -> Tuple[T, int] | None:
        self._load_data()

        if len(self._timestamps) == 0:
            return None

        if ts_ns < self._timestamps[0]:
            return None

        idx = np.searchsorted(self._timestamps, ts_ns, side='right') - 1

        if idx < 0:
            return None

        return (self._values[idx], self._timestamps[idx])

    def window(self, start_ts_ns: int, end_ts_ns: int) -> Tuple[Sequence[T], Sequence[int]]:
        self._load_data()

        if len(self._timestamps) == 0:
            return ([], [])

        start_idx = np.searchsorted(self._timestamps, start_ts_ns, side='left')
        end_idx = np.searchsorted(self._timestamps, end_ts_ns, side='right')

        return (self._values[start_idx:end_idx], self._timestamps[start_idx:end_idx])


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
