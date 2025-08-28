from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .core import IndicesLike, RealNumericArrayLike, Signal, SignalWriter, is_realnum_dtype

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
        self._data: pa.Table | None = None
        # Initialize with empty arrays to satisfy type checkers; real data is loaded lazily
        self._timestamps: np.ndarray = np.empty(0, dtype=np.int64)
        self._values: np.ndarray = np.empty(0, dtype=object)
        # Lazily-loaded columns

    def _load_data(self):
        """Lazily load parquet data into memory as numpy arrays."""
        if self._data is None:
            table = pq.read_table(self.filepath)
            self._timestamps = table['timestamp'].to_numpy()
            self._values = table['value'].to_numpy()
            self._data = table

    def __len__(self) -> int:
        """Returns the number of records in the signal."""
        self._load_data()
        return len(self._timestamps)

    def _ts_at(self, index_or_indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._load_data()
        return self._timestamps[index_or_indices]

    def _values_at(self, index_or_indices: IndicesLike) -> Sequence[T]:
        self._load_data()
        return self._values[index_or_indices]

    def _search_ts(self, ts_or_array: RealNumericArrayLike) -> IndicesLike:
        self._load_data()
        req = np.asarray(ts_or_array)
        if req.size == 0:
            return np.array([], dtype=np.int64)
        if not is_realnum_dtype(req.dtype):
            raise TypeError(f"Invalid timestamp array dtype: {req.dtype}")
        return np.searchsorted(self._timestamps, req, side='right') - 1


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
        self._aborted = False
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
        if self._aborted:
            raise RuntimeError("Cannot append to an aborted writer")

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
        if self._finished or self._aborted:
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

    def abort(self) -> None:
        """Abort writing and remove any partial output file."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError("Cannot abort a finished writer")

        if self._writer is not None:
            self._writer.close()
        self._writer = None

        if self.filepath.exists():
            self.filepath.unlink()

        self._aborted = True
