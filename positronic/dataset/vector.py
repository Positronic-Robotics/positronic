from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .signal import RECORDED_TIME, TIMELINE_METADATA_KEY, IndicesLike, Signal, SignalWriter
from .time import Time, TimeArray, search_timestamps, validate_timeline

T = TypeVar('T')


SIGNAL_VERSION_KEY = b'positronic.signal_version'
SIGNAL_VERSION = b'2'
TIMESTAMP_PREFIX = 'ts.'


def timestamp_table(columns: Mapping[str, Sequence[int]]) -> pa.Table:
    fields = [pa.field(TIMESTAMP_PREFIX + name, pa.int64(), nullable=False) for name in columns]
    schema = pa.schema(fields, metadata={SIGNAL_VERSION_KEY: SIGNAL_VERSION})
    return pa.table({TIMESTAMP_PREFIX + name: values for name, values in columns.items()}, schema=schema)


class ParquetTimeIndex:
    """Named timestamp columns and footer bounds for native and legacy signal files."""

    def __init__(self, path: Path, legacy_column: str):
        self._path = path
        self._legacy_column = legacy_column
        self._loaded: dict[str, np.ndarray] = {}

    @cached_property
    def _file(self) -> pq.ParquetFile:
        return pq.ParquetFile(self._path)

    @cached_property
    def _columns(self) -> dict[str, str]:
        schema = self._file.schema_arrow
        metadata = schema.metadata or {}
        version = metadata.get(SIGNAL_VERSION_KEY)
        if version is None:
            name = metadata.get(TIMELINE_METADATA_KEY, RECORDED_TIME.encode()).decode()
            validate_timeline(name)
            if self._legacy_column not in schema.names:
                raise ValueError(f'Missing legacy timestamp column {self._legacy_column!r}')
            return {name: self._legacy_column}
        if version != SIGNAL_VERSION:
            raise ValueError(f'Unsupported signal format version: {version!r}')
        columns = {}
        for field in schema:
            if field.name.startswith(TIMESTAMP_PREFIX):
                name = field.name[len(TIMESTAMP_PREFIX) :]
                validate_timeline(name)
                if field.type != pa.int64() or field.nullable:
                    raise ValueError('Timeline columns must be non-null int64')
                if name in columns:
                    raise ValueError(f'Duplicate timeline: {name!r}')
                columns[name] = field.name
        if not columns and len(self):
            raise ValueError('A nonempty signal must declare timelines')
        return columns

    @property
    def timelines(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def __len__(self) -> int:
        return self._file.metadata.num_rows

    def _load(self, timelines: tuple[str, ...]) -> None:
        names = [self._columns[name] for name in timelines if name not in self._loaded]
        if names:
            table = self._file.read(columns=names)
            for name in timelines:
                if name not in self._loaded:
                    column = table[self._columns[name]]
                    if column.null_count:
                        raise ValueError(f'Null coordinate on timeline {name!r}')
                    values = column.to_numpy()
                    if np.any(values[1:] < values[:-1]):
                        raise ValueError(f'Timeline {name!r} is not non-decreasing')
                    self._loaded[name] = values

    def read(self, indices: IndicesLike, timelines: tuple[str, ...]) -> TimeArray:
        self._load(timelines)
        if not isinstance(indices, slice):
            indices = np.asarray(indices, dtype=np.int64)
        values = np.column_stack([self._loaded[name][indices] for name in timelines])
        return TimeArray(timelines, values)

    def search(self, queries: Sequence[Time]) -> np.ndarray:
        if not len(queries):
            return np.empty(0, dtype=np.int64)
        self._load(queries[0].timelines)
        return search_timestamps(self._loaded, queries)

    def bounds(self, timelines: tuple[str, ...]) -> tuple[Time, Time]:
        if not len(self):
            raise ValueError('Signal is empty')
        metadata = self._file.metadata
        groups = [metadata.row_group(i) for i in range(metadata.num_row_groups) if metadata.row_group(i).num_rows]
        first, last = {}, {}
        for name in timelines:
            column_name = self._columns[name]
            column_index = next(
                i for i in range(groups[0].num_columns) if groups[0].column(i).path_in_schema == column_name
            )
            low = groups[0].column(column_index).statistics
            high = groups[-1].column(column_index).statistics
            if low is not None and high is not None and low.has_min_max and high.has_min_max:
                first[name], last[name] = int(low.min), int(high.max)
            else:
                # Parquet statistics are optional; read only this coordinate when absent.
                ends = self.read([0, len(self) - 1], (name,))
                first[name], last[name] = ends[0][name], ends[1][name]
        return Time(**first), Time(**last)


class SimpleSignal(Signal[T]):
    """Scalar/vector Parquet signal with independent lazy timestamp and value reads."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._time_index = ParquetTimeIndex(filepath, 'timestamp')
        self._values: np.ndarray | None = None

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._time_index.timelines

    def __len__(self) -> int:
        return len(self._time_index)

    def bounds(self, timelines: tuple[str, ...]) -> tuple[Time, Time]:
        if not len(self):
            raise ValueError('Signal is empty')
        self._validate_selection(timelines)
        return self._time_index.bounds(timelines)

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._time_index.read(indices, timelines)

    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        if self._values is None:
            values = pq.read_table(self.filepath, columns=['value'])['value'].to_numpy()
            if values.dtype == object and len(values) and isinstance(values[0], np.ndarray):
                values = np.stack(values)
            self._values = values
        return cast(Sequence[T], self._values[indices])

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        return self._time_index.search(queries)


class SimpleSignalWriter(SignalWriter[T]):
    """Parquet-based writer for scalar and vector Signals.

    Writes data in chunks to parquet file for memory efficiency.
    Enforces consistent shape/dtype and strictly increasing timestamps.
    Supports scalars and fixed-size vectors/arrays.
    """

    def __init__(self, filepath: Path, chunk_size: int = 10000, drop_equal_bytes_threshold: int | None = None):
        """Initialize Signal writer to save data to a parquet file.

        Args:
            filepath: Path to the output parquet file
            chunk_size: Number of records to accumulate before writing a chunk (default 10000)
            drop_equal_bytes_threshold: If set, and the first record's byte-size is below this
                threshold, subsequent appends will drop values equal to the last written value.
        """
        super().__init__()
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._drop_equal_bytes_threshold = drop_equal_bytes_threshold
        self._writer = None
        self._timestamps: dict[str, list[int]] = {}
        self._values: list[object] = []
        self._finished = False
        self._aborted = False
        self._expected_shape = None
        self._expected_dtype = None
        self._dedupe_enabled = False
        self._last_value: Any | None = None

    @staticmethod
    def _equal(a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        return a == b

    @staticmethod
    def _nbytes(v: Any) -> int:
        if isinstance(v, np.ndarray):
            return int(v.nbytes)
        if isinstance(v, bytes | bytearray):
            return len(v)
        return int(np.array(v).nbytes)

    def _flush_chunk(self):
        if not self._values:
            return
        table = timestamp_table(self._timestamps).append_column('value', pa.array(self._values))
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.filepath, table.schema)
        self._writer.write_table(table)
        self._values.clear()
        for values in self._timestamps.values():
            values.clear()

    def _normalize_value(self, data: T) -> object:
        value: Any = data
        if isinstance(value, pa.Array):
            value = value.to_numpy()
        elif isinstance(value, list | tuple):
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
        else:
            if self._expected_dtype is None:
                self._expected_dtype = type(value)
            else:
                if type(value) is not self._expected_dtype:
                    raise ValueError(f"Data type {type(value)} doesn't match expected type {self._expected_dtype}")
        return value

    def append(self, data: T, timestamps: Time) -> None:
        if self._finished:
            raise RuntimeError('Cannot append to a finished writer')
        if self._aborted:
            raise RuntimeError('Cannot append to an aborted writer')

        self._validate_timestamps(timestamps)
        value = self._normalize_value(data)

        if self._last_time is None and self._drop_equal_bytes_threshold is not None:
            size_bytes = self._nbytes(value)
            if size_bytes < self._drop_equal_bytes_threshold:
                self._dedupe_enabled = True

        self._last_time = timestamps
        if not self._timestamps:
            self._timestamps = {name: [] for name in timestamps}

        if self._dedupe_enabled and self._last_value is not None and self._equal(value, self._last_value):
            return

        self._values.append(value)
        for name, coordinate in timestamps.items():
            self._timestamps[name].append(coordinate)

        self._last_value = value

        if len(self._values) >= self.chunk_size:
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
                table = timestamp_table(self._timestamps).append_column('value', pa.array([], type=pa.int64()))
                pq.write_table(table, self.filepath)

    def abort(self) -> None:
        """Abort writing and remove any partial output file."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError('Cannot abort a finished writer')

        if self._writer is not None:
            self._writer.close()
        self._writer = None

        if self.filepath.exists():
            self.filepath.unlink()

        self._aborted = True
