from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from collections.abc import Sequence as SequenceABC
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, Union, final, runtime_checkable

import numpy as np

T = TypeVar('T')

IndicesLike: TypeAlias = slice | Sequence[int] | np.ndarray
RealNumericArrayLike: TypeAlias = Sequence[int] | Sequence[float] | np.ndarray


def is_realnum_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def _infer_item_dtype_shape(item: Any) -> tuple[Any, Any]:
    """Infer dtype and shape for a single signal element.

    Rules:
    - numpy.ndarray: dtype = array.dtype, shape = array.shape
    - numeric scalars (Python int/float or numpy integer/floating scalars): dtype = type(item), shape = ()
    - tuple: dtype, shape are tuples of per-element dtype/shape
    - other: dtype = type(item), shape = None
    """
    if isinstance(item, np.ndarray):
        return item.dtype, item.shape
    if isinstance(item, np.integer | np.floating | int | float):
        return type(item), ()
    if isinstance(item, tuple):
        dts_shapes = tuple(_infer_item_dtype_shape(x) for x in item)
        dts = tuple(ds[0] for ds in dts_shapes)
        shapes = tuple(ds[1] for ds in dts_shapes)
        return dts, shapes
    return type(item), None


@runtime_checkable
class TimeIndexerLike(Protocol, Generic[T]):
    def __getitem__(
        self, key: int | float | slice | Sequence[int] | Sequence[float] | np.ndarray
    ) -> Union[tuple[T, int], 'Signal[T]']: ...


class Kind(Enum):
    NUMERIC = 'numeric'
    IMAGE = 'image'


@dataclass
class SignalMeta:
    """Container for signal element metadata."""

    dtype: Any
    shape: Any
    kind: 'Kind' = Kind.NUMERIC


class Signal(Sequence[tuple[T, int]], ABC, Generic[T]):
    """Strictly typed, stepwise function of time with a uniform access contract.

    Implementations must provide only a minimal abstract interface; all indexing
    and time semantics are implemented in this base class.

    The base class assumes a *time-ordered* underlying record stream:
    - Each record is a pair: (value, timestamp_ns).
    - Timestamps are integer nanoseconds and are **monotonically non-decreasing**
      with increasing record index.
    - Random access is supported through the protected methods below; the public
      API (`__getitem__`, `time[...]`, `keys()`, `values()`) is implemented in
      terms of those methods and therefore inherits their correctness
      requirements.
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    # "Protected" API

    @abstractmethod
    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        """Return timestamps for the requested *record indices*.

        This is the timestamp backend used by the base class to implement:
        - `keys()`
        - integer/slice/array indexing (`__getitem__`)
        - time-based indexing (`time[...]`)

        Implementations must follow these rules:
        - **Input**: `indices` is either a `slice`, a sequence/array of ints, or
          a numpy integer array (see `IndicesLike`). Indices are *record indices*
          (0-based), not timestamps.
        - **Output**: a 1D sequence/array of integer timestamps (nanoseconds)
          aligned with the request. For array-like indices, the output length
          must match `len(indices)`; for a slice, it must match the slice span.
        - **Ordering**: when indices are in increasing order, returned
          timestamps must be non-decreasing.
        - **Bounds**: the base class validates bounds for public indexing
          paths; implementations may assume indices are valid.

        Returning a numpy array is recommended for efficiency, but not required.
        """
        raise NotImplementedError

    @abstractmethod
    def _values_at(self, indices: IndicesLike) -> Sequence[T]:
        """Return values for the requested *record indices*.

        This is the value backend used by the base class to implement:
        - `values()`
        - integer/slice/array indexing (`__getitem__`)
        - metadata inference (`meta`) when not overridden

        Implementations must follow these rules:
        - **Input**: same indexing contract as `_ts_at` (`IndicesLike`).
        - **Output**: a sequence of values aligned with the request. For
          array-like indices, the output length must match `len(indices)`; for a
          slice, it must match the slice span.
        - **Stability**: repeated calls with the same indices must return values
          corresponding to the same records (i.e., the signal is conceptually
          immutable for readers).
        - **Bounds**: as with `_ts_at`, public indexing does bounds checks; the
          implementation can assume valid indices.
        """
        raise NotImplementedError

    @abstractmethod
    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        """Map timestamps to record indices by searching in the signal timeline.

        The base class uses this method to implement time-based indexing via
        `signal.time[...]`:
        - For a timestamp `t`, we want the most recent record at or before `t`.

        Implementations must follow these rules:
        - **Input**: `ts_array` is array-like (python sequence or numpy array) of
          real numeric timestamps (ints/floats). Timestamps are in the same
          units as stored keys (nanoseconds).
        - **Output**: an integer index (or an array of integer indices) of the
          same shape/length as the input where each element is:
          - the greatest record index `i` such that `keys()[i] <= t`, or
          - `-1` if `t` precedes the first record.
        - **Monotonicity**: if `ts_array` is non-decreasing, the returned
          indices must be non-decreasing.
        - **Empty signal**: should return `-1` (or an array of `-1`s) for all
          queries.

        A typical implementation is `np.searchsorted(keys, ts_array, "right") - 1`.
        """
        raise NotImplementedError

    # Public API

    @property
    def start_ts(self) -> int:
        if len(self) == 0:
            raise ValueError('Signal is empty')
        return int(np.asarray(self._ts_at([0]))[0])

    @property
    def last_ts(self) -> int:
        if len(self) == 0:
            raise ValueError('Signal is empty')
        return int(np.asarray(self._ts_at([len(self) - 1]))[0])

    @property
    @lru_cache(maxsize=1)
    def meta(self) -> SignalMeta:
        """Metadata accessor for signal elements (dtype, shape).

        Default: infer from first value. Subclasses may override to supply
        domain-specific kind/names or alternate inference.
        """
        if len(self) == 0:
            raise ValueError('Signal is empty')
        dtype, shape = _infer_item_dtype_shape(self._values_at([0])[0])
        # Heuristic: uint8 HxWx3 treated as Image, else Numeric
        kind = (
            Kind.IMAGE
            if (dtype == np.uint8 and isinstance(shape, tuple) and len(shape) == 3 and shape[2] == 3)
            else Kind.NUMERIC
        )
        return SignalMeta(dtype=dtype, shape=shape, kind=kind)

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self):
        return self.meta.shape

    @property
    def kind(self) -> Kind:
        return self.meta.kind

    @final
    @property
    def time(self) -> TimeIndexerLike[T]:
        return _SignalViewTime(self)

    @final
    def values(self) -> Sequence[T]:
        return self._values_at(slice(None))

    @final
    def keys(self) -> Sequence[int]:
        return self._ts_at(slice(None))

    @final
    def __getitem__(self, index_or_slice: int | IndicesLike) -> Union[tuple[T, int], 'Signal[T]']:
        match index_or_slice:
            case int() | np.integer() as idx:
                if idx < 0:
                    idx = len(self) + idx
                if idx < 0 or idx >= len(self):
                    raise IndexError(f'Index {idx} out of range for Signal of length {len(self)}')
                return self._values_at([idx])[0], int(self._ts_at([idx])[0])
            case slice() as sl:
                if sl.step is not None and sl.step <= 0:
                    raise ValueError('Slice step must be positive')
                return _SignalView(self, range(*sl.indices(len(self))))
            case np.ndarray() | SequenceABC() as idxs:
                arr = np.asarray(idxs)
                if arr.size == 0:
                    return _SignalView(self, arr.astype(np.int64))
                if arr.dtype == np.bool_:
                    raise IndexError('Boolean mask indexing is not supported for Signal')
                if not np.issubdtype(arr.dtype, np.integer):
                    raise TypeError(f'Unsupported index dtype: {arr.dtype}')
                arr[arr < 0] += len(self)
                if (arr < 0).any() or (arr >= len(self)).any():
                    raise IndexError('Index out of range for Signal')
                return _SignalView(self, arr)
            case _:
                raise TypeError(f'Unsupported index type: {type(index_or_slice)}')


class _SignalViewTime(TimeIndexerLike[T], Generic[T]):
    def __init__(self, signal: Signal[T]):
        self._signal = signal

    def __getitem__(self, ts_or_array: int | IndicesLike) -> Union[tuple[T, int], 'Signal[T]']:
        match ts_or_array:
            case int() | float() | np.floating() as ts:
                idx = int(self._signal._search_ts([ts])[0])
                if idx < 0:
                    raise KeyError(f'Timestamp {ts} precedes the first record')
                return self._signal._values_at([idx])[0], int(self._signal._ts_at([idx])[0])
            case slice() as sl if sl.step is None:
                if len(self._signal) == 0:
                    return _SignalView(self._signal, range(0, 0))
                start = sl.start if sl.start is not None else self._signal.start_ts
                stop = sl.stop if sl.stop is not None else self._signal.last_ts + 1
                start_id, end_id = self._signal._search_ts([start, stop])
                start_id, end_id = int(start_id), int(end_id)
                if stop > self._signal._ts_at([end_id])[0]:
                    end_id += 1
                kwargs = {}
                if start_id > -1 and self._signal._ts_at([start_id])[0] < start:
                    kwargs['start_ts'] = start
                if start_id < 0:
                    start_id = 0
                return _SignalView(self._signal, range(start_id, end_id, 1), **kwargs)
            case slice() as sl if sl.step is not None and sl.step <= 0:
                raise ValueError('Slice step must be positive')
            case slice() as sl if sl.step is not None and sl.start is None:
                raise ValueError('Slice start is required when step is provided')
            case np.ndarray() | SequenceABC() | slice() as tss:
                if isinstance(tss, slice):
                    if len(self._signal) == 0:
                        return _SignalView(self._signal, range(0, 0))
                    start = tss.start if tss.start is not None else self._signal.start_ts
                    stop = tss.stop if tss.stop is not None else self._signal.last_ts + 1
                    if start < self._signal.start_ts:
                        raise KeyError(f'Timestamp {start} precedes the first record')
                    tss = np.arange(start, stop, tss.step)
                idxs = np.asarray(self._signal._search_ts(tss))
                if (idxs < 0).any():
                    raise KeyError('No record at or before some of the requested timestamps')
                return _SignalView(self._signal, idxs, tss)
            case _:
                raise TypeError(f'Unsupported index type: {type(ts_or_array)}')


class _SignalView(Signal[T], Generic[T]):
    def __init__(
        self,
        signal: Signal[T],
        indices: Sequence[int],
        timestamps: Sequence[int] | None = None,
        start_ts: int | None = None,
    ):
        self._signal = signal
        self._indices = indices
        assert timestamps is None or start_ts is None, 'Only one of timestamps or start_ts can be provided'
        self._timestamps = timestamps
        self._start_ts = start_ts

    @property
    def meta(self) -> SignalMeta:
        if len(self) == 0:
            raise ValueError('Signal is empty')
        return self._signal.meta

    def __len__(self) -> int:
        return len(self._indices)

    def _ts_at(self, indices: IndicesLike | slice) -> Sequence[int] | np.ndarray:
        if self._timestamps is not None:
            return np.asarray(self._timestamps)[indices]
        match indices:
            case slice() | np.ndarray() | SequenceABC() as idxs:
                if isinstance(idxs, slice):
                    mapped = self._indices[idxs]
                else:
                    idxs = np.asarray(idxs)
                    mapped = np.asarray(self._indices)[idxs]
                result = self._signal._ts_at(mapped)
                if self._start_ts is not None:
                    if isinstance(idxs, slice):
                        idxs = np.arange(*idxs.indices(len(self)))
                    result[idxs == 0] = self._start_ts
                return result
            case _:
                raise TypeError(f'Unsupported index type: {type(indices)}')

    def _values_at(self, indices: IndicesLike | slice) -> Sequence[T]:
        match indices:
            case slice() | np.ndarray() | SequenceABC():
                if isinstance(indices, slice):
                    mapped = self._indices[indices]
                else:
                    idxs = np.asarray(indices)
                    mapped = np.asarray(self._indices)[idxs]
                return self._signal._values_at(mapped)
            case _:
                raise TypeError(f'Unsupported index type: {type(indices)}')

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        match ts_array:
            case slice() | np.ndarray() | SequenceABC():
                if self._timestamps is None:
                    parent_idx = self._signal._search_ts(ts_array)
                    return np.searchsorted(self._indices, parent_idx, side='right') - 1
                else:
                    return np.searchsorted(self._timestamps, ts_array, side='right') - 1
            case _:
                raise TypeError(f'Unsupported index type: {type(ts_array)}')


class SignalWriter(AbstractContextManager, ABC, Generic[T]):
    """Append-only writer for Signals."""

    @abstractmethod
    def append(self, data: T, ts_ns: int, extra_ts: dict[str, int] | None = None) -> None:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def abort(self) -> None:
        pass


@runtime_checkable
class SupportsEncodedRepresentation(Protocol):
    """Protocol for signals with a raw/encoded representation distinct from decoded values.

    Signals that use lossy encoding (e.g., video, compressed audio) can implement this
    protocol to expose their raw encoded data for efficient transfer without re-encoding.
    This is modality-agnostic - any signal type with lossy encoding can implement it.
    """

    @property
    def encoding_format(self) -> str:
        """Format identifier for the encoded representation.

        Returns a versioned string like 'positronic.video.v1' that identifies
        both the type of encoding and its wire format version.
        """
        ...

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        """Stream all encoded data as opaque bytes.

        The format of the bytes is defined by `encoding_format`. Receivers must
        parse the stream according to the format version.

        Yields:
            Chunks of raw encoded bytes.
        """
        ...
