from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, partial
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike

from positronic import geom
from positronic.utils.lazy import LazySequence, lazy_sequence

from ..signal import IndicesLike, Signal
from ..time import Time, TimeArray, as_time, validate_timelines

T = TypeVar('T')
U = TypeVar('U')


def _as_indices(indices: IndicesLike, n: int) -> np.ndarray:
    """Normalize IndicesLike to a concrete int64 array (handles slice)."""
    if isinstance(indices, slice):
        return np.arange(*indices.indices(n), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


RotRep = geom.Rotation.Representation
NpSignal = Signal[np.ndarray]


class Elementwise(Signal[U]):
    """Element-wise value transform view over a Signal.

    Wraps another `Signal[T]` and applies a function `f` to its values while
    preserving timestamps and ordering. Length and time indexing semantics are
    identical to the underlying signal.

    """

    def __init__(self, signal: Signal[T], fn: Callable[[Sequence[T]], Sequence[U] | np.ndarray]):
        self._signal = signal
        self._fn = fn

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._signal.timelines

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._signal._ts_at(indices, timelines)

    def _values_at(self, indices: IndicesLike) -> Sequence[U]:
        return cast(Sequence[U], self._fn(self._signal._values_at(indices)))

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        return self._signal._search_ts(queries)

    @staticmethod
    def _best_fn_name(fn: Callable[..., Any]) -> str:
        base = fn
        if isinstance(base, partial):
            base = base.func
        name = getattr(base, '__name__', None)
        if name is not None:
            return 'lambda' if name == '<lambda>' else name
        # Fallback to class name for callables
        cls = getattr(base, '__class__', None)
        if cls is not None and hasattr(cls, '__name__'):
            return cls.__name__
        return 'fn'


class IndexOffsets(Signal[tuple]):
    """Join values (and optionally timestamps) at relative index offsets.

    For a base signal ``s`` and relative offsets ``D = [d1, d2, ..., dN]`` (each
    may be negative or positive), this view iterates over base indices ``i`` for
    which all ``i+dk`` are in-bounds. For each valid ``i`` the element is:

    - If ``include_ref_ts=False`` (default):
        ((v[i+d1], ..., v[i+dN]), t[i])
    - If ``include_ref_ts=True`` and ``N == 1`` (single offset):
        ((v[i+d1], t[i+d1]), t[i])
    - If ``include_ref_ts=True`` and ``N > 1``:
        (((v[i+d1], ..., v[i+dN]), (t[i+d1], ..., t[i+dN])), t[i])

    Notes:
      - This class does not modify values; it only aligns and groups neighbors.
    """

    def __init__(self, signal: Signal[T], *relative_indices: int, include_ref_ts: bool = False) -> None:
        self._signal = signal
        if len(relative_indices) == 0:
            raise ValueError('relative_indices must be non-empty')
        offs = np.asarray(relative_indices, dtype=np.int64)
        if offs.size == 0:
            raise ValueError('relative_indices must be non-empty')
        self._offs = offs
        self._min_off = int(np.min(self._offs))
        self._max_off = int(np.max(self._offs))
        self._include_ref_ts = bool(include_ref_ts)

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._signal.timelines

    def __len__(self) -> int:
        n = len(self._signal)
        start_trim = max(0, -self._min_off)
        end_trim = max(0, self._max_off)
        return max(0, n - start_trim - end_trim)

    def _base_start(self) -> int:
        return max(0, -self._min_off)

    def _base_last(self) -> int:
        n = len(self._signal)
        return n - 1 - max(0, self._max_off)

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        base = _as_indices(indices, len(self)) + self._base_start()
        return self._signal._ts_at(base, timelines)

    def _values_at(self, indices: IndicesLike):
        base = _as_indices(indices, len(self)) + self._base_start()
        vals_parts = []
        ts_parts = []
        for off in self._offs:
            idxs = base + int(off)
            vals_parts.append(self._signal._values_at(idxs))
            if self._include_ref_ts:
                ts_parts.append(self._signal._ts_at(idxs, self.timelines))

        n = len(self._offs)
        if not self._include_ref_ts:
            return list(zip(*vals_parts, strict=False)) if n > 1 else vals_parts[0]
        else:
            if n == 1:
                return list(zip(vals_parts[0], ts_parts[0], strict=False))

            ts = list(zip(*ts_parts, strict=True))
            out = [(tuple(parts[i] for parts in vals_parts), ts[i]) for i in range(len(base))]
            return out

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        # Map parent floor indices to view indices, clamping to valid range.
        n = len(self)
        t = queries
        if n == 0:
            return np.full(len(t), -1, dtype=np.int64)  # nothing valid in this view
        p = np.asarray(self._signal._search_ts(t))
        base_start = self._base_start()
        base_last = self._base_last()
        view_idx = p - base_start
        view_idx[p < base_start] = -1
        view_idx[p > base_last] = n - 1
        return view_idx


class TimeOffsets(Signal[tuple]):
    """Sample on named offsets, retaining the base record's full timestamps.

    A single offset yields its value directly; several yield a tuple of values.
    With include_ref_ts, pair those values with their original Time values.
    """

    def __init__(self, signal: Signal[T], *offsets: Time | Mapping[str, int], include_ref_ts: bool = False):
        if not offsets:
            raise ValueError('TimeOffsets requires at least one offset')
        self._offsets = tuple(as_time(offset) for offset in offsets)
        self._names = self._offsets[0].timelines
        signal._validate_selection(self._names)
        for offset in self._offsets[1:]:
            self._offsets[0]._validate_timelines(offset)
        self._signal = signal
        self._include_ref_ts = include_ref_ts

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._signal.timelines

    @cached_property
    def _start(self) -> int:
        if not len(self._signal):
            return 0
        first = self._signal.bounds(self._names).start
        threshold = Time(**{
            name: first[name] - min(0, *(offset[name] for offset in self._offsets)) for name in self._names
        })
        lo, hi = 0, len(self._signal)
        while lo < hi:
            mid = (lo + hi) // 2
            if threshold <= self._signal._ts_at([mid], self._names)[0]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __len__(self) -> int:
        return len(self._signal) - self._start

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._signal._ts_at(_as_indices(indices, len(self)) + self._start, timelines)

    def _values_at(self, indices: IndicesLike):
        base = _as_indices(indices, len(self)) + self._start
        times = self._signal._ts_at(base, self._names)
        parts, references = [], []
        for offset in self._offsets:
            selected = self._signal._search_ts(LazySequence(times, lambda time, offset=offset: time + offset))
            parts.append(self._signal._values_at(selected))
            if self._include_ref_ts:
                references.append(self._signal._ts_at(selected, self.timelines))
        if len(parts) == 1:
            return list(zip(parts[0], references[0], strict=True)) if self._include_ref_ts else parts[0]
        values = list(zip(*parts, strict=True))
        return list(zip(values, zip(*references, strict=True), strict=True)) if self._include_ref_ts else values

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        return np.maximum(-1, np.asarray(self._signal._search_ts(queries)) - self._start)


class Join(Signal[tuple]):
    """Carry input values over their union on an explicit common timeline subset.

    Input coordinates are clamped to the coordinatewise maximum of input starts.
    Equal projected coordinates collapse to one record. Incomparable coordinates
    after the start reject the join. Reference timestamps retain all source axes.
    """

    def __init__(self, *signals: Signal[Any], timelines: tuple[str, ...], include_ref_ts: bool = False):
        if not signals:
            raise ValueError('Join requires at least one signal')
        validate_timelines(timelines)
        for signal in signals:
            signal._validate_selection(timelines)
        self._signals = signals
        self._timelines = timelines
        self._include_ref_ts = include_ref_ts

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._timelines

    @cached_property
    def _times(self) -> TimeArray:
        if any(not len(signal) for signal in self._signals):
            return TimeArray(self.timelines, np.empty((0, len(self.timelines)), dtype=np.int64))
        starts = [signal.bounds(self.timelines).start for signal in self._signals]
        start = np.array([max(ts[name] for ts in starts) for name in self.timelines], dtype=np.int64)
        rows = np.concatenate([
            TimeArray.from_times(signal.timestamps(self.timelines), self.timelines)._values for signal in self._signals
        ])
        # Lexicographic sorting is only a candidate order; every coordinate must agree with it.
        rows = np.unique(np.maximum(rows, start), axis=0)
        times = TimeArray(self.timelines, rows)
        times.validate_order()
        return times

    def __len__(self) -> int:
        return len(self._times)

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._times.select(timelines).take(indices)

    def _values_at(self, indices: IndicesLike):
        times = self._times.take(indices)
        indices_per_signal = [signal._search_ts(times) for signal in self._signals]
        parts = [
            signal._values_at(selected) for signal, selected in zip(self._signals, indices_per_signal, strict=True)
        ]
        values = list(zip(*parts, strict=True))
        if not self._include_ref_ts:
            return values
        references = [
            signal._ts_at(selected, signal.timelines)
            for signal, selected in zip(self._signals, indices_per_signal, strict=True)
        ]
        return list(zip(values, zip(*references, strict=True), strict=True))

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        return self._times.search(queries)


def _concat_per_frame(dtype: np.dtype | None, x: Sequence[tuple]) -> np.ndarray:
    """Pickable callable that concatenates multiple array signals into a single array signal."""
    # x is a sequence of tuples (v1, v2, ..., vN) for the requested indices.
    # High-performance path: preallocate (batch, total_dim) and fill via slicing.
    batch = len(x)
    if batch == 0:
        return np.empty((0, 0), dtype=dtype or np.float32)

    # Infer per-signal dimensions and dtype from the first row
    first_parts = [np.asarray(v) for v in x[0]]
    dims = [p.size for p in first_parts]
    offsets = np.cumsum([0] + dims[:-1]) if dims else [0]
    total_dim = int(sum(dims))
    if dtype is None:
        dtype = np.result_type(*[p.dtype for p in first_parts]) if first_parts else np.dtype(np.float32)
    out = np.empty((batch, total_dim), dtype=dtype)

    for j, p in enumerate(first_parts):  # Fill first row
        out[0, offsets[j] : offsets[j] + dims[j]] = p.ravel().astype(dtype, copy=False)
    for i in range(1, batch):  # Fill remaining rows
        row = x[i]
        for j, v in enumerate(row):
            arr = np.asarray(v)
            if arr.size != dims[j]:
                raise ValueError('concat: inconsistent vector size across rows')
            out[i, offsets[j] : offsets[j] + dims[j]] = arr.ravel().astype(dtype, copy=False)
    return out


def concat(*signals, timelines: tuple[str, ...], dtype: DTypeLike | None = None) -> NpSignal:
    """Concatenate multiple 1D array signals into a single array signal.

    - Aligns signals on the union of timestamps with carry-back semantics.
    - Values are vector-wise concatenations of each signal's values at-or-before t.
    - For batched requests, returns a single 2D array (batch, dim).
    """
    n = len(signals)
    if n == 0:
        raise ValueError('concat requires at least one key')
    return Elementwise(
        Join(*signals, timelines=timelines), partial(_concat_per_frame, np.dtype(dtype) if dtype is not None else None)
    )


def _astype_per_frame(dtype: np.dtype, x: Sequence[np.ndarray]) -> np.ndarray:
    """Pickable callable that casts arrays to a target dtype."""
    arr = np.asarray(x)
    if arr.dtype == dtype:
        return arr
    return arr.astype(dtype, copy=False)


def astype(signal: NpSignal, dtype: np.dtype) -> NpSignal:
    """Return a Signal view that casts batched values to a given dtype."""
    return Elementwise(signal, partial(_astype_per_frame, dtype))


def view(signal: NpSignal, slice: slice) -> NpSignal:
    def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return LazySequence(x, lambda v: v[slice])

    return Elementwise(signal, fn)


def diff(signal: NpSignal, dt_sec: float, order: int = 1, *, timelines: tuple[str, ...]) -> NpSignal:
    """Centered finite-difference derivative of a vector signal.

    Args:
        signal: Input signal with ndarray values of shape (dim,).
        dt_sec: Time window in seconds for the finite difference stencil.
        order: Derivative order. 1 = velocity, 2 = acceleration.
        timelines: Physical time axes, each measured in nanoseconds.

    Returns:
        Signal of per-frame derivative vectors (same dim as input).
        order=1: (f(t+dt) - f(t-dt)) / 2dt
        order=2: (f(t-dt) - 2f(t) + f(t+dt)) / dt²
    """
    if order < 1 or order > 2:
        raise ValueError(f'diff supports order 1 or 2, got {order}')
    validate_timelines(timelines)
    dt_ns = int(dt_sec * 1e9)

    if order == 1:

        def velocity(pairs):
            arr = np.array(pairs)  # (batch, 2, dim)
            return (arr[:, 1] - arr[:, 0]) / (2 * dt_sec)

        return Elementwise(
            TimeOffsets(signal, Time(**dict.fromkeys(timelines, -dt_ns)), Time(**dict.fromkeys(timelines, dt_ns))),
            velocity,
        )
    else:

        def acceleration(triples):
            arr = np.array(triples)  # (batch, 3, dim)
            return (arr[:, 2] - 2 * arr[:, 1] + arr[:, 0]) / (dt_sec * dt_sec)

        return Elementwise(
            TimeOffsets(
                signal,
                Time(**dict.fromkeys(timelines, -dt_ns)),
                Time(**dict.fromkeys(timelines, 0)),
                Time(**dict.fromkeys(timelines, dt_ns)),
            ),
            acceleration,
        )


def norm(signal: NpSignal) -> NpSignal:
    """Per-frame L2 norm. Signal[ndarray(dim,)] → Signal[scalar]."""

    def fn(vals):
        arr = np.array(vals)
        if arr.ndim == 1:
            return np.abs(arr)
        return np.linalg.norm(arr, axis=-1)

    return Elementwise(signal, fn)


# ---------------------------------------------------------------------------
# Scalar aggregators — reduce a Signal to a single value
# ---------------------------------------------------------------------------


def _signal_values(signal: Signal) -> np.ndarray:
    """Extract all values from a Signal as a numpy array."""
    n = len(signal)
    if n == 0:
        return np.array([])
    return np.array(signal._values_at(np.arange(n)))


def agg_max(signal: Signal) -> float:
    """Maximum value across all frames."""
    return float(np.max(_signal_values(signal)))


def agg_mean(signal: Signal) -> float:
    """Mean value across all frames."""
    return float(np.mean(_signal_values(signal)))


def agg_percentile(signal: Signal, q: float) -> float:
    """q-th percentile across all frames (q in 0..100)."""
    return float(np.percentile(_signal_values(signal), q))


def agg_fraction_true(signal: Signal) -> float:
    """For boolean signals: fraction of True values."""
    return float(np.mean(_signal_values(signal)))


class _PairwiseMap:
    def __init__(self, op: Callable[[Any, Any], Any]):
        self._op = op

    def __call__(self, rows: Sequence[tuple]) -> Sequence[Any]:
        out: list[Any] = []
        for a, b in rows:
            out.append(self._op(a, b))
        return out


def pairwise(
    a: Signal[Any], b: Signal[Any], op: Callable[[Any, Any], Any], *, timelines: tuple[str, ...]
) -> Signal[Any]:
    """Apply a binary operation pairwise across two signals aligned on time.

    - Aligns `a` and `b` on the union of timestamps with carry-back semantics.
    - Applies `op(a_value, b_value)` per row and returns a new Signal view of results.
    """
    return Elementwise(Join(a, b, timelines=timelines), _PairwiseMap(op))


def recode_transform(rep_from: RotRep, rep_to: RotRep, signal: NpSignal) -> NpSignal:
    """Return a Signal view with SE(3) vectors recoded to a new rotation representation.

    The input signal must yield vectors produced by ``Transform3D.as_vector``
    that concatenate a translation with a rotation encoded using ``rep_from``.
    Values are lazily converted so that each frame's rotation is expressed in
    ``rep_to`` while translations and timestamps remain untouched.

    Args:
        rep_from: Rotation representation used by the input signal values.
        rep_to: Desired rotation representation for the output signal.
        signal: Source signal providing Transform3D vectors encoded with
            ``rep_from``.
    """
    if rep_from == rep_to:
        return signal

    @lazy_sequence
    def decode(x: np.ndarray) -> np.ndarray:
        return geom.Transform3D.from_vector(x, rep_from).as_vector(rep_to)

    return Elementwise(signal, decode)


def recode_rotation(rep_from: RotRep, rep_to: RotRep, signal: NpSignal, slice: slice | None = None) -> NpSignal:
    """Return a Signal view with rotation vectors recoded to a different representation.

    Args:
        rep_from: Input rotation representation.
        rep_to: Output rotation representation.
        signal: Input Signal with frames shaped (dim,), where dim depends on rep_from.
        slice: Optional slice to select a subset of the input frame before conversion.
    """
    if rep_from == rep_to and slice is None:
        return signal

    @lazy_sequence
    def decode(x: np.ndarray) -> np.ndarray:
        if slice is not None:
            x = x[slice]
        return geom.Rotation.create_from(x, rep_from).to(rep_to).flatten()

    return Elementwise(signal, decode)
