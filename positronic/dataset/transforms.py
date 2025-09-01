from typing import Callable, Sequence, TypeVar, Tuple

import numpy as np

from positronic.dataset.core import Signal, IndicesLike, RealNumericArrayLike

T = TypeVar("T")
U = TypeVar("U")


class Elementwise(Signal[U]):
    """Element-wise value transform view over a Signal.

    Wraps another `Signal[T]` and applies a function `f` to its values while
    preserving timestamps and ordering. Length and time indexing semantics are
    identical to the underlying signal.
    """

    def __init__(self, signal: Signal[T], fn: Callable[[Sequence[T]], Sequence[U]]):
        self._signal = signal
        self._fn = fn

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[U]:
        return self._fn(self._signal._values_at(indices))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        return self._signal._search_ts(ts_array)


class IndexOffsets(Signal[Tuple[Tuple[T, ...], Tuple[int, ...]]]):
    """Join values and timestamps at relative indices around a reference index.

    Given a list of relative indices D = [d1, d2, ..., dN] (each may be negative
    or positive), produces a view over the reference indices i where all
    (i + dk) are in-bounds. For each valid i, returns
        ((v[i+d1], ..., v[i+dN], t[i+d1], ..., t[i+dN]), t[i]).

    Examples:
      - Next with step=1  -> D = [0, 1]
      - Previous step=1   -> D = [0, -1]
    """

    def __init__(self, signal: Signal[T], relative_indices: Sequence[int]) -> None:
        self._signal = signal
        offs = np.asarray(relative_indices, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("relative_indices must be non-empty")
        self._offs = offs
        self._min_off = int(np.min(self._offs))
        self._max_off = int(np.max(self._offs))

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

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        base = np.asarray(indices, dtype=np.int64) + self._base_start()
        return self._signal._ts_at(base)

    def _values_at(self, indices: IndicesLike):
        base = np.asarray(indices, dtype=np.int64) + self._base_start()
        vals_parts = []
        ts_parts = []
        for off in self._offs:
            idxs = base + int(off)
            vals_parts.append(self._signal._values_at(idxs))
            ts_parts.append(np.asarray(self._signal._ts_at(idxs)))

        return list(zip(*vals_parts, *ts_parts, strict=False))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        # Map parent floor indices to view indices, clamping to valid range.
        n = len(self)
        t = np.asarray(ts_array)
        if n == 0:
            return np.full_like(t, -1, dtype=np.int64)  # nothing valid in this view
        p = np.asarray(self._signal._search_ts(t))
        base_start = self._base_start()
        base_last = self._base_last()
        view_idx = p - base_start
        view_idx[p < base_start] = -1
        view_idx[p > base_last] = n - 1
        return view_idx


class TimeOffsets(Signal[Tuple[Tuple[T, ...], Tuple[int, ...]]]):
    """Sample values at time offsets relative to each reference timestamp.

    Given deltas D = [d1, d2, ..., dN], for each valid reference index i returns
        ((v_at(ts[i]+d1), ..., v_at(ts[i]+dN), t_at(ts[i]+d1), ..., t_at(ts[i]+dN)), ts[i])
    where v_at(t) carries back the last value at-or-before t.

    Semantics:
    - For negative deltas: elements whose shifted time precedes the first
      timestamp are dropped (affects the start of the series).
    - For non-negative deltas: when the shifted time exceeds the last
      timestamp, sampling clamps to the last element.
    """

    def __init__(self, signal: Signal[T], deltas_ts: Sequence[int]) -> None:
        self._signal = signal
        offs = np.asarray(deltas_ts, dtype=np.int64)
        if offs.size == 0:
            raise ValueError("deltas_ts must be non-empty")
        self._deltas = offs
        self._bounds_ready = False
        self._start_offset = 0
        self._last_index = -1

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        n = len(self._signal)
        if n == 0:
            self._start_offset = 0
            self._last_index = -1
            self._bounds_ready = True
            return
        start_offset = 0
        last_index = n - 1
        neg = self._deltas[self._deltas < 0]
        if neg.size > 0:
            thr = int(self._signal.start_ts + int(np.max(-neg)))
            floor_idx = int(np.asarray(self._signal._search_ts([thr]))[0])
            if floor_idx < 0:
                start_offset = 0
            else:
                floor_ts = _ts_at_index(self._signal, floor_idx)
                start_offset = floor_idx if floor_ts == thr else floor_idx + 1
        self._start_offset = start_offset
        self._last_index = last_index
        self._bounds_ready = True

    def __len__(self) -> int:
        self._compute_bounds()
        if self._last_index < self._start_offset:
            return 0
        return self._last_index - self._start_offset + 1

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._compute_bounds()
        idxs = np.asarray(indices, dtype=np.int64)
        if self._start_offset == 0:
            return self._signal._ts_at(idxs)
        else:
            return self._signal._ts_at(idxs + self._start_offset)

    def _values_at(self, indices: IndicesLike):
        self._compute_bounds()
        base = np.asarray(indices, dtype=np.int64)
        if self._start_offset > 0:
            base = base + self._start_offset
        ref_ts = np.asarray(self._signal._ts_at(base))
        vals_parts = []
        ts_parts = []
        for d in self._deltas:
            target_ts = ref_ts + int(d)
            idx = np.asarray(self._signal._search_ts(target_ts))
            vals_parts.append(self._signal._values_at(idx))
            ts_parts.append(np.asarray(self._signal._ts_at(idx)))
        out = []
        for row in zip(*vals_parts, *ts_parts, strict=False):
            out.append(tuple(row))
        return out

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        parent_idx = np.asarray(self._signal._search_ts(ts_array))
        if (self._deltas < 0).any():
            shifted = parent_idx - self._start_offset
            shifted[parent_idx < self._start_offset] = -1
            return shifted
        return parent_idx


def _ts_at_index(sig: Signal[T], idx: int) -> int:
    """Fetch a single timestamp at the given index as int."""
    return int(np.asarray(sig._ts_at([idx]))[0])


def _first_idx_at_or_after(sig: Signal[T], ts: int) -> int:
    floor = int(np.asarray(sig._search_ts([ts]))[0])
    if floor < 0:
        return 0
    floor_ts = _ts_at_index(sig, floor)
    return floor if floor_ts == ts else floor + 1


class Interleave(Signal[Tuple[T, U, int]]):
    """Merge two signals on the union of their timestamps with carry-back.

    - Reference times: sorted union of parents' timestamps, starting from
      max(s1.start_ts, s2.start_ts).
      - When `drop_duplicates=False`: if both have a sample at the same
        timestamp, both entries are included (s1 precedes s2).
      - When `drop_duplicates=True`: equal timestamps are collapsed into a
        single entry.
    - Values: at each union timestamp t, returns
      ((v1_at_or_before_t, v2_at_or_before_t, ts2_ref - ts1_ref), t), where
      ts*_ref are the timestamps of the carried-back values in each parent.
    - No materialization in the default mode; when `drop_duplicates=True`, the
      union timestamps are precomputed for O(log N) time lookups (values are not
      materialized).
    """

    def __init__(
        self, s1: Signal[T], s2: Signal[U], drop_duplicates: bool = True
    ) -> None:
        self._s1 = s1
        self._s2 = s2
        self._drop_duplicates = drop_duplicates
        self._bounds_ready = False
        self._s1_start = 0
        self._s2_start = 0
        self._length = 0
        self._union_ts: np.ndarray | None = None

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        n1, n2 = len(self._s1), len(self._s2)
        if n1 == 0 or n2 == 0:
            self._s1_start = n1
            self._s2_start = n2
            self._length = 0
            self._bounds_ready = True
            return
        start_ts = max(self._s1.start_ts, self._s2.start_ts)
        self._s1_start = _first_idx_at_or_after(self._s1, start_ts)
        self._s2_start = _first_idx_at_or_after(self._s2, start_ts)
        if self._drop_duplicates:
            # Build union timestamps using a single merge implementation
            self._union_ts = np.asarray(
                list(self._iter_merged_ts(dedup=True)), dtype=np.int64
            )
            self._length = int(self._union_ts.shape[0])
        else:
            # Include both entries when timestamps are equal
            self._length = (n1 - self._s1_start) + (n2 - self._s2_start)
        self._bounds_ready = True

    def _iter_merged_ts(self, dedup: bool):
        i1, i2 = self._s1_start, self._s2_start
        n1, n2 = len(self._s1), len(self._s2)
        inf_ts = np.iinfo(np.int64).max
        while i1 < n1 or i2 < n2:
            ts1 = _ts_at_index(self._s1, i1) if i1 < n1 else inf_ts
            ts2 = _ts_at_index(self._s2, i2) if i2 < n2 else inf_ts
            if dedup and ts1 == ts2:
                yield ts1
                i1 += 1
                i2 += 1
            elif ts1 <= ts2:
                yield ts1
                i1 += 1
            else:
                yield ts2
                i2 += 1

    def _kth_union_ts(self, k: int) -> int:
        """Find the k-th timestamp (0-based) in the merged union (duplicates kept).

        This uses the classic "k-th element of two sorted arrays" selection
        algorithm. We treat both parent timestamp arrays as sorted, immutable
        sequences and maintain two cursors (a_idx for s1, b_idx for s2), both
        starting from the first valid indices after alignment to the common
        start timestamp. At each step we discard a chunk from one of the arrays
        that cannot contain the k-th union element, shrinking the search space
        geometrically.

        In more detail:
        - If either array is exhausted, the answer is directly the (k)-th
          remaining element of the other array.
        - If k == 0, the answer is simply min(head1, head2).
        - Otherwise, we look ahead by `step = floor((k-1)/2)` elements in each
          array (capped to the array's end). We compare the lookahead elements
          ta = s1[a_idx + step] and tb = s2[b_idx + step]. If ta <= tb, then all
          elements up to and including a_idx+step in s1 cannot be the k-th union
          element (there are at least step+1 elements <= tb from s1 alone), so we
          discard them and reduce k by (step+1). Otherwise, we discard the same
          sized prefix from s2. This yields O(log(k)) steps.

        Complexity: O(log(k)) timestamp fetches; no materialization of the
        merged union.
        """
        # Cursors start at the first valid indices in each parent
        a_start, b_start = self._s1_start, self._s2_start
        n1, n2 = len(self._s1), len(self._s2)
        a_idx, b_idx = a_start, b_start
        k_remaining = int(k)  # remaining 0-based rank to find in the merged stream
        while True:
            # If one array is exhausted, answer is in the other at offset kk
            if a_idx >= n1:
                return _ts_at_index(self._s2, b_idx + k_remaining)
            if b_idx >= n2:
                return _ts_at_index(self._s1, a_idx + k_remaining)
            # Base case: k points at the current heads -> take the smaller ts
            if k_remaining == 0:
                ta = _ts_at_index(self._s1, a_idx)
                tb = _ts_at_index(self._s2, b_idx)
                return min(ta, tb)
            # Probe ahead by step ≈ k_remaining/2 in each array to discard a whole block
            step = (k_remaining - 1) // 2
            a_next = min(a_idx + step, n1 - 1)  # clamp to end
            b_next = min(b_idx + step, n2 - 1)
            ta = _ts_at_index(self._s1, a_next)
            tb = _ts_at_index(self._s2, b_next)
            if ta <= tb:  # discard s1[a_idx : a_next+1]
                k_remaining -= a_next - a_idx + 1
                a_idx = a_next + 1
            else:
                k_remaining -= b_next - b_idx + 1
                b_idx = b_next + 1

    def __len__(self) -> int:
        self._compute_bounds()
        return self._length

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._compute_bounds()
        idxs = np.asarray(indices)
        if self._drop_duplicates:
            return self._union_ts[idxs]
        # Smarter per-position selection without materializing the union
        return np.asarray([self._kth_union_ts(int(k)) for k in idxs], dtype=np.int64)

    def _values_at(self, indices: IndicesLike) -> Sequence[Tuple[T, U, int]]:
        ts = np.asarray(self._ts_at(indices))
        # Sample both parents at these timestamps and compute dt between refs
        idx1 = np.asarray(self._s1._search_ts(ts))
        idx2 = np.asarray(self._s2._search_ts(ts))
        v1 = self._s1._values_at(idx1)
        v2 = self._s2._values_at(idx2)
        t1 = np.asarray(self._s1._ts_at(idx1))
        t2 = np.asarray(self._s2._ts_at(idx2))
        dt = t2 - t1
        return list(zip(v1, v2, dt, strict=True))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        t = np.asarray(ts_array)
        if self._drop_duplicates:
            assert self._union_ts is not None
            return np.searchsorted(self._union_ts, t, side="right") - 1

        # Non-deduped: floor rank is sum of floors in each parent past the start offsets
        f1 = np.asarray(self._s1._search_ts(t))
        f2 = np.asarray(self._s2._search_ts(t))
        c1 = np.maximum(0, f1 - self._s1_start + 1)
        c2 = np.maximum(0, f2 - self._s2_start + 1)
        total = c1 + c2
        return np.maximum(-1, total - 1)
