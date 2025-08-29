from typing import Callable, Sequence, TypeVar, Tuple

import numpy as np

from positronic.dataset.core import Signal, IndicesLike, RealNumericArrayLike

T = TypeVar('T')
U = TypeVar('U')


class Elementwise(Signal[U]):
    """Element-wise value transform view over a Signal.

    Wraps another `Signal[T]` and applies a function `f` to its values while
    preserving timestamps and ordering. Length and time indexing semantics are
    identical to the underlying signal.
    """

    def __init__(self, signal: Signal[T], f: Callable[[T | Sequence[T]], U | Sequence[U]]) -> None:
        self._signal = signal
        self._f = f

    def __len__(self) -> int:
        return len(self._signal)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[U]:
        return self._f(self._signal._values_at(indices))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        return self._signal._search_ts(ts_array)


class Previous(Signal[Tuple[T, T, int]]):
    """Pairs each sample with the previous value and time delta.

    For index i > 0, yields ((cur_value, prev_value, ts[i] - ts[i-1]), ts[i]).
    The first element has no previous sample and is therefore omitted.
    """

    def __init__(self, signal: Signal[T]) -> None:
        self._signal = signal

    def __len__(self) -> int:
        # First element is removed, as it has no previous
        return max(len(self._signal) - 1, 0)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        indices = np.asarray(indices) + 1
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[Tuple[T, T, int]]:
        indices = np.asarray(indices)
        prev_values = self._signal._values_at(indices)
        cur_values = self._signal._values_at(indices + 1)
        prev_ts = self._signal._ts_at(indices)
        cur_ts = self._ts_at(indices)
        return list(zip(cur_values, prev_values, cur_ts - prev_ts, strict=True))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        result = self._signal._search_ts(ts_array)
        return np.asarray(result) - 1


class Next(Signal[Tuple[T, T, int]]):
    """Pairs each sample with the next value and time delta.

    For index i < len-1, yields ((cur_value, next_value, ts[i+1] - ts[i]), ts[i]).
    The last element has no next sample and is therefore omitted.
    """

    def __init__(self, signal: Signal[T]) -> None:
        self._signal = signal

    def __len__(self) -> int:
        # Last element is removed, as it has no next
        return max(len(self._signal) - 1, 0)

    def _ts_at(self, indices: IndicesLike) -> Sequence[int] | np.ndarray:
        indices = np.asarray(indices)
        return self._signal._ts_at(indices)

    def _values_at(self, indices: IndicesLike) -> Sequence[Tuple[T, T, int]]:
        indices = np.asarray(indices)
        next_values = self._signal._values_at(indices + 1)
        cur_values = self._signal._values_at(indices)
        next_ts = self._signal._ts_at(indices + 1)
        cur_ts = self._ts_at(indices)
        return list(zip(cur_values, next_values, next_ts - cur_ts, strict=True))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        result = self._signal._search_ts(ts_array)
        return np.asarray(result)


class JoinDeltaTime(Signal[Tuple[T, T, int]]):
    """Join each sample with the value at a time offset `delta_ts`.

    - Positive `delta_ts` (future join): length is preserved and timestamps are
      unchanged. Each element i returns ((v[i], v_at(ts[i] + delta), dt), ts[i]),
      where v_at(t) carries back the last value at-or-before t. If t exceeds the
      last timestamp, the last value is used and dt becomes 0 (clamped).

    - Negative `delta_ts` (past join): elements whose shifted time precedes the
      first timestamp are dropped. For remaining elements i, returns
      ((v[i], v_at(ts[i] + delta), dt), ts[i]) with dt = ts[i] - ts_at(ts[i] + delta).

    - Zero `delta_ts`: pairs each element with itself and dt = 0.
    """

    def __init__(self, signal: Signal[T], delta_ts: int) -> None:
        self._signal = signal
        self._delta_ts = int(delta_ts)
        self._bounds_ready = False
        self._start_offset = 0
        self._last_index = -1

    def _compute_bounds(self) -> None:
        if self._bounds_ready:
            return
        sig_len = len(self._signal)
        if sig_len == 0:
            self._start_offset = 0
            self._last_index = -1
            self._bounds_ready = True
            return

        # Default bounds include the whole signal
        start_offset = 0
        last_index = sig_len - 1

        if self._delta_ts < 0:
            # Valid i satisfy ts[i] >= start_ts + |delta|
            thr_ts = self._signal.start_ts + (-self._delta_ts)
            floor_idx = int(np.asarray(self._signal._search_ts([thr_ts]))[0])
            if floor_idx < 0:
                start_offset = 0
            else:
                floor_ts = _ts_at_index(self._signal, floor_idx)
                start_offset = floor_idx if floor_ts == thr_ts else floor_idx + 1

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
        idxs = np.asarray(indices)
        if self._start_offset == 0:
            return self._signal._ts_at(idxs)
        else:
            return self._signal._ts_at(idxs + self._start_offset)

    def _values_at(self, indices: IndicesLike) -> Sequence[Tuple[T, T, int]]:
        self._compute_bounds()
        mapped = np.asarray(indices)
        if self._start_offset > 0:
            mapped += self._start_offset
        cur_vals = self._signal._values_at(mapped)
        cur_ts = np.asarray(self._signal._ts_at(mapped))
        target_ts = cur_ts + self._delta_ts
        match_idx = np.asarray(self._signal._search_ts(target_ts))
        delta_vals = self._signal._values_at(match_idx)
        then_ts = np.asarray(self._signal._ts_at(match_idx))
        if self._delta_ts >= 0:
            dt = then_ts - cur_ts
        else:
            dt = cur_ts - then_ts
        return list(zip(cur_vals, delta_vals, dt, strict=True))

    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike:
        self._compute_bounds()
        parent_idx = np.asarray(self._signal._search_ts(ts_array))
        if self._delta_ts >= 0:
            # Same indexing as the parent view
            return parent_idx
        else:
            # Shift by start offset; anything before becomes -1
            shifted = parent_idx - self._start_offset
            shifted[parent_idx < self._start_offset] = -1
            return shifted


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

    def __init__(self, s1: Signal[T], s2: Signal[U], drop_duplicates: bool = True) -> None:
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
            self._union_ts = np.asarray(list(self._iter_merged_ts(dedup=True)), dtype=np.int64)
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
                k_remaining -= (a_next - a_idx + 1)
                a_idx = a_next + 1
            else:
                k_remaining -= (b_next - b_idx + 1)
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
            return np.searchsorted(self._union_ts, t, side='right') - 1

        # Non-deduped: floor rank is sum of floors in each parent past the start offsets
        f1 = np.asarray(self._s1._search_ts(t))
        f2 = np.asarray(self._s2._search_ts(t))
        c1 = np.maximum(0, f1 - self._s1_start + 1)
        c2 = np.maximum(0, f2 - self._s2_start + 1)
        total = c1 + c2
        return np.maximum(-1, total - 1)
