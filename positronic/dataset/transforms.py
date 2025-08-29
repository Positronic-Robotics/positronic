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
                floor_ts = int(np.asarray(self._signal._ts_at([floor_idx]))[0])
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
