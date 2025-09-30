from collections.abc import Sequence

import numpy as np

from positronic.dataset.signal import (
    IndicesLike,
    RealNumericArrayLike,
    Signal,
    SignalMeta,
    is_realnum_dtype,
)


class DummySignal(Signal[int]):
    """Minimal array-backed Signal implementing the abstract API only.

    Used to validate core Signal's generic indexing/time logic and views.
    """

    def __init__(self, timestamps, values, names: Sequence[str] | None = None):
        ts_arr = np.asarray(timestamps, dtype=np.int64)
        vals_arr = np.asarray(values)
        assert ts_arr.ndim == 1
        assert vals_arr.shape[0] == ts_arr.shape[0]
        self._ts = ts_arr
        self._vals = vals_arr
        self._names = list(names) if names is not None else None
        self._meta_override: SignalMeta | None = None

    @property
    def meta(self) -> SignalMeta:
        b_meta = super().meta
        return b_meta.with_names(self._names)

    def __len__(self) -> int:
        return int(self._ts.shape[0])

    def _ts_at(self, index_or_indices: IndicesLike) -> np.ndarray:
        idxs = np.asarray(index_or_indices, dtype=np.int64)
        if idxs.size == 0:
            return np.array([], dtype=np.int64)
        return self._ts[idxs]

    def _values_at(self, index_or_indices: IndicesLike):
        idxs = np.asarray(index_or_indices, dtype=np.int64)
        if idxs.size == 0:
            return []
        return self._vals[idxs]

    def _search_ts(self, ts_or_array: RealNumericArrayLike) -> np.ndarray:
        req = np.asarray(ts_or_array)
        if req.size == 0:
            return np.array([], dtype=np.int64)
        if not is_realnum_dtype(req.dtype):
            raise TypeError(f'Invalid timestamp array dtype: {req.dtype}')
        return np.searchsorted(self._ts, req, side='right') - 1
