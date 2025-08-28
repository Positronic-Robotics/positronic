import numpy as np

from positronic.dataset.core import IndicesLike, RealNumericArrayLike, is_realnum_dtype, Signal


class DummySignal(Signal[int]):
    """Minimal array-backed Signal implementing the abstract API only.

    Used to validate core Signal's generic indexing/time logic and views.
    """

    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        assert timestamps.ndim == 1
        assert values.shape[0] == timestamps.shape[0]
        self._ts = timestamps.astype(np.int64, copy=False)
        self._vals = values

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
            raise TypeError(f"Invalid timestamp array dtype: {req.dtype}")
        return np.searchsorted(self._ts, req, side='right') - 1
