import numpy as np
import pytest

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


@pytest.fixture
def sig_simple():
    ts = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    vals = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    return DummySignal(ts, vals)


class TestCoreSignalBasics:

    def test_start_last_ts_basic(self, sig_simple):
        assert sig_simple.start_ts == 1000
        assert sig_simple.last_ts == 5000

    def test_index_scalar_and_negative(self, sig_simple):
        assert sig_simple[0] == (10, 1000)
        assert sig_simple[2] == (30, 3000)
        assert sig_simple[-1] == (50, 5000)
        assert sig_simple[-5] == (10, 1000)
        with pytest.raises(IndexError):
            _ = sig_simple[5]
        with pytest.raises(IndexError):
            _ = sig_simple[-6]

    def test_index_slice(self, sig_simple):
        view = sig_simple[1:4]
        assert len(view) == 3
        assert view[0] == (20, 2000)
        assert view[2] == (40, 4000)
        step_view = sig_simple[0:5:2]
        assert len(step_view) == 3
        assert step_view[0] == (10, 1000)
        assert step_view[1] == (30, 3000)
        assert step_view[2] == (50, 5000)
        with pytest.raises(ValueError):
            _ = sig_simple[::0]
        with pytest.raises(ValueError):
            _ = sig_simple[::-1]

    def test_index_array(self, sig_simple):
        view = sig_simple[[0, 2, 4]]
        assert list(view) == [(10, 1000), (30, 3000), (50, 5000)]

        view2 = sig_simple[np.array([0, -1, 1, -2], dtype=np.int64)]
        assert list(view2) == [(10, 1000), (50, 5000), (20, 2000), (40, 4000)]
        with pytest.raises(IndexError):
            _ = sig_simple[np.array([0, 5], dtype=np.int64)]
        with pytest.raises(IndexError):
            _ = sig_simple[np.array([True, False, True, False, True], dtype=np.bool_)]
        with pytest.raises(TypeError):
            _ = sig_simple[np.array([0.0, 1.0], dtype=np.float64)]
        empty = sig_simple[[]]
        assert len(empty) == 0

    def test_index_numpy_integer_scalars(self, sig_simple):
        # Previously: numpy integer scalars did not match case int() and raised TypeError
        assert sig_simple[np.int64(2)] == (30, 3000)
        assert sig_simple[np.int32(0)] == (10, 1000)
        # Negative numpy integers are supported via normalization
        assert sig_simple[np.int64(-1)] == (50, 5000)


class TestCoreSignalTime:

    def test_time_scalar_cases(self, sig_simple):
        # Before first
        with pytest.raises(KeyError):
            _ = sig_simple.time[999]
        # Exact
        assert sig_simple.time[1000] == (10, 1000)
        # Between
        assert sig_simple.time[2500] == (20, 2000)
        # Float scalar should be accepted
        assert sig_simple.time[2500.0] == (20, 2000)
        # After last
        assert sig_simple.time[9999] == (50, 5000)

    def test_time_window_basic(self, sig_simple):
        view = sig_simple.time[1500:4500]
        # Injects a start sample at 1500 (carry-back 10), then 2000, 3000, 4000 (end-exclusive)
        assert list(view) == [(10, 1500), (20, 2000), (30, 3000), (40, 4000)]
        # Missing bounds
        v2 = sig_simple.time[:3000]
        assert len(v2) == 2  # 1000, 2000 (start None -> no injection)
        v3 = sig_simple.time[3000:]
        assert len(v3) == 3
        v4 = sig_simple.time[:]
        assert len(v4) == len(sig_simple)
        # Outside ranges
        assert len(sig_simple.time[:900]) == 0
        assert list(sig_simple.time[6000:]) == [(50, 6000)]

    def test_time_window_injects_start(self, sig_simple):
        # Injection occurs when start is provided and >= first_ts but no exact record at start
        v = sig_simple.time[1500:3500]
        assert len(v) == 3
        assert v[0] == (10, 1500)
        assert v[1] == (20, 2000)
        assert v[2] == (30, 3000)

    def test_time_stepped_empty_signal(self):
        ts = np.array([], dtype=np.int64)
        vals = np.array([], dtype=np.int64)
        sig = DummySignal(ts, vals)
        sampled = sig.time[1000:5000:1000]
        assert len(sampled) == 0

    def test_time_window_no_inject_when_exact(self, sig_simple):
        v = sig_simple.time[2000:3500]
        # Exact start at 2000 -> no injection
        assert len(v) == 2
        assert v[0] == (20, 2000)
        assert v[1] == (30, 3000)

    def test_time_window_start_before_first_no_inject(self, sig_simple):
        # start before first_ts -> cannot inject; starts at first record 1000
        assert list(sig_simple.time[500:2500]) == [(10, 1000), (20, 2000)]

    def test_time_window_start_before_first_injects_start(self, sig_simple):
        assert list(sig_simple.time[100:900]) == []

    def test_time_stepped(self, sig_simple):
        sampled = sig_simple.time[1000:6000:2000]
        assert list(sampled) == [(10, 1000), (30, 3000), (50, 5000)]
        # Missing start raises
        with pytest.raises(ValueError):
            _ = sig_simple.time[:5000:1000]
        # Non-positive step raises
        with pytest.raises(ValueError):
            _ = sig_simple.time[1000:3000:0]
        with pytest.raises(ValueError):
            _ = sig_simple.time[1000:3000:-1000]
        # Start before first raises
        with pytest.raises(KeyError):
            _ = sig_simple.time[500:3000:1000]

        # Missing stop samples until last_ts+1
        full = sig_simple.time[1000::1000]
        assert list(full) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]

    def test_time_array_sampling(self, sig_simple):
        req = [1000, 1500, 3000]
        view = sig_simple.time[req]
        assert list(view) == [(10, 1000), (10, 1500), (30, 3000)]
        # Unsorted + duplicates
        view2 = sig_simple.time[[3000, 1000, 3000]]
        assert list(view2) == [(30, 3000), (10, 1000), (30, 3000)]
        # Float timestamps are accepted for arrays
        viewf = sig_simple.time[np.array([1000.0, 2500.0], dtype=np.float64)]
        assert list(viewf) == [(10, 1000.0), (20, 2500.0)]


class TestCoreSignalViews:

    def test_index_then_index(self, sig_simple):
        v1 = sig_simple[1:4]  # values at 2000, 3000, 4000
        v2 = v1[::2]
        assert len(v2) == 2
        assert v2[0] == (20, 2000)
        assert v2[1] == (40, 4000)

    def test_time_slice_then_index(self, sig_simple):
        v = sig_simple.time[1500:4500]  # 1500(injected), 2000, 3000, 4000
        assert v[0] == (10, 1500)
        assert v[-1] == (40, 4000)
        vv = v[1:]
        assert len(vv) == 3
        assert vv[0] == (20, 2000)
        assert vv[1] == (30, 3000)
        assert vv[2] == (40, 4000)

    def test_time_sample_then_time_slice(self, sig_simple):
        sampled = sig_simple.time[[1500, 2500, 3500, 4500]]  # requested ts preserved
        # Slice the sampled view by time (non-stepped)
        sub = sampled.time[2500:4500]
        # End is exclusive, so 4500 is not included
        assert list(sub) == [(20, 2500), (30, 3500)]

    def test_iteration_over_views(self, sig_simple):
        v = sig_simple.time[2000:5000]
        items = list(v)
        assert items == [(20, 2000), (30, 3000), (40, 4000)]

    def test_array_on_slice_indexing(self, sig_simple):
        # Slice then array-index into the slice
        v = sig_simple[1:5]  # indices 1..4 -> ts 2000..5000
        sub = v[[0, 2]]  # pick 2000 and 4000
        assert list(sub) == [(20, 2000), (40, 4000)]

    def test_time_slice_of_time_slice(self, sig_simple):
        # Time slice, then another time slice (non-stepped)
        first = sig_simple.time[1500:4500]  # 1500(injected), 2000, 3000, 4000
        second = first.time[2000:3500]  # 2000, 3000
        assert list(second) == [(20, 2000), (30, 3000)]

    def test_iter_over_stepped_sampling(self, sig_simple):
        # Iteration over stepped time sampling preserves requested timestamps
        sampled = sig_simple.time[1500:3500:1000]
        assert list(sampled) == [(10, 1500), (20, 2500)]

    def test_time_array_mixed_before_first(self, sig_simple):
        # Any timestamp before first must raise for array access
        with pytest.raises(KeyError):
            _ = sig_simple.time[[500, 1000]]

    def test_time_array_empty(self, sig_simple):
        view = sig_simple.time[[]]
        assert len(view) == 0
