import numpy as np
import pytest

from positronic.dataset.signal import Kind
from positronic.dataset.vector import SimpleSignal, SimpleSignalWriter

from .utils import DummySignal


def create_signal(tmp_path, data_timestamps, name='test.parquet', names=None):
    """Helper to create a Signal with data and timestamps."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath, names=names) as writer:
        for data, ts in data_timestamps:
            writer.append(data, ts)
    return SimpleSignal(filepath)


def write_data(tmp_path, data_timestamps, name='test.parquet', names=None):
    """Helper to write data and return filepath."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath, names=names) as writer:
        for data, ts in data_timestamps:
            writer.append(data, ts)
    return filepath


class TestVectorMeta:
    def test_vector_start_last_ts_basic(self, tmp_path):
        fp = tmp_path / 'sig.parquet'
        with SimpleSignalWriter(fp) as w:
            w.append(1, 1000)
            w.append(2, 2000)
            w.append(3, 3000)
        s = SimpleSignal(fp)
        assert s.start_ts == 1000
        assert s.last_ts == 3000

    def test_vector_start_last_ts_empty_raises(self, tmp_path):
        fp = tmp_path / 'empty.parquet'
        with SimpleSignalWriter(fp):
            pass
        s = SimpleSignal(fp)
        with pytest.raises(ValueError):
            _ = s.start_ts
        with pytest.raises(ValueError):
            _ = s.last_ts


class TestSignalWriterAppend:
    def test_append_increasing_timestamps(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert len(signal) == 3
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_append_non_increasing_timestamp_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / 'test.parquet')
        with writer:
            writer.append(42, 1000)
            with pytest.raises(ValueError, match='is not increasing'):
                writer.append(43, 1000)
            with pytest.raises(ValueError, match='is not increasing'):
                writer.append(43, 999)

    def test_drop_equal_bytes_threshold_scalar(self, tmp_path):
        fp = tmp_path / 'dedupe_scalar.parquet'
        with SimpleSignalWriter(fp, drop_equal_bytes_threshold=32) as w:
            w.append(42, 1000)
            w.append(42, 2000)  # equal, dropped
            w.append(43, 3000)  # different, kept
        s = SimpleSignal(fp)
        assert len(s) == 2
        assert s[0] == (42, 1000)
        assert s[1] == (43, 3000)

    def test_drop_equal_bytes_threshold_numpy_small(self, tmp_path):
        fp = tmp_path / 'dedupe_array.parquet'
        with SimpleSignalWriter(fp, drop_equal_bytes_threshold=64) as w:
            w.append(np.array([1, 2, 3], dtype=np.int64), 1000)
            w.append(np.array([1, 2, 3], dtype=np.int64), 2000)  # equal content, dropped
            w.append(np.array([1, 2, 4], dtype=np.int64), 3000)  # different, kept
        s = SimpleSignal(fp)
        assert len(s) == 2
        v0, t0 = s[0]
        v1, t1 = s[1]
        np.testing.assert_array_equal(v0, [1, 2, 3])
        np.testing.assert_array_equal(v1, [1, 2, 4])
        assert (t0, t1) == (1000, 3000)


class TestSignalWriterContext:
    def test_context_empty_writer(self, tmp_path):
        filepath = write_data(tmp_path, [])
        assert filepath.exists()
        signal = SimpleSignal(filepath)
        assert len(signal) == 0

    def test_context_writes_data(self, tmp_path):
        filepath = tmp_path / 'test.parquet'
        with SimpleSignalWriter(filepath) as writer:
            writer.append(42, 1000)
        signal = SimpleSignal(filepath)
        assert signal.time[1000] == (42, 1000)

    def test_context_creates_file(self, tmp_path):
        filepath = tmp_path / 'test.parquet'
        with SimpleSignalWriter(filepath) as writer:
            writer.append(42, 1000)
            assert not filepath.exists()
        assert filepath.exists()

    def test_context_preserves_data_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_context_preserves_data_vector(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.5, 2.5]), 1000), (np.array([3.5, 4.5]), 2000)])
        value0, ts0 = signal[0]
        value1, ts1 = signal[1]
        np.testing.assert_array_equal(value0, [1.5, 2.5])
        np.testing.assert_array_equal(value1, [3.5, 4.5])
        assert ts0 == 1000
        assert ts1 == 2000

    def test_simple_writer_abort_removes_file_and_blocks_usage(self, tmp_path):
        fp = tmp_path / 'abort.parquet'
        with SimpleSignalWriter(fp) as w:
            w.append(1, 1000)
            w.abort()
            assert not fp.exists()
        with pytest.raises(RuntimeError):
            w.append(2, 2000)


class TestVectorInterface:
    def test_len_and_search_ts_empty(self, tmp_path):
        s = create_signal(tmp_path, [], 'empty.parquet')
        assert len(s) == 0
        empty = s._search_ts(np.array([], dtype=np.int64))
        assert isinstance(empty, np.ndarray)
        assert empty.size == 0

    def test_search_ts_numeric_and_invalid_dtype(self, tmp_path):
        s = create_signal(tmp_path, [(1, 1000), (2, 2000), (3, 3000)])
        # Accept float array: floor indices for 500, 1500, 2500, 3500
        idx = s._search_ts(np.array([500.0, 1500.0, 2500.0, 3500.0], dtype=np.float64))
        assert np.array_equal(idx, np.array([-1, 0, 1, 2]))
        # Accept scalar float via list-like contract
        assert s._search_ts([1999.9])[0] == 0
        # Reject non-numeric dtype
        with pytest.raises(TypeError):
            _ = s._search_ts(np.array(['1000'], dtype=object))

    def test_values_and_ts_at(self, tmp_path):
        s = create_signal(tmp_path, [(np.array([1, 2]), 1000), (np.array([3, 4]), 2000)])
        assert s._ts_at([1])[0] == 2000
        ts_arr = s._ts_at(np.array([0, 1], dtype=np.int64))
        assert np.array_equal(ts_arr, np.array([1000, 2000], dtype=np.int64))
        v0 = s._values_at([0])[0]
        np.testing.assert_array_equal(v0, [1, 2])
        varr = s._values_at(np.array([0, 1], dtype=np.int64))
        assert len(varr) == 2
        np.testing.assert_array_equal(varr[0], [1, 2])
        np.testing.assert_array_equal(varr[1], [3, 4])


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
        assert sig_simple[np.int64(2)] == (30, 3000)
        assert sig_simple[np.int32(0)] == (10, 1000)
        assert sig_simple[np.int64(-1)] == (50, 5000)


class TestCoreSignalTime:
    def test_time_scalar_cases(self, sig_simple):
        with pytest.raises(KeyError):
            _ = sig_simple.time[999]
        assert sig_simple.time[1000] == (10, 1000)
        assert sig_simple.time[2500] == (20, 2000)
        assert sig_simple.time[2500.0] == (20, 2000)
        assert sig_simple.time[9999] == (50, 5000)

    def test_time_window_basic(self, sig_simple):
        view = sig_simple.time[1500:4500]
        assert list(view) == [(10, 1500), (20, 2000), (30, 3000), (40, 4000)]
        v2 = sig_simple.time[:3000]
        assert len(v2) == 2
        v3 = sig_simple.time[3000:]
        assert len(v3) == 3
        v4 = sig_simple.time[:]
        assert len(v4) == len(sig_simple)
        assert len(sig_simple.time[:900]) == 0
        assert list(sig_simple.time[6000:]) == [(50, 6000)]

    def test_time_window_injects_start(self, sig_simple):
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
        assert len(v) == 2
        assert v[0] == (20, 2000)
        assert v[1] == (30, 3000)

    def test_time_window_start_before_first_no_inject(self, sig_simple):
        assert list(sig_simple.time[500:2500]) == [(10, 1000), (20, 2000)]

    def test_time_window_start_before_first_injects_start(self, sig_simple):
        assert list(sig_simple.time[100:900]) == []

    def test_time_stepped(self, sig_simple):
        sampled = sig_simple.time[1000:6000:2000]
        assert list(sampled) == [(10, 1000), (30, 3000), (50, 5000)]
        with pytest.raises(ValueError):
            _ = sig_simple.time[:5000:1000]
        with pytest.raises(ValueError):
            _ = sig_simple.time[1000:3000:0]
        with pytest.raises(ValueError):
            _ = sig_simple.time[1000:3000:-1000]
        with pytest.raises(KeyError):
            _ = sig_simple.time[500:3000:1000]
        full = sig_simple.time[1000::1000]
        assert list(full) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]

    def test_time_array_sampling(self, sig_simple):
        req = [1000, 1500, 3000]
        view = sig_simple.time[req]
        assert list(view) == [(10, 1000), (10, 1500), (30, 3000)]
        view2 = sig_simple.time[[3000, 1000, 3000]]
        assert list(view2) == [(30, 3000), (10, 1000), (30, 3000)]
        viewf = sig_simple.time[np.array([1000.0, 2500.0], dtype=np.float64)]
        assert list(viewf) == [(10, 1000.0), (20, 2500.0)]


class TestCoreSignalViews:
    def test_index_then_index(self, sig_simple):
        v1 = sig_simple[1:4]
        v2 = v1[::2]
        assert len(v2) == 2
        assert v2[0] == (20, 2000)
        assert v2[1] == (40, 4000)

    def test_time_slice_then_index(self, sig_simple):
        v = sig_simple.time[1500:4500]
        assert v[0] == (10, 1500)
        assert v[-1] == (40, 4000)
        vv = v[1:]
        assert len(vv) == 3
        assert vv[0] == (20, 2000)
        assert vv[1] == (30, 3000)
        assert vv[2] == (40, 4000)

    def test_time_sample_then_time_slice(self, sig_simple):
        sampled = sig_simple.time[[1500, 2500, 3500, 4500]]
        sub = sampled.time[2500:4500]
        assert list(sub) == [(20, 2500), (30, 3500)]

    def test_iteration_over_views(self, sig_simple):
        v = sig_simple.time[2000:5000]
        items = list(v)
        assert items == [(20, 2000), (30, 3000), (40, 4000)]

    def test_array_on_slice_indexing(self, sig_simple):
        v = sig_simple[1:5]
        sub = v[[0, 2]]
        assert list(sub) == [(20, 2000), (40, 4000)]

    def test_time_slice_of_time_slice(self, sig_simple):
        first = sig_simple.time[1500:4500]
        second = first.time[2000:3500]
        assert list(second) == [(20, 2000), (30, 3000)]

    def test_iter_over_stepped_sampling(self, sig_simple):
        sampled = sig_simple.time[1500:3500:1000]
        assert list(sampled) == [(10, 1500), (20, 2500)]

    def test_time_array_mixed_before_first(self, sig_simple):
        with pytest.raises(KeyError):
            _ = sig_simple.time[[500, 1000]]

    def test_time_array_empty(self, sig_simple):
        view = sig_simple.time[[]]
        assert len(view) == 0


class TestSignalDtypeShape:
    def test_empty_signal_dtype_shape_raises(self, tmp_path):
        fp = tmp_path / 'empty.parquet'
        with SimpleSignalWriter(fp):
            pass
        s = SimpleSignal(fp)
        with pytest.raises(ValueError):
            _ = s.dtype
        with pytest.raises(ValueError):
            _ = s.shape

    def test_scalar_signal_dtype_shape(self, tmp_path):
        s = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        # Python ints may be materialized as numpy integer scalars
        assert s.dtype in (int, np.int64, np.int32)
        assert s.shape == ()

    def test_scalar_signal_names_none(self, tmp_path):
        s = create_signal(tmp_path, [(1, 1000), (2, 2000)])
        assert s.names is None
        assert s.kind == Kind.NUMERIC

    def test_vector_signal_kind_and_names(self, tmp_path):
        arr1 = np.array([1.0, 2.0], dtype=np.float32)
        arr2 = np.array([3.0, 4.0], dtype=np.float32)
        s = create_signal(tmp_path, [(arr1, 1000), (arr2, 2000)], name='vec.parquet')
        assert s.names is None
        assert s.kind == Kind.NUMERIC

    def test_vector_signal_names_persist(self, tmp_path):
        arr1 = np.array([1.0, 2.0], dtype=np.float32)
        arr2 = np.array([3.0, 4.0], dtype=np.float32)
        feature_names = ['pos_x', 'pos_y']
        sig = create_signal(
            tmp_path,
            [(arr1, 1000), (arr2, 2000)],
            name='vec_named.parquet',
            names=feature_names,
        )
        assert sig.names == feature_names

    def test_signal_view_meta_inherits_and_empty_view_raises(self, tmp_path):
        s = create_signal(tmp_path, [(1, 1000), (2, 2000), (3, 3000)])
        view = s[1:3]
        assert view.kind == Kind.NUMERIC
        assert view.names is None
        empty = s[0:0]
        with pytest.raises(ValueError):
            _ = empty.dtype
        with pytest.raises(ValueError):
            _ = empty.shape

    def test_array_signal_dtype_shape(self, tmp_path):
        arr1 = np.array([1.0, 2.0], dtype=np.float32)
        arr2 = np.array([3.0, 4.0], dtype=np.float32)
        s = create_signal(tmp_path, [(arr1, 1000), (arr2, 2000)], name='arr.parquet')
        # dtype eq handles dtype('float32') vs np.float32
        assert s.dtype == np.float32
        assert s.shape == (2,)

    def test_tuple_signal_dtype_shape(self):
        ts = np.array([1000, 2000], dtype=np.int64)
        obj_vals = np.empty(2, dtype=object)
        obj_vals[0] = (np.array([1, 2, 3], dtype=np.int32), 5.0)
        obj_vals[1] = (np.array([4, 5, 6], dtype=np.int32), 6.0)
        sig = DummySignal(ts, obj_vals)
        assert sig.dtype == (np.int32, float)
        assert sig.shape == ((3,), ())

    def test_other_object_dtype_shape(self):
        ts = np.array([1000, 2000], dtype=np.int64)
        obj_vals = np.empty(2, dtype=object)
        obj_vals[0] = [1, 2, 3]
        obj_vals[1] = [4, 5, 6]
        sig = DummySignal(ts, obj_vals)
        assert sig.dtype is list
        assert sig.shape is None
