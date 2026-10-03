import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.dataset.signal import RECORDED_TIME, TIMELINE_METADATA_KEY, Kind
from positronic.dataset.time import Time
from positronic.dataset.transforms.signals import Join, diff, norm
from positronic.dataset.vector import SIGNAL_VERSION, SIGNAL_VERSION_KEY, SimpleSignal, SimpleSignalWriter

from .utils import DummySignal


def create_signal(tmp_path, data_timestamps, name='test.parquet'):
    """Helper to create a Signal with data and timestamps."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath) as writer:
        for data, ts in data_timestamps:
            writer.append(data, Time(**{RECORDED_TIME: ts}))
    return SimpleSignal(filepath)


def write_data(tmp_path, data_timestamps, name='test.parquet'):
    """Helper to write data and return filepath."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath) as writer:
        for data, ts in data_timestamps:
            writer.append(data, Time(**{RECORDED_TIME: ts}))
    return filepath


class TestVectorMeta:
    def test_vector_start_last_ts_basic(self, tmp_path):
        fp = tmp_path / 'sig.parquet'
        with SimpleSignalWriter(fp) as w:
            w.append(1, Time(**{RECORDED_TIME: 1000}))
            w.append(2, Time(**{RECORDED_TIME: 2000}))
            w.append(3, Time(**{RECORDED_TIME: 3000}))
        s = SimpleSignal(fp)
        assert s.bounds(RECORDED_TIME)[0][RECORDED_TIME] == 1000
        assert s.bounds(RECORDED_TIME)[1][RECORDED_TIME] == 3000

    def test_vector_start_last_ts_empty_raises(self, tmp_path):
        fp = tmp_path / 'empty.parquet'
        with SimpleSignalWriter(fp):
            pass
        s = SimpleSignal(fp)
        with pytest.raises(ValueError):
            _ = s.bounds(RECORDED_TIME)[0][RECORDED_TIME]
        with pytest.raises(ValueError):
            _ = s.bounds(RECORDED_TIME)[1][RECORDED_TIME]


class TestSignalWriterAppend:
    def test_append_increasing_timestamps(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert len(signal) == 3
        assert signal[0] == (42, Time(**{RECORDED_TIME: 1000}))
        assert signal[1] == (43, Time(**{RECORDED_TIME: 2000}))
        assert signal[2] == (44, Time(**{RECORDED_TIME: 3000}))

    def test_append_non_increasing_timestamp_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / 'test.parquet')
        with writer:
            writer.append(42, Time(**{RECORDED_TIME: 1000}))
            with pytest.raises(ValueError, match='is not increasing'):
                writer.append(43, Time(**{RECORDED_TIME: 1000}))
            with pytest.raises(ValueError, match='is not increasing'):
                writer.append(43, Time(**{RECORDED_TIME: 999}))

    def test_drop_equal_bytes_threshold_scalar(self, tmp_path):
        fp = tmp_path / 'dedupe_scalar.parquet'
        with SimpleSignalWriter(fp, drop_equal_bytes_threshold=32) as w:
            w.append(42, Time(**{RECORDED_TIME: 1000}))
            w.append(42, Time(**{RECORDED_TIME: 2000}))  # equal, dropped
            w.append(43, Time(**{RECORDED_TIME: 3000}))  # different, kept
        s = SimpleSignal(fp)
        assert len(s) == 2
        assert s[0] == (42, Time(**{RECORDED_TIME: 1000}))
        assert s[1] == (43, Time(**{RECORDED_TIME: 3000}))

    def test_drop_equal_bytes_threshold_numpy_small(self, tmp_path):
        fp = tmp_path / 'dedupe_array.parquet'
        with SimpleSignalWriter(fp, drop_equal_bytes_threshold=64) as w:
            w.append(np.array([1, 2, 3], dtype=np.int64), Time(**{RECORDED_TIME: 1000}))
            w.append(np.array([1, 2, 3], dtype=np.int64), Time(**{RECORDED_TIME: 2000}))  # equal content, dropped
            w.append(np.array([1, 2, 4], dtype=np.int64), Time(**{RECORDED_TIME: 3000}))  # different, kept
        s = SimpleSignal(fp)
        assert len(s) == 2
        v0, t0 = s[0]
        v1, t1 = s[1]
        np.testing.assert_array_equal(v0, [1, 2, 3])
        np.testing.assert_array_equal(v1, [1, 2, 4])
        assert (t0, t1) == (Time(**{RECORDED_TIME: 1000}), Time(**{RECORDED_TIME: 3000}))


class TestSignalWriterContext:
    def test_context_empty_writer(self, tmp_path):
        filepath = write_data(tmp_path, [])
        assert filepath.exists()
        signal = SimpleSignal(filepath)
        assert len(signal) == 0

    def test_context_writes_data(self, tmp_path):
        filepath = tmp_path / 'test.parquet'
        with SimpleSignalWriter(filepath) as writer:
            writer.append(42, Time(**{RECORDED_TIME: 1000}))
        signal = SimpleSignal(filepath)
        assert signal.time[Time(**{RECORDED_TIME: 1000})] == (42, Time(**{RECORDED_TIME: 1000}))

    def test_context_creates_file(self, tmp_path):
        filepath = tmp_path / 'test.parquet'
        with SimpleSignalWriter(filepath) as writer:
            writer.append(42, Time(**{RECORDED_TIME: 1000}))
            assert not filepath.exists()
        assert filepath.exists()

    def test_context_preserves_data_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal[0] == (42, Time(**{RECORDED_TIME: 1000}))
        assert signal[1] == (43, Time(**{RECORDED_TIME: 2000}))
        assert signal[2] == (44, Time(**{RECORDED_TIME: 3000}))

    def test_context_preserves_data_vector(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.5, 2.5]), 1000), (np.array([3.5, 4.5]), 2000)])
        value0, ts0 = signal[0]
        value1, ts1 = signal[1]
        np.testing.assert_array_equal(value0, [1.5, 2.5])
        np.testing.assert_array_equal(value1, [3.5, 4.5])
        assert ts0 == Time(**{RECORDED_TIME: 1000})
        assert ts1 == Time(**{RECORDED_TIME: 2000})

    def test_simple_writer_abort_removes_file_and_blocks_usage(self, tmp_path):
        fp = tmp_path / 'abort.parquet'
        with SimpleSignalWriter(fp) as w:
            w.append(1, Time(**{RECORDED_TIME: 1000}))
            w.abort()
            assert not fp.exists()
        with pytest.raises(RuntimeError):
            w.append(2, Time(**{RECORDED_TIME: 2000}))


class TestSignalWriterChunking:
    def test_large_dataset_chunked_writing(self, tmp_path):
        filepath = tmp_path / 'large_test.parquet'
        chunk_size, num_records = 1000, 10000

        with SimpleSignalWriter(filepath, chunk_size=chunk_size) as writer:
            for i in range(num_records):
                data = np.array([i, i * 2, i * 3], dtype=np.float32)
                timestamp = i * 10**6  # nanoseconds
                writer.append(data, Time(**{RECORDED_TIME: timestamp}))

        assert filepath.exists()

        reader = SimpleSignal(filepath)

        value, ts = reader.time[Time(**{RECORDED_TIME: 5_000_000})]
        np.testing.assert_array_equal(value, np.array([5, 10, 15], dtype=np.float32))
        assert ts == Time(**{RECORDED_TIME: 5_000_000})

        view = reader.time[
            Time(**{RECORDED_TIME: 0}) : Time(**{RECORDED_TIME: 9_000_001}) : Time(**{RECORDED_TIME: 1_000_000})
        ]
        assert len(view) == 10

        for i in range(10):
            value, ts = view[i]
            np.testing.assert_array_equal(value, np.array([i, i * 2, i * 3], dtype=np.float32))
            assert ts == Time(**{RECORDED_TIME: i * 1_000_000})

        assert filepath.stat().st_size > 0


class TestVectorInterface:
    def test_len_and_search_ts_empty(self, tmp_path):
        s = create_signal(tmp_path, [], 'empty.parquet')
        assert len(s) == 0
        empty = s._search_ts([Time(**{RECORDED_TIME: t}) for t in np.array([], dtype=np.int64)])
        assert isinstance(empty, np.ndarray)
        assert empty.size == 0

    def test_search_ts_numeric_and_invalid_dtype(self, tmp_path):
        s = create_signal(tmp_path, [(1, 1000), (2, 2000), (3, 3000)])
        # Select the last coordinate at or before each query.
        idx = s._search_ts([Time(**{RECORDED_TIME: t}) for t in np.array([500, 1500, 2500, 3500], dtype=np.int64)])
        assert np.array_equal(idx, np.array([-1, 0, 1, 2]))
        # Accept scalar float via list-like contract
        assert s._search_ts([Time(**{RECORDED_TIME: t}) for t in [1999]])[0] == 0
        # Reject non-numeric dtype
        with pytest.raises(TypeError):
            _ = s._search_ts([Time(**{RECORDED_TIME: t}) for t in np.array(['1000'], dtype=object)])

    def test_values_and_ts_at(self, tmp_path):
        s = create_signal(tmp_path, [(np.array([1, 2]), 1000), (np.array([3, 4]), 2000)])
        assert s._ts_at([1], (RECORDED_TIME,))[0] == Time(**{RECORDED_TIME: 2000})
        ts_arr = s._ts_at(np.array([0, 1], dtype=np.int64), (RECORDED_TIME,))
        assert list(ts_arr) == [Time(**{RECORDED_TIME: t}) for t in [1000, 2000]]
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
        assert sig_simple.bounds(RECORDED_TIME)[0][RECORDED_TIME] == 1000
        assert sig_simple.bounds(RECORDED_TIME)[1][RECORDED_TIME] == 5000

    def test_index_scalar_and_negative(self, sig_simple):
        assert sig_simple[0] == (10, Time(**{RECORDED_TIME: 1000}))
        assert sig_simple[2] == (30, Time(**{RECORDED_TIME: 3000}))
        assert sig_simple[-1] == (50, Time(**{RECORDED_TIME: 5000}))
        assert sig_simple[-5] == (10, Time(**{RECORDED_TIME: 1000}))
        with pytest.raises(IndexError):
            _ = sig_simple[5]
        with pytest.raises(IndexError):
            _ = sig_simple[-6]

    def test_index_slice(self, sig_simple):
        view = sig_simple[1:4]
        assert len(view) == 3
        assert view[0] == (20, Time(**{RECORDED_TIME: 2000}))
        assert view[2] == (40, Time(**{RECORDED_TIME: 4000}))
        step_view = sig_simple[0:5:2]
        assert len(step_view) == 3
        assert step_view[0] == (10, Time(**{RECORDED_TIME: 1000}))
        assert step_view[1] == (30, Time(**{RECORDED_TIME: 3000}))
        assert step_view[2] == (50, Time(**{RECORDED_TIME: 5000}))
        with pytest.raises(ValueError):
            _ = sig_simple[::0]
        with pytest.raises(ValueError):
            _ = sig_simple[::-1]

    def test_index_array(self, sig_simple):
        view = sig_simple[[0, 2, 4]]
        assert list(view) == [
            (10, Time(**{RECORDED_TIME: 1000})),
            (30, Time(**{RECORDED_TIME: 3000})),
            (50, Time(**{RECORDED_TIME: 5000})),
        ]

        with pytest.raises(ValueError):
            sig_simple[np.array([0, -1, 1, -2], dtype=np.int64)]
        with pytest.raises(IndexError):
            _ = sig_simple[np.array([0, 5], dtype=np.int64)]
        with pytest.raises(TypeError):
            _ = sig_simple[np.array([True, False, True, False, True], dtype=np.bool_)]
        with pytest.raises(TypeError):
            _ = sig_simple[np.array([0.0, 1.0], dtype=np.float64)]
        empty = sig_simple[[]]
        assert len(empty) == 0

    def test_index_numpy_integer_scalars(self, sig_simple):
        assert sig_simple[np.int64(2)] == (30, Time(**{RECORDED_TIME: 3000}))
        assert sig_simple[np.int32(0)] == (10, Time(**{RECORDED_TIME: 1000}))
        assert sig_simple[np.int64(-1)] == (50, Time(**{RECORDED_TIME: 5000}))


class TestCoreSignalTime:
    def test_time_scalar_cases(self, sig_simple):
        with pytest.raises(KeyError):
            _ = sig_simple.time[Time(**{RECORDED_TIME: 999})]
        assert sig_simple.time[Time(**{RECORDED_TIME: 1000})] == (10, Time(**{RECORDED_TIME: 1000}))
        assert sig_simple.time[Time(**{RECORDED_TIME: 2500})] == (20, Time(**{RECORDED_TIME: 2000}))
        with pytest.raises(TypeError):
            sig_simple.time[2500]
        assert sig_simple.time[Time(**{RECORDED_TIME: 9999})] == (50, Time(**{RECORDED_TIME: 5000}))

    def test_time_window_basic(self, sig_simple):
        view = sig_simple.time[Time(**{RECORDED_TIME: 1500}) : Time(**{RECORDED_TIME: 4500})]
        assert list(view) == [
            (10, Time(**{RECORDED_TIME: 1500})),
            (20, Time(**{RECORDED_TIME: 2000})),
            (30, Time(**{RECORDED_TIME: 3000})),
            (40, Time(**{RECORDED_TIME: 4000})),
        ]
        v2 = sig_simple.time[: Time(**{RECORDED_TIME: 3000})]
        assert len(v2) == 2
        v3 = sig_simple.time[Time(**{RECORDED_TIME: 3000}) :]
        assert len(v3) == 3
        with pytest.raises(ValueError):
            sig_simple.time[:]
        assert len(sig_simple.time[: Time(**{RECORDED_TIME: 900})]) == 0
        assert list(sig_simple.time[Time(**{RECORDED_TIME: 6000}) :]) == [(50, Time(**{RECORDED_TIME: 6000}))]

    def test_time_window_injects_start(self, sig_simple):
        v = sig_simple.time[Time(**{RECORDED_TIME: 1500}) : Time(**{RECORDED_TIME: 3500})]
        assert len(v) == 3
        assert v[0] == (10, Time(**{RECORDED_TIME: 1500}))
        assert v[1] == (20, Time(**{RECORDED_TIME: 2000}))
        assert v[2] == (30, Time(**{RECORDED_TIME: 3000}))

    def test_time_stepped_empty_signal(self):
        ts = np.array([], dtype=np.int64)
        vals = np.array([], dtype=np.int64)
        sig = DummySignal(ts, vals)
        sampled = sig.time[
            Time(**{RECORDED_TIME: 1000}) : Time(**{RECORDED_TIME: 5000}) : Time(**{RECORDED_TIME: 1000})
        ]
        assert len(sampled) == 0

    def test_time_window_no_inject_when_exact(self, sig_simple):
        v = sig_simple.time[Time(**{RECORDED_TIME: 2000}) : Time(**{RECORDED_TIME: 3500})]
        assert len(v) == 2
        assert v[0] == (20, Time(**{RECORDED_TIME: 2000}))
        assert v[1] == (30, Time(**{RECORDED_TIME: 3000}))

    def test_time_window_start_before_first_no_inject(self, sig_simple):
        assert list(sig_simple.time[Time(**{RECORDED_TIME: 500}) : Time(**{RECORDED_TIME: 2500})]) == [
            (10, Time(**{RECORDED_TIME: 1000})),
            (20, Time(**{RECORDED_TIME: 2000})),
        ]

    def test_time_window_start_before_first_injects_start(self, sig_simple):
        assert list(sig_simple.time[Time(**{RECORDED_TIME: 100}) : Time(**{RECORDED_TIME: 900})]) == []

    def test_time_stepped(self, sig_simple):
        sampled = sig_simple.time[
            Time(**{RECORDED_TIME: 1000}) : Time(**{RECORDED_TIME: 6000}) : Time(**{RECORDED_TIME: 2000})
        ]
        assert list(sampled) == [
            (10, Time(**{RECORDED_TIME: 1000})),
            (30, Time(**{RECORDED_TIME: 3000})),
            (50, Time(**{RECORDED_TIME: 5000})),
        ]
        with pytest.raises(ValueError):
            _ = sig_simple.time[: Time(**{RECORDED_TIME: 5000}) : Time(**{RECORDED_TIME: 1000})]
        with pytest.raises(ValueError):
            _ = sig_simple.time[
                Time(**{RECORDED_TIME: 1000}) : Time(**{RECORDED_TIME: 3000}) : Time(**{RECORDED_TIME: 0})
            ]
        with pytest.raises(ValueError):
            _ = sig_simple.time[
                Time(**{RECORDED_TIME: 1000}) : Time(**{RECORDED_TIME: 3000}) : Time(**{RECORDED_TIME: -1000})
            ]
        with pytest.raises(KeyError):
            _ = sig_simple.time[
                Time(**{RECORDED_TIME: 500}) : Time(**{RECORDED_TIME: 3000}) : Time(**{RECORDED_TIME: 1000})
            ]
        full = sig_simple.time[Time(**{RECORDED_TIME: 1000}) :: Time(**{RECORDED_TIME: 1000})]
        assert list(full) == [
            (10, Time(**{RECORDED_TIME: 1000})),
            (20, Time(**{RECORDED_TIME: 2000})),
            (30, Time(**{RECORDED_TIME: 3000})),
            (40, Time(**{RECORDED_TIME: 4000})),
            (50, Time(**{RECORDED_TIME: 5000})),
        ]

    def test_time_array_sampling(self, sig_simple):
        req = [1000, 1500, 3000]
        view = sig_simple.time[[Time(**{RECORDED_TIME: t}) for t in req]]
        assert list(view) == [
            (10, Time(**{RECORDED_TIME: 1000})),
            (10, Time(**{RECORDED_TIME: 1500})),
            (30, Time(**{RECORDED_TIME: 3000})),
        ]
        with pytest.raises(ValueError):
            sig_simple.time[[Time(**{RECORDED_TIME: t}) for t in [3000, 1000, 3000]]]


class TestCoreSignalViews:
    def test_index_then_index(self, sig_simple):
        v1 = sig_simple[1:4]
        v2 = v1[::2]
        assert len(v2) == 2
        assert v2[0] == (20, Time(**{RECORDED_TIME: 2000}))
        assert v2[1] == (40, Time(**{RECORDED_TIME: 4000}))

    def test_time_slice_then_index(self, sig_simple):
        v = sig_simple.time[Time(**{RECORDED_TIME: 1500}) : Time(**{RECORDED_TIME: 4500})]
        assert v[0] == (10, Time(**{RECORDED_TIME: 1500}))
        assert v[-1] == (40, Time(**{RECORDED_TIME: 4000}))
        vv = v[1:]
        assert len(vv) == 3
        assert vv[0] == (20, Time(**{RECORDED_TIME: 2000}))
        assert vv[1] == (30, Time(**{RECORDED_TIME: 3000}))
        assert vv[2] == (40, Time(**{RECORDED_TIME: 4000}))

    def test_time_sample_then_time_slice(self, sig_simple):
        sampled = sig_simple.time[[Time(**{RECORDED_TIME: t}) for t in [1500, 2500, 3500, 4500]]]
        sub = sampled.time[Time(**{RECORDED_TIME: 2500}) : Time(**{RECORDED_TIME: 4500})]
        assert list(sub) == [(20, Time(**{RECORDED_TIME: 2500})), (30, Time(**{RECORDED_TIME: 3500}))]

    def test_iteration_over_views(self, sig_simple):
        v = sig_simple.time[Time(**{RECORDED_TIME: 2000}) : Time(**{RECORDED_TIME: 5000})]
        items = list(v)
        assert items == [
            (20, Time(**{RECORDED_TIME: 2000})),
            (30, Time(**{RECORDED_TIME: 3000})),
            (40, Time(**{RECORDED_TIME: 4000})),
        ]

    def test_array_on_slice_indexing(self, sig_simple):
        v = sig_simple[1:5]
        sub = v[[0, 2]]
        assert list(sub) == [(20, Time(**{RECORDED_TIME: 2000})), (40, Time(**{RECORDED_TIME: 4000}))]

    def test_time_slice_of_time_slice(self, sig_simple):
        first = sig_simple.time[Time(**{RECORDED_TIME: 1500}) : Time(**{RECORDED_TIME: 4500})]
        second = first.time[Time(**{RECORDED_TIME: 2000}) : Time(**{RECORDED_TIME: 3500})]
        assert list(second) == [(20, Time(**{RECORDED_TIME: 2000})), (30, Time(**{RECORDED_TIME: 3000}))]

    def test_iter_over_stepped_sampling(self, sig_simple):
        sampled = sig_simple.time[
            Time(**{RECORDED_TIME: 1500}) : Time(**{RECORDED_TIME: 3500}) : Time(**{RECORDED_TIME: 1000})
        ]
        assert list(sampled) == [(10, Time(**{RECORDED_TIME: 1500})), (20, Time(**{RECORDED_TIME: 2500}))]

    def test_time_array_mixed_before_first(self, sig_simple):
        with pytest.raises(KeyError):
            _ = sig_simple.time[[Time(**{RECORDED_TIME: t}) for t in [500, 1000]]]

    def test_time_array_empty(self, sig_simple):
        view = sig_simple.time[[Time(**{RECORDED_TIME: t}) for t in []]]
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

    def test_signal_view_meta_inherits_and_empty_view_raises(self, tmp_path):
        s = create_signal(tmp_path, [(1, 1000), (2, 2000), (3, 3000)])
        view = s[1:3]
        assert view.kind == Kind.NUMERIC
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


class TestExtraTimelines:
    def test_simple_signal_writer_with_extra_timelines(self, tmp_path):
        """Test that SimpleSignalWriter stores extra timelines in separate columns."""
        fp = tmp_path / 'extra_timelines.parquet'
        with SimpleSignalWriter(fp) as w:
            w.append(10, Time(**{RECORDED_TIME: 1000}, producer=900, consumer=1100))
            w.append(20, Time(**{RECORDED_TIME: 2000}, producer=1900, consumer=2100))
            w.append(30, Time(**{RECORDED_TIME: 3000}, producer=2900, consumer=3100))

        table = pq.read_table(fp)
        assert {f'ts.{RECORDED_TIME}', 'value', 'ts.consumer', 'ts.producer'} == set(table.column_names)

        # Verify the data
        assert table[f'ts.{RECORDED_TIME}'].to_pylist() == [1000, 2000, 3000]
        assert table['value'].to_pylist() == [10, 20, 30]
        assert table['ts.producer'].to_pylist() == [900, 1900, 2900]
        assert table['ts.consumer'].to_pylist() == [1100, 2100, 3100]

    def test_simple_signal_empty_with_extra_timelines(self, tmp_path):
        """Test that empty signal writer with no appends doesn't create extra timeline columns."""
        fp = tmp_path / 'empty_extra.parquet'
        with SimpleSignalWriter(fp):
            pass

        table = pq.read_table(fp)
        assert {'value'} == set(table.column_names)
        assert len(table) == 0

    def test_inconsistent_timestamp_keys_raises(self, tmp_path):
        """Test that inconsistent timestamp keys across appends raises ValueError."""
        fp = tmp_path / 'inconsistent.parquet'
        with pytest.raises(ValueError, match='Timeline names must be consistent'):
            with SimpleSignalWriter(fp) as w:
                w.append(10, Time(**{RECORDED_TIME: 1000}, producer=900))
                w.append(20, Time(**{RECORDED_TIME: 2000}, producer=1900, consumer=2100))

    def test_missing_timestamp_after_first_raises(self, tmp_path):
        """Test that omitting timestamp after providing it first raises ValueError."""
        fp = tmp_path / 'missing.parquet'
        with pytest.raises(ValueError, match='Timeline names must be consistent'):
            with SimpleSignalWriter(fp) as w:
                w.append(10, Time(**{RECORDED_TIME: 1000}, producer=900))
                w.append(20, Time(**{RECORDED_TIME: 2000}))

    def test_adding_timestamp_after_none_raises(self, tmp_path):
        """Test that adding timestamp after first append without it raises ValueError."""
        fp = tmp_path / 'late_extra.parquet'
        with pytest.raises(ValueError, match='Timeline names must be consistent'):
            with SimpleSignalWriter(fp) as w:
                w.append(10, Time(**{RECORDED_TIME: 1000}))
                w.append(20, Time(**{RECORDED_TIME: 2000}, producer=1900))

    @pytest.mark.parametrize('chunk_size', [1, 2])
    @pytest.mark.parametrize('invalid', [{'world': 30}, {'world': 30, 'wall': 40, 'message': 50}])
    def test_timeline_names_remain_fixed_across_chunks(self, tmp_path, chunk_size, invalid):
        path = tmp_path / 'signal.parquet'
        with SimpleSignalWriter(path, chunk_size=chunk_size) as writer:
            writer.append(1, Time(world=10, wall=20))
            writer.append(2, Time(world=20, wall=30))
            with pytest.raises(ValueError, match='Timeline names must be consistent'):
                writer.append(3, Time(**invalid))
            writer.append(3, Time(world=30, wall=40))
        table = pq.read_table(path)
        assert table['ts.world'].to_pylist() == [10, 20, 30]
        assert table['ts.wall'].to_pylist() == [20, 30, 40]

    def test_invalid_timestamps_are_rejected_for_duplicate_values(self, tmp_path):
        with SimpleSignalWriter(tmp_path / 'signal.parquet', drop_equal_bytes_threshold=32) as writer:
            writer.append(1, Time(world=10, wall=20))
            with pytest.raises(ValueError, match='Timeline names must be consistent'):
                writer.append(1, Time(world=20))


def _make_signal(ts_sec, vals):
    """Create a DummySignal from timestamps in seconds and a 2D value array."""
    ts = (np.asarray(ts_sec, dtype=np.float64) * 1e9).astype(np.int64)
    return DummySignal(ts, np.asarray(vals, dtype=np.float64))


class TestDiff:
    def test_velocity_of_linear_is_constant(self):
        # f(t) = [t, 2t] → velocity = [1, 2]
        sig = _make_signal([0, 1, 2, 3, 4], [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]])
        vel = diff(sig, dt_sec=1.0, timelines=(RECORDED_TIME,))
        # Centered diff trims 1 from start; last point clamps so skip it
        vals = np.array(vel._values_at(np.arange(len(vel) - 1)))
        np.testing.assert_allclose(vals, [[1.0, 2.0]] * len(vals))

    def test_acceleration_of_quadratic_is_constant(self):
        # f(t) = [t²] → acceleration = [2]
        t = np.arange(7, dtype=np.float64)
        sig = _make_signal(t, (t**2).reshape(-1, 1))
        accel = diff(sig, dt_sec=1.0, order=2, timelines=(RECORDED_TIME,))
        # Trims 1 from start; last point clamps so skip it
        vals = np.array(accel._values_at(np.arange(len(accel) - 1)))
        np.testing.assert_allclose(vals, [[2.0]] * len(vals))

    def test_invalid_order_raises(self):
        sig = _make_signal([0, 1], [[0], [1]])
        with pytest.raises(ValueError, match='order'):
            diff(sig, dt_sec=1.0, order=3, timelines=(RECORDED_TIME,))


class TestNorm:
    def test_norm_2d(self):
        sig = _make_signal([0, 1, 2], [[3, 4], [6, 8], [0, 0]])
        n = norm(sig)
        vals = np.array(n._values_at(np.arange(3)))
        np.testing.assert_allclose(vals, [5.0, 10.0, 0.0])

    def test_norm_1d_is_abs(self):
        sig = _make_signal([0, 1, 2], [[-3], [0], [5]])
        n = norm(sig)
        vals = np.array(n._values_at(np.arange(3)))
        np.testing.assert_allclose(vals, [3.0, 0.0, 5.0])


@pytest.mark.parametrize('count', [0, 3])
def test_named_columns_and_empty_signal(tmp_path, count):
    path = tmp_path / 'named.parquet'
    with SimpleSignalWriter(path, chunk_size=1) as writer:
        for i in range(count):
            writer.append(i, Time(world=100 + i * 10, wall=1000 + i * 10))
    assert pq.read_schema(path).metadata[SIGNAL_VERSION_KEY] == SIGNAL_VERSION
    signal = SimpleSignal(path)
    assert signal.timelines == (('world', 'wall') if count else ())
    assert len(signal) == count
    if count:
        assert signal.time[Time(world=115)] == (1, Time(world=110, wall=1010))


def test_unnamed_legacy_file_queries(tmp_path):
    path = tmp_path / 'legacy.parquet'
    pq.write_table(pa.table({'timestamp': [100, 200], 'value': [1, 2]}), path)
    signal = SimpleSignal(path)
    assert signal.timelines == (RECORDED_TIME,)
    assert signal.time[Time(**{RECORDED_TIME: 150})] == (1, Time(**{RECORDED_TIME: 100}))
    assert signal.bounds(RECORDED_TIME) == (Time(**{RECORDED_TIME: 100}), Time(**{RECORDED_TIME: 200}))


class TestNamedTimelines:
    @pytest.fixture
    def signal(self):
        return DummySignal([[100, 900], [140, 950], [220, 1100]], [1, 2, 3], timelines=('A', 'B'))

    def test_point_lookup_and_name_order(self, signal):
        assert signal.time[{'A': 150}] == (2, Time(A=140, B=950))
        assert signal.time[{'A': 250, 'B': 960}] == signal.time[{'B': 960, 'A': 250}]
        assert signal.time[{'A': 250, 'B': 960}] == (2, Time(A=140, B=950))
        with pytest.raises(KeyError):
            signal.time[{'A': 99}]
        with pytest.raises(KeyError):
            signal.time[{'missing': 100}]
        with pytest.raises(ValueError):
            signal.time[{}]

    def test_repeated_subset_selects_last_record(self):
        signal = DummySignal([[100, 1000], [200, 1000], [300, 1000]], [1, 2, 3], timelines=('A', 'B'))
        assert signal.time[Time(B=1000)] == (3, Time(A=300, B=1000))
        assert list(signal.timestamps(('B',))) == [Time(B=1000)] * 3

    def test_sampling_retains_unqueried_coordinates(self, signal):
        sampled = signal.time[Time(A=100) : Time(A=251) : Time(A=50)]
        assert list(sampled) == [
            (1, Time(A=100, B=900)),
            (2, Time(A=150, B=950)),
            (2, Time(A=200, B=950)),
            (3, Time(A=250, B=1100)),
        ]
        assert sampled.time[Time(B=950)] == (2, Time(A=200, B=950))
        assert signal.time[Time(A=150)] == (2, Time(A=140, B=950))

    @pytest.mark.parametrize(
        'queries',
        [
            [Time(A=150), Time(A=100)],
            [Time(A=150), Time(A=150)],
            [Time(A=150, B=950), Time(A=200, B=940)],
            [Time(A=150), Time(B=950)],
        ],
    )
    def test_invalid_sampling_grid(self, signal, queries):
        with pytest.raises(ValueError):
            signal.time[queries]

    def test_empty_selections(self, signal):
        assert len(signal.time[[]]) == 0
        for names in [(), ('A', 'A')]:
            with pytest.raises(ValueError):
                signal.timestamps(names)
            with pytest.raises(ValueError):
                signal.bounds(names)
        with pytest.raises(ValueError):
            signal.time[:]
        assert list(signal[:]) == list(signal)

    def test_window_projection_and_order(self, signal):
        assert list(signal.time[Time(A=150) : Time(A=230)]) == [(2, Time(A=150, B=950)), (3, Time(A=220, B=1100))]
        with pytest.raises(ValueError):
            signal.time[Time(A=150, B=1200) :]

    def test_stepping_zero_coordinates_and_inclusive_end(self):
        signal = DummySignal([[100, 1000], [200, 1000]], [1, 2], timelines=('A', 'B'))
        assert list(
            signal.time[Time(A=100, B=1000) : Time(A=200, B=1000) : Time(A=50, B=0)].timestamps(('A', 'B'))
        ) == [Time(A=100, B=1000), Time(A=150, B=1000)]
        assert len(signal.time[Time(A=100, B=1000) :: Time(A=50, B=0)]) == 3
        for step in [Time(A=0), Time(A=-1)]:
            with pytest.raises(ValueError):
                signal.time[Time(A=100) :: step]

    def test_projection_bounds_and_readonly_rows(self, signal):
        assert signal.bounds(('B', 'A')) == (Time(B=900, A=100), Time(B=1100, A=220))
        times = signal.timestamps(('B', 'A'))
        assert times[0].timelines == ('B', 'A')
        with pytest.raises(TypeError):
            times[0]['A'] = 123
        assert signal[0][1] == Time(A=100, B=900)

    @pytest.mark.parametrize('image', [False, True])
    def test_single_name_bounds_and_timestamps(self, tmp_path, image):
        path = tmp_path / 'episode'
        value = np.zeros((16, 16, 3), dtype=np.uint8) if image else 1
        with DiskEpisodeWriter(path) as writer:
            writer.append('value', value, Time(world=100, wall=900))
            writer.append('value', value, Time(world=200, wall=950))
            writer.append('unrelated', 2, Time(tick=1))
        episode = DiskEpisode(path)
        signal = episode['value']
        expected = (Time(world=100), Time(world=200))
        for source in (signal, signal[:]):
            assert source.bounds('world') == source.bounds(('world',)) == expected
            assert list(source.timestamps('world')) == list(source.timestamps(('world',))) == list(expected)
            with pytest.raises(KeyError):
                source.bounds('missing')
            with pytest.raises(KeyError):
                source.timestamps('missing')
            for invalid in ('', '   '):
                with pytest.raises(ValueError):
                    source.bounds(invalid)
                with pytest.raises(ValueError):
                    source.timestamps(invalid)
        assert episode.bounds('world') == episode.bounds(('world',)) == expected
        with pytest.raises(ValueError):
            episode.bounds('')

    def test_join_subset_and_conflicting_order(self, signal):

        other = DummySignal([[100, 1], [170, 2]], [4, 5], timelines=('A', 'C'))
        joined = Join(signal, other, timelines=('A',))
        assert joined.timelines == ('A',)
        assert list(joined) == [
            ((1, 4), Time(A=100)),
            ((2, 4), Time(A=140)),
            ((2, 5), Time(A=170)),
            ((3, 5), Time(A=220)),
        ]
        crossing = DummySignal([[100, 900], [180, 940]], [4, 5], timelines=('A', 'B'))
        for names in [('A', 'B'), ('B', 'A')]:
            with pytest.raises(ValueError):
                list(Join(signal, crossing, timelines=names))

    def test_episode_filters_and_common_grid(self, signal):

        other = DummySignal([[100, 1], [170, 2]], [4, 5], timelines=('A', 'C'))
        disjoint = DummySignal([[1]], [6], timelines=('D',))
        episode = EpisodeContainer({'left': signal, 'right': other, 'unrelated': disjoint, 'task': 'test'})
        assert episode.time[Time(A=150)] == {'left': 2, 'right': 4, 'task': 'test'}
        assert episode.time[Time(B=950)] == {'left': 2, 'task': 'test'}
        assert episode.time[Time(B=950, C=1)] == {'task': 'test'}
        assert episode.bounds(('A',)) == (Time(A=100), Time(A=220))
        samples = episode.time[Time(A=100) :: Time(A=50)]
        assert list(samples['left']) == [1, 2, 2]
        assert list(samples['right']) == [4, 4, 5]
        with pytest.raises(KeyError):
            episode.time[Time(A=99)]
        with pytest.raises(ValueError):
            episode.bounds(('missing',))

    @pytest.mark.parametrize('image', [False, True])
    def test_storage_roundtrip_and_order(self, tmp_path, image):

        path = tmp_path / 'episode'
        value = np.zeros((16, 16, 3), dtype=np.uint8) if image else 1
        with DiskEpisodeWriter(path) as writer:
            writer.append('value', value, Time(A=100, B=1000))
            writer.append('value', value, Time(B=1000, A=200))
            for invalid in [Time(A=200, B=1000), Time(A=300, B=999), Time(A=300), Time(A=300, B=1001, C=1)]:
                with pytest.raises(ValueError):
                    writer.append('value', value, invalid)
            writer.append('value', value, Time(A=200, B=1001))
        signal = DiskEpisode(path)['value']
        assert signal.timelines == ('A', 'B')
        assert list(signal.timestamps(signal.timelines)) == [
            Time(A=100, B=1000),
            Time(A=200, B=1000),
            Time(A=200, B=1001),
        ]
        assert signal.time[Time(A=200)][1] == Time(A=200, B=1001)
        schema = pq.read_schema(path / ('value.frames.parquet' if image else 'value.parquet'))
        assert schema.metadata[SIGNAL_VERSION_KEY] == SIGNAL_VERSION
        for name in ('ts.A', 'ts.B'):
            assert not schema.field(name).nullable

    def test_deduplication_checks_every_attempt(self, tmp_path):
        path = tmp_path / 'dedup.parquet'
        with SimpleSignalWriter(path, drop_equal_bytes_threshold=32) as writer:
            writer.append(1, Time(A=100, B=1000))
            writer.append(1, Time(A=200, B=1000))
            with pytest.raises(ValueError):
                writer.append(2, Time(A=150, B=1001))
            writer.append(2, Time(A=200, B=1001))
        assert len(SimpleSignal(path)) == 2

    def test_legacy_columns_and_unknown_version(self, tmp_path):

        path = tmp_path / 'old.parquet'
        table = pa.table({'timestamp': [100, 200], 'value': [1, 2], 'ts_ns.server': [50, 10]})
        pq.write_table(table.replace_schema_metadata({TIMELINE_METADATA_KEY: b'world'}), path)
        signal = SimpleSignal(path)
        assert signal.timelines == ('world',)
        assert signal.time[Time(world=150)] == (1, Time(world=100))
        pq.write_table(table.replace_schema_metadata({SIGNAL_VERSION_KEY: b'999'}), path)
        with pytest.raises(ValueError, match='Unsupported signal format'):
            _ = SimpleSignal(path).timelines


def test_window_without_start_keeps_repeated_projected_coordinates():
    signal = DummySignal([[100, 1000], [200, 1000], [300, 1100]], [1, 2, 3], timelines=('A', 'B'))
    assert list(signal.time[: Time(B=1100)]) == [(1, Time(A=100, B=1000)), (2, Time(A=200, B=1000))]


def test_native_literal_names_and_integer_query_limits(tmp_path):
    names = ('server.wall', 'server/wall')
    times = [Time(**dict.fromkeys(names, value)) for value in [-(2**63), 2**63 - 2, 2**63 - 1]]
    path = tmp_path / 'limits.parquet'
    with SimpleSignalWriter(path, chunk_size=1) as writer:
        for value, ts in enumerate(times):
            writer.append(value, ts)
    signal = SimpleSignal(path)
    assert signal.timelines == names
    assert signal.bounds(names) == (times[0], times[-1])
    assert signal.time[times[1]] == (1, times[1])
    assert signal.time[Time(**dict.fromkeys(names, 2**100))] == (2, times[-1])
    with pytest.raises(KeyError):
        signal.time[Time(**dict.fromkeys(names, -(2**100)))]
