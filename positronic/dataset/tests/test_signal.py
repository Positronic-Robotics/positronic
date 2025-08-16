import numpy as np
import pyarrow as pa
import pytest

from positronic.dataset.vector import SimpleSignal, SimpleSignalWriter


def create_signal(tmp_path, data_timestamps, name="test.parquet"):
    """Helper to create a Signal with data and timestamps."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath) as writer:
        for data, ts in data_timestamps:
            writer.append(data, ts)
    return SimpleSignal(filepath)


def write_data(tmp_path, data_timestamps, name="test.parquet"):
    """Helper to write data and return filepath."""
    filepath = tmp_path / name
    with SimpleSignalWriter(filepath) as writer:
        for data, ts in data_timestamps:
            writer.append(data, ts)
    return filepath


class TestSignalTimeAccess:

    def test_time_empty_Signal(self, tmp_path):
        signal = create_signal(tmp_path, [], "empty.parquet")
        with pytest.raises(KeyError):
            signal.time[1000]

    def test_time_before_first_timestamp(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        with pytest.raises(KeyError):
            signal.time[999]

    def test_time_exact_timestamp_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        assert signal.time[1000] == (42, 1000)

    def test_time_between_timestamps_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal.time[1500] == (42, 1000)
        assert signal.time[2500] == (43, 2000)

    def test_time_after_last_timestamp_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        assert signal.time[3000] == (43, 2000)

    def test_vector_start_last_ts_basic(self, tmp_path):
        fp = tmp_path / "sig.parquet"
        w = SimpleSignalWriter(fp)
        w.append(1, 1000)
        w.append(2, 2000)
        w.append(3, 3000)
        w.finish()

        s = SimpleSignal(fp)
        assert s.start_ts == 1000
        assert s.last_ts == 3000

    def test_vector_start_last_ts_empty_raises(self, tmp_path):
        fp = tmp_path / "empty.parquet"
        w = SimpleSignalWriter(fp)
        w.finish()
        s = SimpleSignal(fp)
        with pytest.raises(ValueError):
            _ = s.start_ts
        with pytest.raises(ValueError):
            _ = s.last_ts

    def test_time_vector_data(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.0, 2.0, 3.0]), 1000), (np.array([4.0, 5.0, 6.0]), 2000)])
        value, ts = signal.time[1500]
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0])
        assert ts == 1000


class TestSignalIndexAccess:

    def test_index_access_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_index_access_negative(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal[-1] == (44, 3000)
        assert signal[-2] == (43, 2000)
        assert signal[-3] == (42, 1000)

    def test_index_out_of_range(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        with pytest.raises(IndexError):
            signal[2]
        with pytest.raises(IndexError):
            signal[-3]

    def test_index_slice(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal[1:3]
        assert len(view) == 2
        assert view[0] == (43, 2000)
        assert view[1] == (44, 3000)

    def test_index_slice_negative_indices(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal[-2:]
        assert len(view) == 2
        assert view[0] == (44, 3000)
        assert view[1] == (45, 4000)

    def test_index_slice_step_not_supported(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000), (46, 5000)])
        view = signal[0:5:2]
        assert len(view) == 3
        assert view[0] == (42, 1000)
        assert view[1] == (44, 3000)
        assert view[2] == (46, 5000)

    def test_index_slice_zero_or_negative_step(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        with pytest.raises(ValueError):
            _ = signal[0:3:-1]
        with pytest.raises(ValueError):
            _ = signal[2:0:-1]

    def test_nested_view_slicing(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000), (46, 5000)])
        view1 = signal[1:4]  # Contains (43, 2000), (44, 3000), (45, 4000)
        view2 = view1[1:]  # Should contain (44, 3000), (45, 4000)
        assert len(view2) == 2
        assert view2[0] == (44, 3000)
        assert view2[1] == (45, 4000)


class TestSignalTimeWindow:

    def test_time_window_empty_Signal(self, tmp_path):
        signal = create_signal(tmp_path, [], "empty.parquet")
        sampled = signal.time[1000:2000:500]
        assert len(sampled) == 0

    def test_time_window_no_data_in_range(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        # Start before first timestamp is not supported for stepped access
        with pytest.raises(KeyError):
            _ = signal.time[500:1500:500]

    def test_time_window_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        sampled = signal.time[1000:3000:1000]  # End is exclusive
        assert len(sampled) == 2
        assert sampled[0] == (42, 1000)
        assert sampled[1] == (43, 2000)

    def test_time_window_partial_range_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        sampled = signal.time[2000:4000:1000]
        assert len(sampled) == 2
        assert sampled[0] == (43, 2000)
        assert sampled[1] == (44, 3000)

    def test_time_window_vector_data(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.0, 2.0]), 1000), (np.array([3.0, 4.0]), 2000),
                                          (np.array([5.0, 6.0]), 3000)])
        sampled = signal.time[1000:3000:1000]
        assert len(sampled) == 2
        value0, ts0 = sampled[0]
        np.testing.assert_array_equal(value0, [1.0, 2.0])
        assert ts0 == 1000
        value1, ts1 = sampled[1]
        np.testing.assert_array_equal(value1, [3.0, 4.0])
        assert ts1 == 2000

    def test_time_window_single_point(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        sampled = signal.time[2000:2001:1]
        assert len(sampled) == 1
        assert sampled[0] == (43, 2000)


class TestSignalTimeStepped:

    def test_time_stepped_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000), (46, 5000)])
        sampled = signal.time[1000:5000:1000]
        assert len(sampled) == 4
        assert sampled[0] == (42, 1000)
        assert sampled[1] == (43, 2000)
        assert sampled[2] == (44, 3000)
        assert sampled[3] == (45, 4000)

    def test_time_stepped_between_timestamps(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 3000), (44, 5000)])
        sampled = signal.time[1000:5000:2000]
        # At 1000: exact match -> 42
        # At 3000: exact match -> 43
        assert len(sampled) == 2
        assert sampled[0] == (42, 1000)
        assert sampled[1] == (43, 3000)

    def test_time_stepped_before_first_timestamp(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 2000), (43, 3000), (44, 4000)])
        # Start before first timestamp is not supported for stepped access
        with pytest.raises(KeyError):
            _ = signal.time[1000:5000:1000]

    def test_time_stepped_empty_signal(self, tmp_path):
        signal = create_signal(tmp_path, [], "empty.parquet")
        sampled = signal.time[1000:5000:1000]
        assert len(sampled) == 0

    def test_time_stepped_zero_step(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        with pytest.raises(ValueError):
            _ = signal.time[1000:2000:0]

    def test_time_stepped_negative_step(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        with pytest.raises(ValueError):
            _ = signal.time[1000:2000:-1000]

    def test_time_stepped_vector_data(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.0, 2.0]), 1000), (np.array([3.0, 4.0]), 2000),
                                          (np.array([5.0, 6.0]), 3000)])
        sampled = signal.time[1500:3000:1000]
        assert len(sampled) == 2
        value0, ts0 = sampled[0]
        value1, ts1 = sampled[1]
        np.testing.assert_array_equal(value0, [1.0, 2.0])  # At 1500, gets value from 1000
        np.testing.assert_array_equal(value1, [3.0, 4.0])  # At 2500, gets value from 2000
        assert ts0 == 1500
        assert ts1 == 2500

    def test_time_stepped_view_consistency(self, tmp_path):
        # Test that stepped views maintain Signal interface
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        sampled = signal.time[1000:3000:1000]

        assert len(sampled) == 2
        assert sampled[0] == (42, 1000)
        assert sampled[1] == (43, 2000)
        assert sampled[-1] == (43, 2000)

    def test_time_stepped_missing_stop(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)])
        sampled = signal.time[1000::1000]
        assert len(sampled) == 5
        assert sampled[0] == (10, 1000)
        assert sampled[-1] == (50, 5000)

    def test_time_stepped_missing_start_raises(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(ValueError):
            _ = signal.time[:3000:1000]

    def test_time_stepped_missing_both_raises(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(ValueError):
            _ = signal.time[::1000]

    def test_time_stepped_returns_requested_timestamps_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        # Request off-grid timestamps 1500 and 2500
        sampled = signal.time[1500:3500:1000]
        assert len(sampled) == 2
        assert sampled[0] == (10, 1500)  # carry-back value, requested timestamp
        assert sampled[1] == (20, 2500)


class TestSignalWriterAppend:

    def test_append_increasing_timestamps(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert len(signal) == 3
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_append_non_increasing_timestamp_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        with writer:
            writer.append(42, 1000)
            with pytest.raises(ValueError, match="is not increasing"):
                writer.append(43, 1000)
            with pytest.raises(ValueError, match="is not increasing"):
                writer.append(43, 999)

    def test_append_after_context_exit_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        with writer:
            writer.append(42, 1000)
        with pytest.raises(RuntimeError, match="Cannot append to a finished writer"):
            writer.append(43, 2000)

    def test_append_vector_shape_consistency(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(np.array([1.0, 2.0, 3.0]), 1000)
        with pytest.raises(ValueError, match="shape"):
            writer.append(np.array([4.0, 5.0]), 2000)

    def test_append_vector_dtype_consistency(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(np.array([1.0, 2.0], dtype=np.float32), 1000)
        with pytest.raises(ValueError, match="dtype"):
            writer.append(np.array([3, 4], dtype=np.int32), 2000)

    def test_append_scalar_type_consistency(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(42, 1000)
        with pytest.raises(ValueError, match="type"):
            writer.append(43.5, 2000)

    def test_append_numpy_arrays(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1, 2, 3]), 1000), (np.array([4, 5, 6]), 2000)])
        value0, _ = signal[0]
        value1, _ = signal[1]
        np.testing.assert_array_equal(value0, [1, 2, 3])
        np.testing.assert_array_equal(value1, [4, 5, 6])

    def test_append_pyarrow_arrays(self, tmp_path):
        signal = create_signal(tmp_path, [(pa.array([1.0, 2.0]), 1000), (pa.array([3.0, 4.0]), 2000)])
        value0, _ = signal[0]
        value1, _ = signal[1]
        np.testing.assert_array_equal(value0, [1.0, 2.0])
        np.testing.assert_array_equal(value1, [3.0, 4.0])

    def test_append_lists_as_vectors(self, tmp_path):
        signal = create_signal(tmp_path, [([1, 2, 3], 1000), ([4, 5, 6], 2000)])
        value0, _ = signal[0]
        value1, _ = signal[1]
        np.testing.assert_array_equal(value0, [1, 2, 3])
        np.testing.assert_array_equal(value1, [4, 5, 6])


class TestSignalWriterContext:

    def test_context_empty_writer(self, tmp_path):
        filepath = write_data(tmp_path, [])
        assert filepath.exists()
        signal = SimpleSignal(filepath)
        assert len(signal) == 0

    def test_context_writes_data(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        with SimpleSignalWriter(filepath) as writer:
            writer.append(42, 1000)
        signal = SimpleSignal(filepath)
        assert signal.time[1000] == (42, 1000)

    def test_context_creates_file(self, tmp_path):
        filepath = tmp_path / "test.parquet"
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


class TestSignalIndexArrayAccess:

    def test_index_array_basic(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000), (40, 4000)])
        view = signal[[0, 2]]
        assert len(view) == 2
        assert view[0] == (10, 1000)
        assert view[1] == (30, 3000)

    def test_index_array_numpy_and_negative(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000), (40, 4000)])
        view = signal[np.array([-1, -3, 1], dtype=np.int64)]
        assert len(view) == 3
        assert view[0] == (40, 4000)
        assert view[1] == (20, 2000)
        assert view[2] == (20, 2000)

    def test_index_array_out_of_range_raises(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(IndexError):
            _ = signal[np.array([0, 3], dtype=np.int64)]

    def test_index_array_boolean_not_supported(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(IndexError):
            _ = signal[np.array([True, False, True], dtype=np.bool_)]

    def test_index_array_invalid_dtype_raises(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(TypeError):
            _ = signal[np.array([0.0, 1.0], dtype=np.float64)]

    def test_index_array_empty(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        view = signal[[]]
        assert len(view) == 0


class TestSignalTimeArrayAccess:

    def test_time_array_exact_and_offgrid(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        req = [1000, 1500, 3000]
        view = signal.time[req]
        assert len(view) == 3
        v0, t0 = view[0]
        v1, t1 = view[1]
        v2, t2 = view[2]
        assert (v0, t0) == (10, 1000)
        assert (v1, t1) == (10, 1500)
        assert (v2, t2) == (30, 3000)

    def test_time_array_unsorted_and_repeated(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        req = [3000, 1000, 3000]
        view = signal.time[req]
        assert len(view) == 3
        assert view[0] == (30, 3000)
        assert view[1] == (10, 1000)
        assert view[2] == (30, 3000)

    def test_time_array_before_first_raises(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(KeyError):
            _ = signal.time[[500, 1000]]

    def test_time_array_empty(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(TypeError):
            _ = signal.time[[]]

    def test_time_array_float_inputs_cast(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        with pytest.raises(TypeError):
            _ = signal.time[np.array([1000.0, 2500.0], dtype=np.float64)]


class TestSignalTimeWindowNoStep:

    def test_time_window_nostep_basic_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[1500:3500]
        assert len(view) == 2
        assert view[0] == (43, 2000)
        assert view[1] == (44, 3000)

    def test_time_window_nostep_inclusive_start_exclusive_end(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[1000:3000]
        assert len(view) == 2
        assert view[0] == (42, 1000)
        assert view[1] == (43, 2000)

    def test_time_window_nostep_outside_bounds(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        full_view = signal.time[500:5000]
        assert len(full_view) == 4
        assert full_view[0] == (42, 1000)
        assert full_view[-1] == (45, 4000)

        empty1 = signal.time[100:900]
        assert len(empty1) == 0
        empty2 = signal.time[4500:5000]
        assert len(empty2) == 0

    def test_time_window_nostep_vector_data(self, tmp_path):
        signal = create_signal(
            tmp_path,
            [
                (np.array([1.0, 2.0]), 1000),
                (np.array([3.0, 4.0]), 2000),
                (np.array([5.0, 6.0]), 3000),
            ],
        )
        view = signal.time[1500:3000]
        assert len(view) == 1
        value, ts = view[0]
        np.testing.assert_array_equal(value, [3.0, 4.0])
        assert ts == 2000

    def test_time_window_nostep_empty_signal(self, tmp_path):
        signal = create_signal(tmp_path, [], "empty.parquet")
        view = signal.time[1000:2000]
        assert len(view) == 0

    def test_time_window_nostep_missing_start(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[:3000]
        assert len(view) == 2
        assert view[0] == (42, 1000)
        assert view[1] == (43, 2000)

    def test_time_window_nostep_missing_stop(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[2000:]
        assert len(view) == 3
        assert view[0] == (43, 2000)
        assert view[-1] == (45, 4000)

    def test_time_window_nostep_full_slice(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[:]
        assert len(view) == 4
        assert view[0] == (42, 1000)
        assert view[-1] == (45, 4000)

    def test_time_window_nostep_missing_bounds_outside(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        empty1 = signal.time[:900]
        assert len(empty1) == 0
        empty2 = signal.time[3000:]
        assert len(empty2) == 0


class TestSignalIteration:

    def test_iter_over_signal(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        items = list(signal)
        assert items == [(10, 1000), (20, 2000), (30, 3000)]

    def test_iter_over_index_slice(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000), (40, 4000)])
        view = signal[1:3]
        items = list(view)
        assert items == [(20, 2000), (30, 3000)]

    def test_iter_over_time_slice(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000), (40, 4000)])
        view = signal.time[1500:3500]
        items = list(view)
        assert items == [(20, 2000), (30, 3000)]

    def test_iter_over_time_stepped(self, tmp_path):
        signal = create_signal(tmp_path, [(10, 1000), (20, 2000), (30, 3000)])
        sampled = signal.time[1500:3500:1000]
        items = list(sampled)
        assert items == [(10, 1500), (20, 2500)]
