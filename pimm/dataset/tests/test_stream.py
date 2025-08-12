import numpy as np
import pyarrow as pa
import pytest

from pimm.dataset.vector import SimpleSignal, SimpleSignalWriter


def create_signal(tmp_path, data_timestamps, name="test.parquet"):
    """Helper to create a Signal with data and timestamps."""
    filepath = tmp_path / name
    writer = SimpleSignalWriter(filepath)
    for data, ts in data_timestamps:
        writer.append(data, ts)
    writer.finish()
    return SimpleSignal(filepath)


def write_data(tmp_path, data_timestamps, name="test.parquet"):
    """Helper to write data and return filepath."""
    filepath = tmp_path / name
    writer = SimpleSignalWriter(filepath)
    for data, ts in data_timestamps:
        writer.append(data, ts)
    writer.finish()
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

    def test_time_vector_data(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.0, 2.0, 3.0]), 1000), (np.array([4.0, 5.0, 6.0]), 2000)])
        value, ts = signal.time[1500]
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0])
        assert ts == 1000


class TestSignalTimeWindow:
    def test_time_window_empty_Signal(self, tmp_path):
        signal = create_signal(tmp_path, [], "empty.parquet")
        view = signal.time[1000:2000]
        assert len(view) == 0

    def test_time_window_no_data_in_range(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000)])
        view = signal.time[3000:4000]
        assert len(view) == 0

    def test_time_window_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        view = signal.time[1000:3000]  # End is exclusive
        assert len(view) == 2
        assert view[0] == (42, 1000)
        assert view[1] == (43, 2000)

    def test_time_window_partial_range_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000), (45, 4000)])
        view = signal.time[1500:3500]
        assert len(view) == 2
        assert view[0] == (43, 2000)
        assert view[1] == (44, 3000)

    def test_time_window_vector_data(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.0, 2.0]), 1000), (np.array([3.0, 4.0]), 2000), (np.array([5.0, 6.0]), 3000)])
        view = signal.time[1000:2500]
        assert len(view) == 2
        value0, ts0 = view[0]
        np.testing.assert_array_equal(value0, [1.0, 2.0])
        assert ts0 == 1000
        value1, ts1 = view[1]
        np.testing.assert_array_equal(value1, [3.0, 4.0])
        assert ts1 == 2000

    def test_time_window_single_point(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        view = signal.time[2000:2001]  # Very small window
        assert len(view) == 1
        assert view[0] == (43, 2000)


class TestSignalWriterAppend:
    def test_append_increasing_timestamps(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert len(signal) == 3
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_append_non_increasing_timestamp_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(42, 1000)
        with pytest.raises(ValueError, match="is not increasing"):
            writer.append(43, 1000)
        with pytest.raises(ValueError, match="is not increasing"):
            writer.append(43, 999)

    def test_append_after_finish_raises(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(42, 1000)
        writer.finish()
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


class TestSignalWriterFinish:
    def test_finish_empty_writer(self, tmp_path):
        filepath = write_data(tmp_path, [])
        assert filepath.exists()
        signal = SimpleSignal(filepath)
        assert len(signal) == 0

    def test_finish_idempotent(self, tmp_path):
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        writer.append(42, 1000)
        writer.finish()
        writer.finish()  # Should not raise
        signal = SimpleSignal(tmp_path / "test.parquet")
        assert signal.time[1000] == (42, 1000)

    def test_finish_creates_file(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleSignalWriter(filepath)
        writer.append(42, 1000)
        assert not filepath.exists()
        writer.finish()
        assert filepath.exists()

    def test_finish_preserves_data_scalar(self, tmp_path):
        signal = create_signal(tmp_path, [(42, 1000), (43, 2000), (44, 3000)])
        assert signal[0] == (42, 1000)
        assert signal[1] == (43, 2000)
        assert signal[2] == (44, 3000)

    def test_finish_preserves_data_vector(self, tmp_path):
        signal = create_signal(tmp_path, [(np.array([1.5, 2.5]), 1000), (np.array([3.5, 4.5]), 2000)])
        value0, ts0 = signal[0]
        value1, ts1 = signal[1]
        np.testing.assert_array_equal(value0, [1.5, 2.5])
        np.testing.assert_array_equal(value1, [3.5, 4.5])
        assert ts0 == 1000
        assert ts1 == 2000
