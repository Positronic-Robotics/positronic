import numpy as np
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


class TestVectorMeta:

    def test_vector_start_last_ts_basic(self, tmp_path):
        fp = tmp_path / "sig.parquet"
        with SimpleSignalWriter(fp) as w:
            w.append(1, 1000)
            w.append(2, 2000)
            w.append(3, 3000)
        s = SimpleSignal(fp)
        assert s.start_ts == 1000
        assert s.last_ts == 3000

    def test_vector_start_last_ts_empty_raises(self, tmp_path):
        fp = tmp_path / "empty.parquet"
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
        writer = SimpleSignalWriter(tmp_path / "test.parquet")
        with writer:
            writer.append(42, 1000)
            with pytest.raises(ValueError, match="is not increasing"):
                writer.append(43, 1000)
            with pytest.raises(ValueError, match="is not increasing"):
                writer.append(43, 999)


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

    def test_simple_writer_abort_removes_file_and_blocks_usage(self, tmp_path):
        fp = tmp_path / "abort.parquet"
        with SimpleSignalWriter(fp) as w:
            w.append(1, 1000)
            w.abort()
            assert not fp.exists()
        with pytest.raises(RuntimeError):
            w.append(2, 2000)


class TestVectorInterface:

    def test_len_and_search_ts_empty(self, tmp_path):
        s = create_signal(tmp_path, [], "empty.parquet")
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
            _ = s._search_ts(np.array(["1000"], dtype=object))

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
