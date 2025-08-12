import pytest
import numpy as np
import pyarrow as pa
from pathlib import Path
import tempfile
import shutil
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector import SimpleStream, SimpleStreamWriter


class TestStreamAt:
    def test_at_empty_stream(self, tmp_path):
        filepath = tmp_path / "empty.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.finish()
        
        stream = SimpleStream(filepath)
        assert stream.at(1000) is None
    
    def test_at_before_first_timestamp(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        assert stream.at(999) is None
    
    def test_at_exact_timestamp_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        value, ts = stream.at(1000)
        assert value == 42
        assert ts == 1000
    
    def test_at_between_timestamps_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        value, ts = stream.at(1500)
        assert value == 42
        assert ts == 1000
        
        value, ts = stream.at(2500)
        assert value == 43
        assert ts == 2000
    
    def test_at_after_last_timestamp_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        value, ts = stream.at(3000)
        assert value == 43
        assert ts == 2000
    
    def test_at_vector_data(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1.0, 2.0, 3.0]), 1000)
        writer.append(np.array([4.0, 5.0, 6.0]), 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        value, ts = stream.at(1500)
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0])
        assert ts == 1000


class TestStreamWindow:
    def test_window_empty_stream(self, tmp_path):
        filepath = tmp_path / "empty.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(1000, 2000)
        assert len(values) == 0
        assert len(timestamps) == 0
    
    def test_window_no_data_in_range(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(3000, 4000)
        assert len(values) == 0
        assert len(timestamps) == 0
    
    def test_window_inclusive_bounds_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(1000, 3000)
        np.testing.assert_array_equal(values, [42, 43, 44])
        np.testing.assert_array_equal(timestamps, [1000, 2000, 3000])
    
    def test_window_partial_range_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.append(45, 4000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(1500, 3500)
        np.testing.assert_array_equal(values, [43, 44])
        np.testing.assert_array_equal(timestamps, [2000, 3000])
    
    def test_window_vector_data(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1.0, 2.0]), 1000)
        writer.append(np.array([3.0, 4.0]), 2000)
        writer.append(np.array([5.0, 6.0]), 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(1000, 2500)
        assert len(values) == 2
        np.testing.assert_array_equal(values[0], [1.0, 2.0])
        np.testing.assert_array_equal(values[1], [3.0, 4.0])
        np.testing.assert_array_equal(timestamps, [1000, 2000])
    
    def test_window_single_point(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(2000, 2000)
        np.testing.assert_array_equal(values, [43])
        np.testing.assert_array_equal(timestamps, [2000])


class TestStreamWriterAppend:
    def test_append_increasing_timestamps(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(0, 10000)
        np.testing.assert_array_equal(values, [42, 43, 44])
        np.testing.assert_array_equal(timestamps, [1000, 2000, 3000])
    
    def test_append_non_increasing_timestamp_raises(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        
        with pytest.raises(ValueError, match="is not increasing"):
            writer.append(43, 1000)
        
        with pytest.raises(ValueError, match="is not increasing"):
            writer.append(43, 999)
    
    def test_append_after_finish_raises(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.finish()
        
        with pytest.raises(RuntimeError, match="Cannot append to a finished writer"):
            writer.append(43, 2000)
    
    def test_append_vector_shape_consistency(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1.0, 2.0, 3.0]), 1000)
        
        with pytest.raises(ValueError, match="shape"):
            writer.append(np.array([4.0, 5.0]), 2000)
    
    def test_append_vector_dtype_consistency(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1.0, 2.0], dtype=np.float32), 1000)
        
        with pytest.raises(ValueError, match="dtype"):
            writer.append(np.array([3, 4], dtype=np.int32), 2000)
    
    def test_append_scalar_type_consistency(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        
        with pytest.raises(ValueError, match="type"):
            writer.append(43.5, 2000)
    
    def test_append_numpy_arrays(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1, 2, 3]), 1000)
        writer.append(np.array([4, 5, 6]), 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, _ = stream.window(0, 3000)
        np.testing.assert_array_equal(values[0], [1, 2, 3])
        np.testing.assert_array_equal(values[1], [4, 5, 6])
    
    def test_append_pyarrow_arrays(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(pa.array([1.0, 2.0]), 1000)
        writer.append(pa.array([3.0, 4.0]), 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, _ = stream.window(0, 3000)
        np.testing.assert_array_equal(values[0], [1.0, 2.0])
        np.testing.assert_array_equal(values[1], [3.0, 4.0])
    
    def test_append_lists_as_vectors(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append([1, 2, 3], 1000)
        writer.append([4, 5, 6], 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, _ = stream.window(0, 3000)
        np.testing.assert_array_equal(values[0], [1, 2, 3])
        np.testing.assert_array_equal(values[1], [4, 5, 6])


class TestStreamWriterFinish:
    def test_finish_empty_writer(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.finish()
        
        assert filepath.exists()
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(0, 10000)
        assert len(values) == 0
        assert len(timestamps) == 0
    
    def test_finish_idempotent(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.finish()
        writer.finish()  # Should not raise
        
        stream = SimpleStream(filepath)
        value, ts = stream.at(1000)
        assert value == 42
        assert ts == 1000
    
    def test_finish_creates_file(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        
        assert not filepath.exists()
        writer.finish()
        assert filepath.exists()
    
    def test_finish_preserves_data_scalar(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(42, 1000)
        writer.append(43, 2000)
        writer.append(44, 3000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(0, 10000)
        np.testing.assert_array_equal(values, [42, 43, 44])
        np.testing.assert_array_equal(timestamps, [1000, 2000, 3000])
    
    def test_finish_preserves_data_vector(self, tmp_path):
        filepath = tmp_path / "test.parquet"
        writer = SimpleStreamWriter(filepath)
        writer.append(np.array([1.5, 2.5]), 1000)
        writer.append(np.array([3.5, 4.5]), 2000)
        writer.finish()
        
        stream = SimpleStream(filepath)
        values, timestamps = stream.window(0, 10000)
        np.testing.assert_array_equal(values[0], [1.5, 2.5])
        np.testing.assert_array_equal(values[1], [3.5, 4.5])
        np.testing.assert_array_equal(timestamps, [1000, 2000])