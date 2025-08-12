#!/usr/bin/env python3
"""Test script to verify chunked writing works with large datasets."""

import tempfile
import time
from pathlib import Path

import numpy as np

from pimm.dataset.vector import SimpleSignal, SimpleSignalWriter


def test_large_dataset_chunked_writing():
    """Test that chunked writing handles large datasets efficiently."""

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "large_test.parquet"

        chunk_size, num_records = 1000, 10000
        writer = SimpleSignalWriter(filepath, chunk_size=chunk_size)
        start_time = time.time()
        for i in range(num_records):
            data = np.array([i, i*2, i*3], dtype=np.float32)
            timestamp = i * 10 ** 6  # Nanoseconds
            writer.append(data, timestamp)

        writer.finish()
        write_time = time.time() - start_time
        print(f"Writing completed in {write_time:.2f} seconds")

        assert filepath.exists()

        start_time = time.time()
        reader = SimpleSignal(filepath)

        result = reader.at(5000000)
        assert result is not None
        value, ts = result
        np.testing.assert_array_equal(value, np.array([5, 10, 15], dtype=np.float32))

        # Test window query (window appears to be inclusive on both ends)
        values, timestamps = reader.window(0, 9000000)  # First 10 records (0-9)
        assert len(values) == 10, f"Expected 10 values, got {len(values)}"
        assert len(timestamps) == 10, f"Expected 10 timestamps, got {len(timestamps)}"

        # Verify the window content
        for i in range(10):
            np.testing.assert_array_equal(values[i], np.array([i, i*2, i*3], dtype=np.float32))
            assert timestamps[i] == i * 1000000

        read_time = time.time() - start_time
        print(f"Reading completed in {read_time:.2f} seconds")

        # Check file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Records per chunk: {chunk_size}")
        print(f"Number of chunks written: {(num_records + chunk_size - 1) // chunk_size}")


if __name__ == "__main__":
    test_large_dataset_chunked_writing()