#!/usr/bin/env python3
"""Test script to verify chunked writing works with large datasets."""

import tempfile
import time
from pathlib import Path

import numpy as np

from positronic.dataset.vector import SimpleSignal, SimpleSignalWriter


def test_large_dataset_chunked_writing():
    """Test that chunked writing handles large datasets efficiently."""

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "large_test.parquet"

        chunk_size, num_records = 1000, 10000
        start_time = time.time()
        with SimpleSignalWriter(filepath, chunk_size=chunk_size) as writer:
            for i in range(num_records):
                data = np.array([i, i*2, i*3], dtype=np.float32)
                timestamp = i * 10 ** 6  # Nanoseconds
                writer.append(data, timestamp)
        write_time = time.time() - start_time
        print(f"Writing completed in {write_time:.2f} seconds")

        assert filepath.exists()

        start_time = time.time()
        reader = SimpleSignal(filepath)

        # Test time access
        value, ts = reader.time[5000000]
        np.testing.assert_array_equal(value, np.array([5, 10, 15], dtype=np.float32))
        assert ts == 5000000

        view = reader.time[0:9000001:1000000]
        assert len(view) == 10, f"Expected 10 values, got {len(view)}"

        # Verify the window content using index access
        for i in range(10):
            value, ts = view[i]
            np.testing.assert_array_equal(value, np.array([i, i*2, i*3], dtype=np.float32))
            assert ts == i * 1000000

        read_time = time.time() - start_time
        print(f"Reading completed in {read_time:.2f} seconds")

        # Check file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Records per chunk: {chunk_size}")
        print(f"Number of chunks written: {(num_records + chunk_size - 1) // chunk_size}")


if __name__ == "__main__":
    test_large_dataset_chunked_writing()
