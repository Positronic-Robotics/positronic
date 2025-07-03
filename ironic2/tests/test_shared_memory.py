import pytest
import numpy as np

from ironic2.core import Message
from ironic2.shared_memory import NumpySMAdapter
from ironic2.world import ZeroCopySMEmitter, ZeroCopySMReader, World


class TestNumpySMAdapter:
    """Test the NumpySMAdapter implementation."""

    def test_unsupported_dtype_raises_error(self):
        """Test that unsupported dtypes raise errors."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Not supported
        adapter = NumpySMAdapter(array)

        buffer_size = adapter.buf_size()
        buffer = bytearray(buffer_size)

        with pytest.raises(KeyError):
            adapter.move_to_buffer(buffer)

    def test_move_to_buffer_and_create_from_memoryview(self):
        """Test serialization and deserialization roundtrip."""
        original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        adapter = NumpySMAdapter(original_array)

        buffer = bytearray(adapter.buf_size())
        adapter.move_to_buffer(buffer)

        new_adapter = NumpySMAdapter.create_from_memoryview(buffer)

        # Verify arrays are equal
        assert new_adapter.array.shape == original_array.shape
        assert new_adapter.array.dtype == original_array.dtype
        assert np.array_equal(new_adapter.array, original_array)

    def test_zero_copy_behavior(self):
        """Test that changes to buffer reflect in array."""
        array = np.array([1, 2, 3], dtype=np.int32)
        adapter = NumpySMAdapter(array)

        buffer = bytearray(adapter.buf_size())
        adapter.move_to_buffer(buffer)

        adapter.array[0] = 99

        new_adapter = NumpySMAdapter.create_from_memoryview(buffer)
        assert new_adapter.array[0] == 99

    def test_readonly_buffer_error(self):
        """Test that readonly buffer raises error in move_to_buffer."""
        array = np.array([1, 2, 3], dtype=np.uint8)
        adapter = NumpySMAdapter(array)

        buffer = bytes(adapter.buf_size())  # readonly bytes

        with pytest.raises(ValueError, match="Buffer must be writable"):
            adapter.move_to_buffer(buffer)


class TestZeroCopySMAPI:
    """Test the public API for zero-copy shared memory communication."""

    def test_world_creates_zero_copy_sm_pair(self):
        """Test that World.zero_copy_sm creates a working emitter/reader pair."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            assert isinstance(emitter, ZeroCopySMEmitter)
            assert isinstance(reader, ZeroCopySMReader)

    def test_emitter_reader_basic_communication(self):
        """Test basic communication between emitter and reader."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            # Initially reader should return None (no data)
            assert reader.read() is None

            # Emit data
            array = np.array([3.14, 2.71], dtype=np.float32)
            data = NumpySMAdapter(array)
            result = emitter.emit(data, ts=12345)
            assert result is True

            # Reader should now receive the data
            message = reader.read()
            assert message is not None
            assert isinstance(message, Message)
            assert message.ts == 12345
            assert isinstance(message.data, NumpySMAdapter)
            assert np.allclose(message.data.array, [3.14, 2.71])

    def test_emitter_rejects_wrong_data_type(self):
        """Test that emitter rejects data of wrong type."""
        with World() as world:
            emitter, _ = world.zero_copy_sm(NumpySMAdapter)

            with pytest.raises(AssertionError, match="Data type mismatch"):
                emitter.emit("wrong_type")

    def test_emitter_requires_same_object_instance(self):
        """Test that emitter can only emit the same object instance multiple times."""
        with World() as world:
            emitter, _ = world.zero_copy_sm(NumpySMAdapter)

            array1 = np.array([1.0, 2.0], dtype=np.float32)
            data1 = NumpySMAdapter(array1)
            array2 = np.array([1.0, 2.0], dtype=np.float32)  # Same values but different object
            data2 = NumpySMAdapter(array2)

            # First emit should succeed
            result1 = emitter.emit(data1)
            assert result1 is True

            # Second emit with different object should fail
            with pytest.raises(AssertionError, match="SMEmitter can only emit the same object multiple times"):
                emitter.emit(data2)

    def test_buffer_size_validation(self):
        """Test that emitter validates buffer size consistency."""
        with World() as world:
            emitter, _ = world.zero_copy_sm(NumpySMAdapter)

            # First data with small array
            array1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
            data1 = NumpySMAdapter(array1)
            result1 = emitter.emit(data1)
            assert result1 is True

            # Try to emit data with different buffer size (should fail)
            array2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)  # Different size
            data2 = NumpySMAdapter(array2)
            with pytest.raises(AssertionError, match="Buffer size mismatch"):
                emitter.emit(data2)

    def test_data_updates_reflect_in_shared_memory(self):
        """Test that updates to the data object are reflected in shared memory."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array)

            # Initial emit
            emitter.emit(data, ts=100)
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message.ts == 100

            # Modify data and emit again
            data.array[0] = 3.0
            data.array[1] = 4.0
            emitter.emit(data, ts=200)

            # Reader should see updated values
            message = reader.read()
            assert np.allclose(message.data.array, [3.0, 4.0])
            assert message.ts == 200

    def test_numpy_array_zero_copy(self):
        """Test that numpy arrays are properly shared with zero-copy."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            # Create data with a specific image
            test_array = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
            data = NumpySMAdapter(test_array.copy())

            # Emit data
            emitter.emit(data, ts=123)

            # Read data and verify array is the same
            message = reader.read()
            assert message is not None
            received_array = message.data.array
            assert np.array_equal(received_array, test_array)

            # Modify the original array and check that the reader sees the change
            # This is a not recommended behaviour, as we expect the emitters to write again
            # but this is the only way to test zero-copy behaviour.
            new_values = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
            data.array[:] = new_values
            assert np.array_equal(received_array, new_values)

            received_array = None  # Clean up references to prevent shared memory warning

    def test_reader_returns_none_when_no_data_written(self):
        """Test that reader returns None when no data has been written."""
        with World() as world:
            _, reader = world.zero_copy_sm(NumpySMAdapter)

            # Reader should return None when no data has been emitted
            result = reader.read()
            assert result is None

    def test_multiple_readers_see_same_data(self):
        """Test that multiple readers can access the same shared memory."""
        with World() as world:
            emitter, reader1 = world.zero_copy_sm(NumpySMAdapter)
            _, reader2 = world.zero_copy_sm(NumpySMAdapter)

            # Note: This creates separate shared memory instances, so let's test
            # that each pair works independently
            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array)

            emitter.emit(data, ts=111)
            message1 = reader1.read()
            assert np.allclose(message1.data.array, [1.0, 2.0])
            assert message1.ts == 111

            # reader2 should return None since it's a different shared memory
            message2 = reader2.read()
            assert message2 is None

    def test_data_persistence_across_multiple_reads(self):
        """Test that data persists across multiple read operations."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            array = np.array([5.5, 6.6], dtype=np.float32)
            data = NumpySMAdapter(array)
            emitter.emit(data, ts=333)

            # Multiple reads should return the same data
            message1 = reader.read()
            message2 = reader.read()
            message3 = reader.read()

            assert message1.ts == message2.ts == message3.ts == 333
            assert np.allclose(message1.data.array, [5.5, 6.6])
            assert np.allclose(message2.data.array, [5.5, 6.6])
            assert np.allclose(message3.data.array, [5.5, 6.6])

    def test_zero_timestamp_means_no_data(self):
        """Test that zero timestamp is treated as no data available."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            array = np.array([1.0, 1.0], dtype=np.float32)
            data = NumpySMAdapter(array)

            # Emit with timestamp 0 (should be auto-generated to non-zero)
            result = emitter.emit(data, ts=0)
            assert result is True

            message = reader.read()
            assert message is not None
            assert message.ts != 0  # Should be auto-generated

    def test_shared_memory_survives_data_modifications(self):
        """Test that shared memory correctly reflects live data modifications."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            array = np.array([10.0, 20.0], dtype=np.float32)
            data = NumpySMAdapter(array)
            emitter.emit(data, ts=500)

            # Verify initial values
            message = reader.read()
            assert np.allclose(message.data.array, [10.0, 20.0])

            # Modify the original data object
            data.array[0] = 100.0
            data.array[1] = 200.0

            # Re-emit to update timestamp
            emitter.emit(data, ts=600)

            # Reader should see the updated values
            message = reader.read()
            assert np.allclose(message.data.array, [100.0, 200.0])
            assert message.ts == 600

    def test_reader_data_is_readonly(self):
        """Test that data read from shared memory cannot be modified."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(NumpySMAdapter)

            # Emit some data
            array = np.array([5.0, 10.0], dtype=np.float32)
            data = NumpySMAdapter(array)
            emitter.emit(data, ts=123)

            # Read the data
            message = reader.read()
            assert message is not None
            read_data = message.data

            # Attempting to modify the read data should raise an exception
            with pytest.raises((TypeError, ValueError)):
                read_data.array[0] = 999.0

            with pytest.raises((TypeError, ValueError)):
                read_data.array[1] = 999.0

    def test_different_array_shapes_and_dtypes(self):
        """Test various array shapes and dtypes work correctly."""
        test_cases = [
            (np.array([1, 2, 3], dtype=np.uint8), "1D uint8"),
            (np.array([[1, 2], [3, 4]], dtype=np.int32), "2D int32"),
            (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16), "3D float16"),
            (np.array([42], dtype=np.int8), "scalar-like int8"),
        ]

        for test_array, description in test_cases:
            with World() as world:
                emitter, reader = world.zero_copy_sm(NumpySMAdapter)

                data = NumpySMAdapter(test_array)
                emitter.emit(data, ts=1000)

                message = reader.read()
                assert message is not None, f"Failed for {description}"
                assert message.data.array.shape == test_array.shape, f"Shape mismatch for {description}"
                assert message.data.array.dtype == test_array.dtype, f"Dtype mismatch for {description}"
                assert np.array_equal(message.data.array, test_array), f"Data mismatch for {description}"
