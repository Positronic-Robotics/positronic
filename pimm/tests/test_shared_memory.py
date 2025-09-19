from typing import Iterator
import pytest
import numpy as np

from pimm.core import Clock, Message, NoOpEmitter, SignalEmitter, SignalReceiver, Sleep
from pimm.shared_memory import NumpySMAdapter
from pimm.utils import DefaultReceiver, ValueUpdated
from pimm.world import TransportMode, World


class TestNumpySMAdapter:
    """Test the NumpySMAdapter implementation."""

    def test_set_to_buffer_and_read_from_buffer(self):
        original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        adapter = NumpySMAdapter(original_array.shape, original_array.dtype)
        adapter.array = original_array

        buffer = bytearray(adapter.buf_size())
        adapter.set_to_buffer(buffer)

        new_adapter = NumpySMAdapter(original_array.shape, original_array.dtype)
        new_adapter.read_from_buffer(buffer)

        # Verify arrays are equal
        assert new_adapter.array.shape == original_array.shape
        assert new_adapter.array.dtype == original_array.dtype
        assert np.array_equal(new_adapter.array, original_array)


class TestSharedMemoryAPI:
    """Test the public API for shared memory communication."""

    def test_world_creates_shared_memory_pair(self):
        """Test that forcing shared memory yields a working emitter/reader pair."""
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            assert isinstance(emitter, SignalEmitter)
            assert isinstance(reader, SignalReceiver)
            assert emitter.uses_shared_memory
            assert reader.uses_shared_memory

    def test_emitter_reader_basic_communication(self):
        """Test basic communication between emitter and reader."""
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            # Initially reader should return None (no data)
            assert reader.read() is None

            # Emit data
            array = np.array([3.14, 2.71], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

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
            emitter, _ = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            # First emit defines type
            emitter.emit(NumpySMAdapter(shape=(2,), dtype=np.float32))

            with pytest.raises(TypeError,
                               match="Shared memory transport selected; data must implement SMCompliant"):
                emitter.emit("wrong_type")

    def test_buffer_size_validation(self):
        """Test that emitter validates buffer size consistency."""
        with World() as world:
            emitter, _ = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            # First data with small array
            array1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
            data1 = NumpySMAdapter(array1.shape, array1.dtype)
            data1.array = array1
            result1 = emitter.emit(data1)
            assert result1 is True

            # Try to emit data with different buffer size (should fail)
            array2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)  # Different size
            data2 = NumpySMAdapter(array2.shape, array2.dtype)
            data2.array = array2
            with pytest.raises(AssertionError, match="Buffer size mismatch"):
                emitter.emit(data2)

    def test_data_updates_reflected_in_shared_memory_with_emit(self):
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

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

    def test_data_updates_not_reflected_in_shared_memory_without_emit(self):
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            # Initial emit
            emitter.emit(data, ts=100)
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message.ts == 100

            # Modify data but don't emit again
            data.array[0] = 3.0
            data.array[1] = 4.0

            # Reader should see updated values
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message.ts == 100

    def test_reader_returns_none_when_no_data_written(self):
        """Test that reader returns None when no data has been written."""
        with World() as world:
            _, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            # Reader should return None when no data has been emitted
            result = reader.read()
            assert result is None

    def test_multiple_readers_see_same_data(self):
        """Test that multiple readers can access the same shared memory."""
        with World() as world:
            emitter, reader1 = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)
            _, reader2 = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            # Note: This creates separate shared memory instances, so let's test
            # that each pair works independently
            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

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
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            array = np.array([5.5, 6.6], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, ts=333)

            # Multiple reads should return the same data
            message1 = reader.read()
            message2 = reader.read()
            message3 = reader.read()

            assert message1.ts == message2.ts == message3.ts == 333
            assert np.allclose(message1.data.array, [5.5, 6.6])
            assert np.allclose(message2.data.array, [5.5, 6.6])
            assert np.allclose(message3.data.array, [5.5, 6.6])

    def test_negative_timestamp_means_no_data(self):
        """Test that zero timestamp is treated as no data available."""
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 1.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            # Emit with timestamp 0 (should be auto-generated to non-zero)
            result = emitter.emit(data, ts=-1)
            assert result is True

            message = reader.read()
            assert message is not None
            assert message.ts >= 0  # Should be auto-generated

    def test_shared_memory_survives_data_modifications(self):
        """Test that shared memory correctly reflects live data modifications."""
        with World() as world:
            emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            array = np.array([10.0, 20.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
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

    def test_different_array_shapes_and_dtypes(self):
        """Test various array shapes and dtypes work correctly."""
        test_cases = [
            # Unsigned integers
            (np.array([1, 2, 3], dtype=np.uint8), "1D uint8"),
            (np.array([[1, 2], [3, 4]], dtype=np.uint16), "2D uint16"),
            (np.array([42], dtype=np.uint32), "scalar-like uint32"),
            (np.array([100], dtype=np.uint64), "scalar-like uint64"),
            # Signed integers
            (np.array([42], dtype=np.int8), "scalar-like int8"),
            (np.array([1, 2, 3], dtype=np.int16), "1D int16"),
            (np.array([[1, 2], [3, 4]], dtype=np.int32), "2D int32"),
            (np.array([100], dtype=np.int64), "scalar-like int64"),
            # Floating point
            (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16), "3D float16"),
            (np.array([3.14, 2.71], dtype=np.float32), "1D float32"),
            (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), "2D float64"),
            # Complex floating point
            (np.array([1 + 2j, 3 + 4j], dtype=np.complex64), "1D complex64"),
            (np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128), "2D complex128"),
        ]

        for test_array, description in test_cases:
            with World() as world:
                emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

                data = NumpySMAdapter(test_array.shape, test_array.dtype)
                data.array = test_array
                emitter.emit(data, ts=1000)

                message = reader.read()
                assert message is not None, f"Failed for {description}"
                assert message.data.array.shape == test_array.shape, f"Shape mismatch for {description}"
                assert message.data.array.dtype == test_array.dtype, f"Dtype mismatch for {description}"
                assert np.array_equal(message.data.array, test_array), f"Data mismatch for {description}"

    def test_world_context_required(self):
        w = World()
        with pytest.raises(AssertionError,
                           match=r"Shared memory transport is only available after entering the world context\."):
            w.mp_pipe(transport=TransportMode.SHARED_MEMORY)


class TestEmitterControlLoop:
    emitter: SignalEmitter = NoOpEmitter()

    def run(self, _should_stop: SignalReceiver, _clock: Clock) -> Iterator[Sleep]:
        yield Sleep(0.2)

        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = NumpySMAdapter(test_array.shape, test_array.dtype)
        data.array = test_array

        self.emitter.emit(data, ts=12345)

        yield Sleep(0.2)

        data.array[0] = 10.0
        self.emitter.emit(data, ts=67890)

        yield Sleep(0.2)


class TestSharedMemoryMultiprocessing:
    """Test shared memory in multiprocessing environment."""

    def test_simple_multiprocessing_communication(self):
        """Test that shared memory works across processes."""

        emitter_control_loop = TestEmitterControlLoop()

        with World() as world:
            emitter_control_loop.emitter, reader = world.mp_pipe(transport=TransportMode.SHARED_MEMORY)

            reader = DefaultReceiver(ValueUpdated(reader), (None, False))
            world.start_in_subprocess(emitter_control_loop.run)

            data = []
            while not world.should_stop:
                value, updated = reader.value
                if updated:
                    data.append(value.array.copy())

            assert len(data) == 2
            assert np.allclose(data[0], [1.0, 2.0, 3.0])
            assert np.allclose(data[1], [10.0, 2.0, 3.0])
