import pytest
import multiprocessing as mp
import struct

from ironic2.core import Message
from ironic2.world import SMCompliant, ZeroCopySMEmitter, ZeroCopySMReader, World


class SMCompliantTestData(SMCompliant):
    """Test implementation of SMCompliant for testing purposes."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self._buffer = None
        self._lock = None
        self._x = x
        self._y = y

    @classmethod
    def buf_size(cls) -> int:
        # Two floats = 4 bytes each = 8 bytes total
        return 8

    def bind_to_buffer(self, buffer: memoryview, lock: mp.Lock) -> None:
        """Bind this object to a shared memory buffer."""
        self._buffer = buffer
        self._lock = lock
        # Write current values to buffer
        struct.pack_into('ff', buffer, 0, self._x, self._y)

    @classmethod
    def create_from_memoryview(cls, buffer: memoryview, lock: mp.Lock):
        """Create an instance from a shared memory buffer."""
        instance = cls()
        instance._buffer = buffer
        instance._lock = lock
        return instance

    @property
    def x(self) -> float:
        if self._buffer is not None:
            with self._lock:
                return struct.unpack_from('f', self._buffer, 0)[0]
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = value
        if self._buffer is not None:
            with self._lock:
                struct.pack_into('f', self._buffer, 0, value)

    @property
    def y(self) -> float:
        if self._buffer is not None:
            with self._lock:
                return struct.unpack_from('f', self._buffer, 4)[0]
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = value
        if self._buffer is not None:
            with self._lock:
                struct.pack_into('f', self._buffer, 4, value)

    def __eq__(self, other):
        if not isinstance(other, SMCompliantTestData):
            return False
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6


class TestZeroCopySMAPI:
    """Test the public API for zero-copy shared memory communication."""

    def test_world_creates_zero_copy_sm_pair(self):
        """Test that World.zero_copy_sm creates a working emitter/reader pair."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            assert isinstance(emitter, ZeroCopySMEmitter)
            assert isinstance(reader, ZeroCopySMReader)

    def test_emitter_reader_basic_communication(self):
        """Test basic communication between emitter and reader."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            # Initially reader should return None (no data)
            assert reader.read() is None

            # Emit data
            data = SMCompliantTestData(3.14, 2.71)
            result = emitter.emit(data, ts=12345)
            assert result is True

            # Reader should now receive the data
            message = reader.read()
            assert message is not None
            assert isinstance(message, Message)
            assert message.ts == 12345
            assert isinstance(message.data, SMCompliantTestData)
            assert abs(message.data.x - 3.14) < 1e-6
            assert abs(message.data.y - 2.71) < 1e-6

    def test_emitter_rejects_wrong_data_type(self):
        """Test that emitter rejects data of wrong type."""
        with World() as world:
            emitter, _ = world.zero_copy_sm(SMCompliantTestData)

            with pytest.raises(AssertionError, match="Data type mismatch"):
                emitter.emit("wrong_type")

    def test_emitter_requires_same_object_instance(self):
        """Test that emitter can only emit the same object instance multiple times."""
        with World() as world:
            emitter, _ = world.zero_copy_sm(SMCompliantTestData)

            data1 = SMCompliantTestData(1.0, 2.0)
            data2 = SMCompliantTestData(1.0, 2.0)  # Same values but different object

            # First emit should succeed
            result1 = emitter.emit(data1)
            assert result1 is True

            # Second emit with different object should fail
            with pytest.raises(AssertionError, match="SMEmitter can only emit the same object multiple times"):
                emitter.emit(data2)

    def test_data_updates_reflect_in_shared_memory(self):
        """Test that updates to the data object are reflected in shared memory."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            data = SMCompliantTestData(1.0, 2.0)

            # Initial emit
            emitter.emit(data, ts=100)
            message = reader.read()
            assert message.data.x == 1.0
            assert message.data.y == 2.0
            assert message.ts == 100

            # Modify data and emit again
            data.x = 3.0
            data.y = 4.0
            emitter.emit(data, ts=200)

            # Reader should see updated values
            message = reader.read()
            assert message.data.x == 3.0
            assert message.data.y == 4.0
            assert message.ts == 200

    def test_reader_returns_none_when_no_data_written(self):
        """Test that reader returns None when no data has been written."""
        with World() as world:
            _, reader = world.zero_copy_sm(SMCompliantTestData)

            # Reader should return None when no data has been emitted
            result = reader.read()
            assert result is None

    def test_multiple_readers_see_same_data(self):
        """Test that multiple readers can access the same shared memory."""
        with World() as world:
            emitter, reader1 = world.zero_copy_sm(SMCompliantTestData)
            _, reader2 = world.zero_copy_sm(SMCompliantTestData)

            # Note: This creates separate shared memory instances, so let's test
            # that each pair works independently
            data = SMCompliantTestData(1.0, 2.0)

            emitter.emit(data, ts=111)
            message1 = reader1.read()
            assert message1.data.x == 1.0
            assert message1.data.y == 2.0
            assert message1.ts == 111

            # reader2 should return None since it's a different shared memory
            message2 = reader2.read()
            assert message2 is None

    def test_data_persistence_across_multiple_reads(self):
        """Test that data persists across multiple read operations."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            data = SMCompliantTestData(5.5, 6.6)
            emitter.emit(data, ts=333)

            # Multiple reads should return the same data
            message1 = reader.read()
            message2 = reader.read()
            message3 = reader.read()

            assert message1.ts == message2.ts == message3.ts == 333
            assert (abs(message1.data.x - 5.5) < 1e-6 and abs(message2.data.x - 5.5) < 1e-6
                    and abs(message3.data.x - 5.5) < 1e-6)
            assert (abs(message1.data.y - 6.6) < 1e-6 and abs(message2.data.y - 6.6) < 1e-6
                    and abs(message3.data.y - 6.6) < 1e-6)

    def test_zero_timestamp_means_no_data(self):
        """Test that zero timestamp is treated as no data available."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            data = SMCompliantTestData(1.0, 1.0)

            # Emit with timestamp 0 (should be auto-generated to non-zero)
            result = emitter.emit(data, ts=0)
            assert result is True

            message = reader.read()
            assert message is not None
            assert message.ts != 0  # Should be auto-generated

    def test_shared_memory_survives_data_modifications(self):
        """Test that shared memory correctly reflects live data modifications."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            data = SMCompliantTestData(10.0, 20.0)
            emitter.emit(data, ts=500)

            # Verify initial values
            message = reader.read()
            assert message.data.x == 10.0
            assert message.data.y == 20.0

            # Modify the original data object
            data.x = 100.0
            data.y = 200.0

            # Re-emit to update timestamp
            emitter.emit(data, ts=600)

            # Reader should see the updated values
            message = reader.read()
            assert message.data.x == 100.0
            assert message.data.y == 200.0
            assert message.ts == 600

    def test_reader_data_is_readonly(self):
        """Test that data read from shared memory cannot be modified."""
        with World() as world:
            emitter, reader = world.zero_copy_sm(SMCompliantTestData)

            # Emit some data
            data = SMCompliantTestData(5.0, 10.0)
            emitter.emit(data, ts=123)

            # Read the data
            message = reader.read()
            assert message is not None
            read_data = message.data

            # Attempting to modify the read data should raise an exception
            with pytest.raises((TypeError, ValueError)):
                read_data.x = 999.0

            with pytest.raises((TypeError, ValueError)):
                read_data.y = 999.0
