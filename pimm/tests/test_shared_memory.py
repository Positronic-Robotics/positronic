import time
from collections.abc import Iterator
from multiprocessing.managers import BaseProxy

import numpy as np
import pytest

from pimm.core import Clock, Message, NoOpEmitter, NoOpReceiver, SignalEmitter, SignalReceiver, Sleep, Yield
from pimm.shared_memory import NumpySMAdapter
from pimm.time import EMITTED_WALL, Time
from pimm.world import MAX_PICKLED_TIME_BYTES, MultiprocessReceiver, TransportMode, World


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
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            assert isinstance(emitter, SignalEmitter)
            assert isinstance(reader, SignalReceiver)
            assert emitter.uses_shared_memory
            assert reader.uses_shared_memory

    def test_emitter_reader_basic_communication(self):
        """Test basic communication between emitter and reader."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # Initially reader should return None (no data)
            assert reader.read() is None

            # Emit data
            array = np.array([3.14, 2.71], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            emitter.emit(data, time=Time(source=12345))

            # Reader should now receive the data
            message = reader.read()
            assert message is not None
            assert isinstance(message, Message)
            assert message is not None
            assert message.time['source'] == 12345
            assert isinstance(message.data, NumpySMAdapter)
            assert np.allclose(message.data.array, [3.14, 2.71])
            assert message.updated is True

            stale = reader.read()
            assert stale is not None
            assert stale.updated is False

    def test_emitter_rejects_wrong_data_type(self):
        """Test that emitter rejects data of wrong type."""
        with World() as world:
            emitter, _ = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # First emit defines type
            emitter.emit(NumpySMAdapter(shape=(2,), dtype=np.float32))

            with pytest.raises(TypeError, match='Shared memory transport selected; data must implement SMCompliant'):
                emitter.emit('wrong_type')

    def test_buffer_size_validation(self):
        """Test that emitter validates buffer size consistency."""
        with World() as world:
            emitter, _ = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # First data with small array
            array1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
            data1 = NumpySMAdapter(array1.shape, array1.dtype)
            data1.array = array1
            emitter.emit(data1)

            # Try to emit data with different buffer size (should fail)
            array2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)  # Different size
            data2 = NumpySMAdapter(array2.shape, array2.dtype)
            data2.array = array2
            with pytest.raises(AssertionError, match='Buffer size mismatch'):
                emitter.emit(data2)

    def test_data_updates_reflected_in_shared_memory_with_emit(self):
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            # Initial emit
            emitter.emit(data, time=Time(source=100))
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message is not None
            assert message.time['source'] == 100

            # Modify data and emit again
            data.array[0] = 3.0
            data.array[1] = 4.0
            emitter.emit(data, time=Time(source=200))

            # Reader should see updated values
            message = reader.read()
            assert np.allclose(message.data.array, [3.0, 4.0])
            assert message is not None
            assert message.time['source'] == 200

    def test_data_updates_not_reflected_in_shared_memory_without_emit(self):
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            # Initial emit
            emitter.emit(data, time=Time(source=100))
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message is not None
            assert message.time['source'] == 100

            # Modify data but don't emit again
            data.array[0] = 3.0
            data.array[1] = 4.0

            # Reader should see updated values
            message = reader.read()
            assert np.allclose(message.data.array, [1.0, 2.0])
            assert message is not None
            assert message.time['source'] == 100

    def test_reader_returns_none_when_no_data_written(self):
        """Test that reader returns None when no data has been written."""
        with World() as world:
            _, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # Reader should return None when no data has been emitted
            result = reader.read()
            assert result is None

    def test_multiple_readers_see_same_data(self):
        """Test that multiple readers can access the same shared memory."""
        with World() as world:
            emitter, reader1 = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)
            _, reader2 = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # Note: This creates separate shared memory instances, so let's test
            # that each pair works independently
            array = np.array([1.0, 2.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            emitter.emit(data, time=Time(source=111))
            message1 = reader1.read()
            assert np.allclose(message1.data.array, [1.0, 2.0])
            assert message1 is not None
            assert message1.time['source'] == 111

            # reader2 should return None since it's a different shared memory
            message2 = reader2.read()
            assert message2 is None

    def test_data_persistence_across_multiple_reads(self):
        """Test that data persists across multiple read operations."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            array = np.array([5.5, 6.6], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=333))

            # Multiple reads should return the same data
            message1 = reader.read()
            message2 = reader.read()
            message3 = reader.read()

            assert message1 is not None
            assert message2 is not None and message3 is not None
            assert message1.time['source'] == message2.time['source'] == message3.time['source'] == 333
            assert np.allclose(message1.data.array, [5.5, 6.6])
            assert np.allclose(message2.data.array, [5.5, 6.6])
            assert np.allclose(message3.data.array, [5.5, 6.6])

    def test_negative_timestamp_means_no_data(self):
        """Test that zero timestamp is treated as no data available."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            array = np.array([1.0, 1.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array

            # Emit with timestamp 0 (should be auto-generated to non-zero)
            emitter.emit(data, time=None)
            message = reader.read()
            assert message is not None
            assert message is not None
            assert message.time[EMITTED_WALL] >= 0  # Should be auto-generated

    def test_shared_memory_survives_data_modifications(self):
        """Test that shared memory correctly reflects live data modifications."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            array = np.array([10.0, 20.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=500))

            # Verify initial values
            message = reader.read()
            assert np.allclose(message.data.array, [10.0, 20.0])

            # Modify the original data object
            data.array[0] = 100.0
            data.array[1] = 200.0

            # Re-emit to update timestamp
            emitter.emit(data, time=Time(source=600))

            # Reader should see the updated values
            message = reader.read()
            assert np.allclose(message.data.array, [100.0, 200.0])
            assert message is not None
            assert message.time['source'] == 600

    def test_different_array_shapes_and_dtypes(self):
        """Test various array shapes and dtypes work correctly."""
        test_cases = [
            # Unsigned integers
            (np.array([1, 2, 3], dtype=np.uint8), '1D uint8'),
            (np.array([[1, 2], [3, 4]], dtype=np.uint16), '2D uint16'),
            (np.array([42], dtype=np.uint32), 'scalar-like uint32'),
            (np.array([100], dtype=np.uint64), 'scalar-like uint64'),
            # Signed integers
            (np.array([42], dtype=np.int8), 'scalar-like int8'),
            (np.array([1, 2, 3], dtype=np.int16), '1D int16'),
            (np.array([[1, 2], [3, 4]], dtype=np.int32), '2D int32'),
            (np.array([100], dtype=np.int64), 'scalar-like int64'),
            # Floating point
            (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16), '3D float16'),
            (np.array([3.14, 2.71], dtype=np.float32), '1D float32'),
            (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), '2D float64'),
            # Complex floating point
            (np.array([1 + 2j, 3 + 4j], dtype=np.complex64), '1D complex64'),
            (np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128), '2D complex128'),
        ]

        for test_array, description in test_cases:
            with World() as world:
                emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

                data = NumpySMAdapter(test_array.shape, test_array.dtype)
                data.array = test_array
                emitter.emit(data, time=Time(source=1000))

                message = reader.read()
                assert message is not None, f'Failed for {description}'
                assert message.data.array.shape == test_array.shape, f'Shape mismatch for {description}'
                assert message.data.array.dtype == test_array.dtype, f'Dtype mismatch for {description}'
                assert np.array_equal(message.data.array, test_array), f'Data mismatch for {description}'

    def test_world_context_required(self):
        w = World()
        with pytest.raises(
            AssertionError, match=r'Shared memory transport is only available after entering the world context\.'
        ):
            w.mp_pipes(transport=TransportMode.SHARED_MEMORY)


class TestEmitterControlLoop:
    emitter: SignalEmitter = NoOpEmitter()

    def run(self, _should_stop: SignalReceiver, _clock: Clock) -> Iterator[Sleep]:
        yield Sleep(0.2)

        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data = NumpySMAdapter(test_array.shape, test_array.dtype)
        data.array = test_array

        self.emitter.emit(data, time=Time(source=12345))

        yield Sleep(0.2)

        data.array[0] = 10.0
        self.emitter.emit(data, time=Time(source=67890))

        yield Sleep(0.2)


class TestSharedMemoryMultiprocessing:
    """Test shared memory in multiprocessing environment."""

    def test_simple_multiprocessing_communication(self):
        """Test that shared memory works across processes."""

        emitter_control_loop = TestEmitterControlLoop()

        with World() as world:
            emitter_control_loop.emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            world.start_in_subprocess(emitter_control_loop.run)

            data = []
            while not world.should_stop:
                msg = reader.read()
                if msg is not None and msg.updated:
                    data.append(msg.data.array.copy())

            assert len(data) == 2
            assert np.allclose(data[0], [1.0, 2.0, 3.0])
            assert np.allclose(data[1], [10.0, 2.0, 3.0])


# Nanoseconds since the epoch. A double holds a value of this size only to 256 ns, so a float field loses digits.
NS_TIMESTAMP = 1_758_650_000_123_456_789


class FrameStampEmitter:
    """Emits frames whose every element equals the frame's timestamp, as fast as the pipe takes them."""

    emitter: SignalEmitter = NoOpEmitter()

    def run(self, should_stop: SignalReceiver, _clock: Clock) -> Iterator[Yield]:
        data = NumpySMAdapter((256, 1024), np.dtype(np.int64))
        ts = 0
        while not should_stop.value:
            data.array[:] = ts
            self.emitter.emit(data, time=Time(capture=ts))
            ts += 1
            yield Yield()


class SingleFrameEmitter:
    emitter: SignalEmitter = NoOpEmitter()

    def run(self, should_stop: SignalReceiver, _clock: Clock) -> Iterator[Sleep]:
        data = NumpySMAdapter((3,), np.dtype(np.float32))
        data.array[:] = [1.0, 2.0, 3.0]
        self.emitter.emit(data, time=Time(capture=NS_TIMESTAMP))
        while not should_stop.value:
            yield Sleep(0.01)


class TimestampEcho:
    """Reads frames in its own process and sends the timestamp and data of each new frame back over a second pipe."""

    frames: SignalReceiver = NoOpReceiver()
    echo: SignalEmitter = NoOpEmitter()

    def run(self, should_stop: SignalReceiver, _clock: Clock) -> Iterator[Sleep]:
        while not should_stop.value:
            msg = self.frames.read()
            if msg is not None and msg.updated:
                self.echo.emit((msg.time['capture'], msg.data.array.tolist()))
            yield Sleep(0.001)


def _single_pipe(world: World, transport: TransportMode = TransportMode.UNDECIDED):
    emitter, receiver = world.mp_pipes(transport=transport)
    assert isinstance(receiver, MultiprocessReceiver)
    return emitter, receiver


def _wait_for(receiver: SignalReceiver, predicate, timeout: float = 20.0) -> Message:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = receiver.read()
        if msg is not None and predicate(msg):
            return msg
        time.sleep(0.001)
    raise AssertionError('no matching message before the timeout')


class TestSharedMemoryAcrossProcesses:
    def test_a_read_never_returns_a_frame_torn_by_a_concurrent_emit(self):
        frames_to_check = 50
        emitter_loop = FrameStampEmitter()

        with World() as world:
            emitter_loop.emitter, reader = _single_pipe(world, TransportMode.SHARED_MEMORY)
            world.start_in_subprocess(emitter_loop.run)

            seen = set()
            deadline = time.monotonic() + 20.0
            while len(seen) < frames_to_check and time.monotonic() < deadline:
                msg = reader.read()
                if msg is not None:
                    ts = msg.time['capture']
                    assert np.all(msg.data.array == ts), f"frame {ts} holds another frame's data"
                    seen.add(ts)

        assert len(seen) >= frames_to_check

    def test_an_undecided_pipe_takes_the_transport_its_subprocess_emitter_picks(self):
        emitter_loop = SingleFrameEmitter()

        with World() as world:
            emitter_loop.emitter, reader = _single_pipe(world)
            world.start_in_subprocess(emitter_loop.run)

            msg = _wait_for(reader, lambda m: m.updated)

            assert reader.uses_shared_memory
            assert msg.time['capture'] == NS_TIMESTAMP
            assert np.allclose(msg.data.array, [1.0, 2.0, 3.0])

    def test_a_reader_in_a_subprocess_gets_each_frame_and_its_nanosecond_timestamp(self):
        echo_loop = TimestampEcho()

        with World() as world:
            emitter, echo_loop.frames = _single_pipe(world, TransportMode.SHARED_MEMORY)
            echo_loop.echo, echoes = _single_pipe(world, TransportMode.QUEUE)
            world.start_in_subprocess(echo_loop.run)

            data = NumpySMAdapter((2,), np.dtype(np.float32))
            for step in range(3):
                ts = NS_TIMESTAMP + step
                data.array[:] = [step, -step]
                emitter.emit(data, time=Time(capture=ts))
                msg = _wait_for(echoes, lambda m, ts=ts: m.data[0] == ts)
                assert msg.data[1] == [step, -step]

    def test_emit_and_read_make_no_call_to_the_manager_process(self, monkeypatch):
        with World() as world:
            emitter, reader = _single_pipe(world, TransportMode.SHARED_MEMORY)
            data = NumpySMAdapter((2,), np.dtype(np.float32))
            emitter.emit(data, time=Time(capture=1))
            assert reader.read() is not None  # The first read takes the buffer's name from a manager queue.

            calls = []
            original = BaseProxy._callmethod

            def counting_callmethod(self, *args, **kwargs):
                calls.append(args)
                return original(self, *args, **kwargs)

            monkeypatch.setattr(BaseProxy, '_callmethod', counting_callmethod)
            for ts in range(2, 12):
                emitter.emit(data, time=Time(capture=ts))
                msg = reader.read()
                assert msg is not None and msg.time['capture'] == ts

            assert calls == []

    def test_a_time_too_large_for_the_pipe_is_refused_and_leaves_the_last_frame(self):
        with World() as world:
            emitter, reader = _single_pipe(world, TransportMode.SHARED_MEMORY)
            data = NumpySMAdapter((2,), np.dtype(np.float32))
            data.array[:] = 1.0
            emitter.emit(data, time=Time(capture=1))

            data.array[:] = 2.0
            too_large = Time(**{f'timeline_{i}': i for i in range(MAX_PICKLED_TIME_BYTES)})
            with pytest.raises(ValueError, match=str(MAX_PICKLED_TIME_BYTES)):
                emitter.emit(data, time=too_large)

            msg = reader.read()
            assert msg is not None
            assert msg.time['capture'] == 1 and msg.data.array.tolist() == [1.0, 1.0]


class TestBroadcastCommunication:
    """Test broadcast communication (one emitter, multiple receivers)."""

    def test_broadcast_with_shared_memory(self):
        """Test that one emitter can broadcast to multiple receivers via shared memory."""
        with World() as world:
            emitter, readers = world.mp_pipes(transport=TransportMode.SHARED_MEMORY, num_receivers=3)

            assert isinstance(readers, list)
            assert len(readers) == 3
            assert all(isinstance(r, SignalReceiver) for r in readers)
            assert all(r.uses_shared_memory for r in readers)

            # Initially all readers should return None
            assert all(r.read() is None for r in readers)

            # Emit data
            array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=100))

            # All readers should receive the same data
            messages = [r.read() for r in readers]
            assert all(msg is not None for msg in messages)
            assert all(msg is not None and msg.time['source'] == 100 for msg in messages)
            assert all(msg.updated is True for msg in messages)
            assert all(np.allclose(msg.data.array, [1.0, 2.0, 3.0]) for msg in messages)

            # Update and emit again
            data.array[0] = 10.0
            emitter.emit(data, time=Time(source=200))

            # All readers should see the updated values
            messages = [r.read() for r in readers]
            assert all(msg is not None for msg in messages)
            assert all(msg is not None and msg.time['source'] == 200 for msg in messages)
            assert all(msg.updated is True for msg in messages)
            assert all(np.allclose(msg.data.array, [10.0, 2.0, 3.0]) for msg in messages)

    def test_broadcast_with_queue(self):
        """Test that one emitter can broadcast to multiple receivers via queue."""
        with World() as world:
            emitter, readers = world.mp_pipes(transport=TransportMode.QUEUE, num_receivers=3)

            assert isinstance(readers, list)
            assert len(readers) == 3
            assert all(isinstance(r, SignalReceiver) for r in readers)
            assert all(not r.uses_shared_memory for r in readers)

            # Initially all readers should return None
            assert all(r.read() is None for r in readers)

            # Emit data (regular Python object, not SMCompliant)
            data = {'value': 42, 'name': 'test'}
            emitter.emit(data, time=Time(source=100))

            # All readers should receive the same data
            messages = [r.read() for r in readers]
            assert all(msg is not None for msg in messages)
            assert all(msg is not None and msg.time['source'] == 100 for msg in messages)
            assert all(msg.updated is True for msg in messages)
            assert all(msg.data == {'value': 42, 'name': 'test'} for msg in messages)

            # Emit again with different data
            data2 = {'value': 99, 'name': 'updated'}
            emitter.emit(data2, time=Time(source=200))

            # All readers should see the new data
            messages = [r.read() for r in readers]
            assert all(msg is not None for msg in messages)
            assert all(msg is not None and msg.time['source'] == 200 for msg in messages)
            assert all(msg.updated is True for msg in messages)
            assert all(msg.data == {'value': 99, 'name': 'updated'} for msg in messages)

    def test_broadcast_independent_reader_states(self):
        """Test that each receiver maintains independent read state."""
        with World() as world:
            emitter, readers = world.mp_pipes(transport=TransportMode.SHARED_MEMORY, num_receivers=2)

            array = np.array([5.0, 6.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=100))

            # First reader reads the message (updated=True)
            msg1 = readers[0].read()
            assert msg1.updated is True
            assert msg1 is not None
            assert msg1.time['source'] == 100

            # Second reader hasn't read yet, should still see updated=True
            msg2 = readers[1].read()
            assert msg2.updated is True
            assert msg2 is not None
            assert msg2.time['source'] == 100

            # Both readers read again without new emit (updated=False for both)
            msg1_stale = readers[0].read()
            msg2_stale = readers[1].read()
            assert msg1_stale.updated is False
            assert msg2_stale.updated is False


class TestConnectOneToOne:
    """Test one-to-one connections between emitter and receiver."""

    def test_single_receiver_with_shared_memory(self):
        """Test that mp_pipes returns a single receiver (not a list) when num_receivers=1."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY, num_receivers=1)

            # Should return a single receiver, not a list
            assert isinstance(reader, SignalReceiver)
            assert not isinstance(reader, list)
            assert reader.uses_shared_memory

            # Test basic communication
            array = np.array([7.0, 8.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=300))

            message = reader.read()
            assert message is not None
            assert message is not None
            assert message.time['source'] == 300
            assert np.allclose(message.data.array, [7.0, 8.0])

    def test_single_receiver_with_queue(self):
        """Test one-to-one communication with queue transport."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.QUEUE, num_receivers=1)

            # Should return a single receiver, not a list
            assert isinstance(reader, SignalReceiver)
            assert not isinstance(reader, list)
            assert not reader.uses_shared_memory

            # Test basic communication
            data = {'status': 'active', 'count': 123}
            emitter.emit(data, time=Time(source=400))

            message = reader.read()
            assert message is not None
            assert message is not None
            assert message.time['source'] == 400
            assert message.data == {'status': 'active', 'count': 123}

    def test_default_num_receivers_is_one(self):
        """Test that num_receivers defaults to 1."""
        with World() as world:
            emitter, reader = world.mp_pipes(transport=TransportMode.SHARED_MEMORY)

            # Default should be single receiver
            assert isinstance(reader, SignalReceiver)
            assert not isinstance(reader, list)

            array = np.array([99.0], dtype=np.float32)
            data = NumpySMAdapter(array.shape, array.dtype)
            data.array = array
            emitter.emit(data, time=Time(source=500))

            message = reader.read()
            assert message is not None
            assert message is not None
            assert message.time['source'] == 500
            assert np.allclose(message.data.array, [99.0])
