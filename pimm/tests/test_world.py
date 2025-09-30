import multiprocessing as mp
import struct
import time
from queue import Empty, Full
from unittest.mock import Mock, patch

import pytest

from pimm.core import (
    ControlSystem,
    ControlSystemEmitter,
    ControlSystemReceiver,
    Message,
    SignalEmitter,
    SignalReceiver,
    Sleep,
)
from pimm.shared_memory import SMCompliant
from pimm.tests.testing import MockClock
from pimm.world import EventReceiver, QueueEmitter, QueueReceiver, SystemClock, World


def dummy_process(stop_reader, clock):
    """A simple background process that runs until stopped."""
    while not stop_reader.read().data:
        yield Sleep(0.01)


class DummyControlSystem(ControlSystem):
    """Minimal control system used for integration-style tests."""

    def __init__(self, name: str, steps: int = 1):
        self.name = name
        self.steps = steps
        self.emitter = ControlSystemEmitter(self)
        self.receiver = ControlSystemReceiver(self)
        self.invocations = []

    def run(self, should_stop, clock):  # pragma: no cover - exercised via tests
        self.invocations.append((should_stop, clock))
        for _ in range(self.steps):
            yield Sleep(0.0)

    def __repr__(self):
        return f'DummyControlSystem(name={self.name!r})'


class DummySMValue(SMCompliant):
    """Simple SMCompliant payload used to test adaptive transports."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def buf_size(self) -> int:
        return 8

    def instantiation_params(self) -> tuple[float]:
        return (0.0,)

    def set_to_buffer(self, buffer: memoryview | bytes | bytearray) -> None:
        buffer[:8] = struct.pack('d', self.value)

    def read_from_buffer(self, buffer: memoryview | bytes) -> None:
        self.value = struct.unpack('d', buffer[:8])[0]


class TestQueueEmitter:
    """Test the QueueEmitter class."""

    def test_queue_emitter_emit_success(self):
        """Test successful emission to queue."""
        queue = mp.Manager().Queue()
        emitter = QueueEmitter(queue, SystemClock())

        result = emitter.emit('test_data')
        assert result is True

        # Verify the message was added to the queue
        message = queue.get_nowait()
        assert isinstance(message, Message)
        assert message.data == 'test_data'
        assert isinstance(message.ts, int)

    def test_queue_emitter_emit_with_timestamp(self):
        """Test emission with explicit timestamp."""
        queue = mp.Manager().Queue()
        emitter = QueueEmitter(queue, SystemClock())
        timestamp = 1234567890

        result = emitter.emit('test_data', ts=timestamp)
        assert result is True

        message = queue.get_nowait()
        assert message.data == 'test_data'
        assert message.ts == timestamp

    def test_queue_emitter_full_queue_removes_old_message(self):
        """Test that full queue removes old message before adding new one."""
        queue = mp.Manager().Queue(maxsize=1)
        emitter = QueueEmitter(queue, SystemClock())

        # Fill the queue
        emitter.emit('old_data')

        # Add another message (should remove old one)
        result = emitter.emit('new_data')
        assert result is True

        # Only new message should be in queue
        message = queue.get_nowait()
        assert message.data == 'new_data'

        # Queue should be empty now
        with pytest.raises(Empty):
            queue.get_nowait()

    @patch('multiprocessing.Queue')
    def test_queue_emitter_handles_full_exception(self, mock_queue_class):
        """Test handling of Full exception when queue put fails."""
        mock_queue = Mock()
        mock_queue.put_nowait.side_effect = Full()  # put fails
        mock_queue.get_nowait.side_effect = Full()  # get also fails (queue behavior)
        mock_queue_class.return_value = mock_queue

        emitter = QueueEmitter(mock_queue, SystemClock())
        result = emitter.emit('test_data')

        assert result is False
        mock_queue.put_nowait.assert_called_once()  # Only called once since get_nowait fails


class TestQueueReceiver:
    """Test the QueueReceiver class."""

    def test_queue_reader_initial_state(self):
        """Test that QueueReceiver initially returns None."""
        queue = mp.Manager().Queue()
        reader = QueueReceiver(queue)

        result = reader.read()
        assert result is None

    def test_queue_reader_reads_message(self):
        """Test reading a message from the queue."""
        manager = mp.Manager()
        queue = manager.Queue()
        reader = QueueReceiver(queue)

        # Put a message in the queue
        test_message = Message('test_data', 123)
        queue.put_nowait(test_message)

        result = reader.read()
        assert result == test_message
        assert result.data == 'test_data'
        assert result.ts == 123

    def test_queue_reader_returns_last_value_when_empty(self):
        """Test that reader returns last value when queue is empty."""
        manager = mp.Manager()
        queue = manager.Queue()
        reader = QueueReceiver(queue)

        # Put and read a message
        test_message = Message('test_data', 123)
        queue.put_nowait(test_message)
        first_result = reader.read()
        assert first_result == test_message

        # Queue is now empty, should return same message
        second_result = reader.read()
        assert second_result == test_message

    def test_queue_reader_updates_with_new_messages(self):
        """Test that reader updates with new messages."""
        manager = mp.Manager()
        queue = manager.Queue()
        reader = QueueReceiver(queue)

        # Put first message
        message1 = Message('data1', 100)
        queue.put_nowait(message1)
        result1 = reader.read()
        assert result1 == message1

        # Put second message
        message2 = Message('data2', 200)
        queue.put_nowait(message2)
        result2 = reader.read()
        assert result2 == message2

        # Should still return latest message when queue is empty
        result3 = reader.read()
        assert result3 == message2


class TestEventReceiver:
    """Test the EventReceiver class."""

    def test_event_reader_unset_event(self):
        """Test reading from an unset event."""
        event = mp.Event()
        reader = EventReceiver(event, SystemClock())

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is False
        assert isinstance(result.ts, int)

    def test_event_reader_set_event(self):
        """Test reading from a set event."""
        event = mp.Event()
        event.set()
        reader = EventReceiver(event, SystemClock())

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is True
        assert isinstance(result.ts, int)

    def test_event_reader_uses_clock(self):
        """Test that EventReceiver uses clocks for timestamps."""
        event = mp.Event()
        clk = MockClock()
        clk.set(0.987654321)
        reader = EventReceiver(event, clk)

        result = reader.read()
        assert result.ts == 987654321


class TestWorld:
    """Test the World class."""

    @pytest.mark.parametrize('pipe_fn_name', ['mp_pipe', 'local_pipe'])
    def test_world_pipe_creation(self, pipe_fn_name):
        """Test that World.pipe creates emitter and reader pair."""
        with World() as world:
            emitter, reader = getattr(world, pipe_fn_name)()

            assert isinstance(emitter, SignalEmitter)
            assert isinstance(reader, SignalReceiver)

    def test_background_process(self):
        """Test that background processes will run simple control loop."""
        world = World()
        with world:
            world.start_in_subprocess(dummy_process)

            time.sleep(0.2)  # Some time to let the process run
            assert len(world.background_processes) == 1

            # We have to set the private event manually, because out of the scope of the context manager
            # we can't access exit code of the process
            world._stop_event.set()
            world.background_processes[0].join(timeout=0.5)
            assert not world.background_processes[0].is_alive()
            assert world.background_processes[0].exitcode == 0

    @pytest.mark.parametrize('pipe_fn_name', ['mp_pipe', 'local_pipe'])
    def test_world_pipe_communication(self, pipe_fn_name):
        """Test that pipe emitter and reader can communicate."""
        with World() as world:
            emitter, reader = getattr(world, pipe_fn_name)()

            # Initially reader should return None
            assert reader.read() is None

            for message in ['test_message_1', 'test_message_2', 'test_message_3']:
                # Emit a message
                emitter.emit(message)

                # Reader should now have the message
                result = reader.read()
                assert isinstance(result, Message)
                assert result.data == message

    def test_world_context_manager_enter(self):
        """Test that World.__enter__ returns self."""
        world = World()
        with world as w:
            assert w is world

    def test_world_context_manager_should_stop(self):
        """Test that should_stop becomes True after exiting context."""
        world = World()

        # Initially should_stop is False
        assert world.should_stop is False

        # Use as context manager
        with world:
            assert world.should_stop is False

        # After exiting context, should_stop should be True
        assert world.should_stop is True

    def test_world_context_manager_stops_background_processes(self):
        """Test that background processes are stopped when exiting context."""
        world = World()

        with world:
            # Start a background process
            world.start_in_subprocess(dummy_process)

            # Verify process is running
            assert len(world.background_processes) == 1
            assert world.background_processes[0].is_alive()

            # should_stop should still be False while in context
            assert world.should_stop is False

        # After exiting context, should_stop should be True
        assert world.should_stop is True

        # Background processes should be terminated and cleaned up
        # We can't check is_alive() after exit because processes are closed
        assert len(world.background_processes) == 1

    def test_world_context_manager_with_exception(self):
        """Test that background processes are stopped even when exception occurs."""
        world = World()

        try:
            with world:
                # Start a background process
                world.start_in_subprocess(dummy_process)

                # Verify process is running
                assert len(world.background_processes) == 1
                assert world.background_processes[0].is_alive()

                # Raise an exception to test cleanup
                raise ValueError('Test exception')
        except ValueError:
            pass  # Expected exception

        # After exiting context (even with exception), should_stop should be True
        assert world.should_stop is True

        # Background processes should be terminated and cleaned up
        # We can't check is_alive() after exit because processes are closed
        assert len(world.background_processes) == 1

    def test_mp_pipe_uses_queue_for_non_shared_memory_payloads(self):
        with World() as world:
            emitter, reader = world.mp_pipe()

            emitter.emit('hello', ts=123)

            message = reader.read()
            assert message is not None
            assert message.data == 'hello'
            assert message.ts == 123
            assert hasattr(emitter, 'uses_shared_memory') and not emitter.uses_shared_memory
            assert hasattr(reader, 'uses_shared_memory') and not reader.uses_shared_memory

    def test_mp_pipe_switches_to_shared_memory_when_supported(self):
        with World() as world:
            emitter, reader = world.mp_pipe()

            payload = DummySMValue(3.14)
            assert emitter.emit(payload, ts=456)

            message = reader.read()
            assert message is not None
            assert isinstance(message.data, DummySMValue)
            assert message.data.value == pytest.approx(3.14)
            assert message.ts == 456
            assert emitter.uses_shared_memory
            assert reader.uses_shared_memory

    def test_mp_pipe_rejects_incompatible_payload_after_shared_memory_selected(self):
        with World() as world:
            emitter, _ = world.mp_pipe()

            emitter.emit(DummySMValue(1.0))

            with pytest.raises(TypeError, match='Shared memory transport selected'):  # type: ignore[arg-type]
                emitter.emit('not-compatible')


class TestWorldControlSystems:
    """Tests exercising ControlSystem wiring and scheduling."""

    def test_connect_enforces_unique_receiver(self):
        producer = DummyControlSystem('producer')
        consumer = DummyControlSystem('consumer')

        with World() as world:
            world.connect(producer.emitter, consumer.receiver)
            with pytest.raises(AssertionError):
                world.connect(producer.emitter, consumer.receiver)

    def test_mirror_from_emitter_creates_receiver_and_applies_wrapper(self):
        clock = MockClock(0.0)
        system = DummyControlSystem('loop')
        wrapper = Mock(side_effect=lambda emitter: emitter)

        with World(clock) as world:
            mirrored = world.pair(system.emitter, wrapper=wrapper)

            assert isinstance(mirrored, ControlSystemReceiver)
            wrapper.assert_not_called()

            scheduler = world.start(system)
            wrapper.assert_called_once_with(system.emitter)

            system.emitter.emit('payload')
            message = mirrored.read()
            assert message is not None
            assert message.data == 'payload'
            assert message.ts == 0

            scheduler.close()

    def test_mirror_from_receiver_creates_emitter_and_applies_wrapper(self):
        clock = MockClock(0.0)
        system = DummyControlSystem('loop')
        wrapper = Mock(side_effect=lambda receiver: receiver)

        with World(clock) as world:
            mirrored = world.pair(system.receiver, wrapper=wrapper)

            assert isinstance(mirrored, ControlSystemEmitter)
            wrapper.assert_not_called()

            scheduler = world.start(system)
            wrapper.assert_called_once_with(system.receiver)

            mirrored.emit('payload')
            message = system.receiver.read()
            assert message is not None
            assert message.data == 'payload'
            assert message.ts == 0

            scheduler.close()

    def test_mirror_rejects_unknown_connector(self):
        with World() as world:
            with pytest.raises(ValueError, match='Unsupported connector type'):
                world.pair(object())

    def test_start_sets_up_local_connections(self):
        clock = MockClock(0.0)
        producer = DummyControlSystem('producer')
        consumer = DummyControlSystem('consumer')
        with World(clock) as world:
            world.connect(producer.emitter, consumer.receiver)

            scheduler = world.start([producer, consumer])

            producer.emitter.emit('payload')
            result = consumer.receiver.read()
            assert result is not None
            assert result.data == 'payload'
            assert result.ts == 0

            sleeps = list(scheduler)
            assert [cmd.seconds for cmd in sleeps] == [0.0, 0.0]
            assert world.should_stop

            assert len(producer.invocations) == 1
            assert len(consumer.invocations) == 1
            producer_stop_reader, producer_clock = producer.invocations[0]
            consumer_stop_reader, consumer_clock = consumer.invocations[0]
            assert isinstance(producer_stop_reader, EventReceiver)
            assert isinstance(consumer_stop_reader, EventReceiver)
            assert producer_clock is clock
            assert consumer_clock is clock

    def test_start_uses_mp_pipe_for_cross_process_connections(self, monkeypatch):
        clock = MockClock(0.0)
        main_cs = DummyControlSystem('main')
        background_cs = DummyControlSystem('background')

        captured_clocks = []

        def fake_mp_pipe(self, maxsize=1, clock=None):
            captured_clocks.append(clock)
            return self.local_pipe(maxsize)

        monkeypatch.setattr(World, 'mp_pipe', fake_mp_pipe)

        started_background = []

        def fake_start_in_subprocess(self, *loops):
            started_background.append(loops)

        monkeypatch.setattr(World, 'start_in_subprocess', fake_start_in_subprocess)

        with World(clock) as world:
            world.connect(background_cs.emitter, main_cs.receiver)

            scheduler = world.start(main_process=main_cs, background=background_cs)

            background_cs.emitter.emit('payload')
            result = main_cs.receiver.read()
            assert result is not None
            assert result.data == 'payload'

            assert captured_clocks and isinstance(captured_clocks[0], SystemClock)
            assert started_background == [(background_cs.run,)]
            assert background_cs.invocations == []

            sleeps = list(scheduler)
            assert [cmd.seconds for cmd in sleeps] == [0.0]
            assert len(main_cs.invocations) == 1
            stop_reader, used_clock = main_cs.invocations[0]
            assert isinstance(stop_reader, EventReceiver)
            assert used_clock is clock

    def test_start_handles_empty_main_process(self, monkeypatch):
        clock = MockClock(0.0)
        background_cs = DummyControlSystem('background')

        started_background = []

        def fake_start_in_subprocess(self, *loops):
            started_background.append(loops)

        monkeypatch.setattr(World, 'start_in_subprocess', fake_start_in_subprocess)

        with World(clock) as world:
            scheduler = world.start([], background=background_cs)

            assert started_background == [(background_cs.run,)]
            assert list(scheduler) == []

            scheduler.close()

    def test_start_requires_known_emitter_owner(self):
        known = DummyControlSystem('known')
        unknown = DummyControlSystem('unknown')

        with World() as world:
            world.connect(unknown.emitter, known.receiver)

            with pytest.raises(ValueError, match='Emitter .* is not in any control system'):
                world.start(main_process=known)

    def test_start_requires_known_receiver_owner(self):
        producer = DummyControlSystem('producer')
        missing_consumer = DummyControlSystem('missing_consumer')

        with World() as world:
            world.connect(producer.emitter, missing_consumer.receiver)

            with pytest.raises(ValueError, match='Receiver .* is not in any control system'):
                world.start(main_process=producer)


# Integration tests
class TestIntegration:
    """Integration tests for world components."""

    def test_full_pipeline(self):
        """Test a complete pipeline with World, emitters, and readers."""
        with World() as world:
            # Create communication channels
            emitter1, reader1 = world.mp_pipe()
            emitter2, reader2 = world.mp_pipe()

            # Test data flow
            emitter1.emit('message1')
            emitter2.emit('message2')

            result1 = reader1.read()
            result2 = reader2.read()

            assert result1.data == 'message1'
            assert result2.data == 'message2'

    def test_event_reader_integration(self):
        """Test EventReceiver with actual multiprocessing Event."""
        event = mp.Event()
        reader = EventReceiver(event, SystemClock())

        # Initially event is not set
        result = reader.read()
        assert result.data is False

        # Set the event
        event.set()
        result = reader.read()
        assert result.data is True

        # Clear the event
        event.clear()
        result = reader.read()
        assert result.data is False


class TestWorldInterleave:
    """Test the World.interleave method with comprehensive scenarios."""

    def test_single_loop(self):
        """Test interleaving with multiple scenarios: single loop, multiple loops, timing, and scheduling."""
        clock = MockClock(0.0)

        with World(clock) as world:
            execution_order = []

            # Test single loop
            def single_loop(stop_reader, clock):
                """Simple loop that runs 2 times."""
                for i in range(2):
                    execution_order.append(f'single_{i}')
                    yield Sleep(0.1)

            sleep_times = list(world.interleave(single_loop))

            # Should have 2 sleep times - one after each execution
            assert len(sleep_times) == 2
            assert execution_order == ['single_0', 'single_1']

            # Stop event should be set after loop completes
            assert world.should_stop

    def test_two_loops(self):
        clock = MockClock(0.0)
        execution_order = []

        with World(clock) as world:

            def loop_a(stop_reader, clock):
                for i in range(2):
                    execution_order.append(f'a_{i}')
                    yield Sleep(0.1)

            def loop_b(stop_reader, clock):
                for i in range(2):
                    execution_order.append(f'b_{i}')
                    yield Sleep(0.1)

            sleep_times = list(world.interleave(loop_a, loop_b))

            assert len(sleep_times) == 4

            # Both loops should have executed all their steps
            assert len([item for item in execution_order if item.startswith('a_')]) == 2
            assert len([item for item in execution_order if item.startswith('b_')]) == 2

            # Stop event should be set after first loop completes
            assert world.should_stop

    def test_no_loops(self):
        clock = MockClock(100.0)

        with World(clock) as world:
            sleep_times = list(world.interleave())
            assert len(sleep_times) == 0
            assert not world.should_stop

        # Test exception handling

    def test_failing_loop(self):
        clock = MockClock(100.0)
        with World(clock) as world:
            execution_order = []

            def failing_loop(stop_reader, clock):
                """Loop that raises an exception."""
                execution_order.append('before_exception')
                raise ValueError('Test exception')
                yield Sleep(0.1)  # This should never be reached

            # The exception should be raised and stop the interleave
            with pytest.raises(ValueError, match='Test exception'):
                list(world.interleave(failing_loop))

            assert 'before_exception' in execution_order

    def test_interleave_stop_behavior(self):
        """Test stop event behavior: early stopping and completion detection."""
        clock = MockClock(0.0)

        with World(clock) as world:
            execution_order = []

            def stop_checking_loop(stop_reader, clock):
                """Loop that checks stop signal and exits early."""
                for i in range(10):  # Would run 10 times if not stopped
                    if stop_reader.value:
                        execution_order.append(f'stopped_at_{i}')
                        return
                    execution_order.append(f'step_{i}')
                    yield Sleep(0.1)

            def short_loop(stop_reader, clock):
                """Short loop that completes quickly."""
                for i in range(2):
                    execution_order.append(f'short_{i}')
                    yield Sleep(0.1)

            # The short loop should complete first and set the stop event
            sleep_times = list(world.interleave(stop_checking_loop, short_loop))

            # Both loops should run some steps
            assert len(sleep_times) >= 4
            assert world.should_stop

            # Should have some execution from both loops
            assert any(item.startswith('step_') for item in execution_order)
            assert any(item.startswith('short_') for item in execution_order)

            # The stop_checking_loop should detect the stop event and exit early
            assert any(item.startswith('stopped_at_') for item in execution_order)

    def test_interleave_scheduling_order(self):
        """Test that loops are scheduled in the correct order based on their sleep times."""
        clock = MockClock(0.0)

        with World(clock) as world:
            execution_order = []

            def loop_a(stop_reader, clock):
                """Loop A with specific timing."""
                execution_order.append('a_0')
                yield Sleep(0.3)  # Will run next at time 0.3
                execution_order.append('a_1')
                yield Sleep(0.1)  # Will run next at time 0.4

            def loop_b(stop_reader, clock):
                """Loop B with different timing."""
                execution_order.append('b_0')
                yield Sleep(0.1)  # Will run next at time 0.1
                execution_order.append('b_1')
                yield Sleep(0.1)  # Will run next at time 0.2

            sleep_times = list(world.interleave(loop_a, loop_b))

            # Expected execution order based on scheduling:
            # t=0.0: a_0, b_0 (both start simultaneously)
            # t=0.1: b_1 (loop_b scheduled first)
            # t=0.2: (no loops ready)
            # t=0.3: a_1 (loop_a scheduled next)
            # Should have 4 steps total
            assert len(sleep_times) == 4
            assert execution_order == ['a_0', 'b_0', 'b_1', 'a_1']

    def test_interleave_introducing_new_loop_not_affect_order_of_existing_loops(self):
        clock = MockClock(0.0)
        execution_order = []

        def loop_a(stop_reader, clock):
            for i in range(5):
                execution_order.append(f'a_{i}')
                yield Sleep(0.1)

        def loop_b(stop_reader, clock):
            for i in range(6):
                execution_order.append(f'b_{i}')
            yield Sleep(0.2)

        def loop_c(stop_reader, clock):
            for i in range(7):
                execution_order.append(f'c_{i}')
                yield Sleep(0.3)

        with World(clock) as world:
            for time_to_sleep in world.interleave(loop_a, loop_b):
                clock.advance(time_to_sleep.seconds)
            original_order = execution_order.copy()
            execution_order.clear()
            for time_to_sleep in world.interleave(loop_c, loop_a, loop_c, loop_b, loop_c):
                clock.advance(time_to_sleep.seconds)

        execution_order = [item for item in execution_order if not item.startswith('c_')]
        assert execution_order == original_order

    def test_iterleave_loops_with_sleep_0_execute_interchangeably(self):
        clock = MockClock(0.0)
        execution_order = []

        def loop_a(stop_reader, clock):
            for i in range(4):
                execution_order.append(f'a_{i}')
                yield Sleep(0.0)

        def loop_b(stop_reader, clock):
            for i in range(4):
                execution_order.append(f'b_{i}')
                yield Sleep(0.0)

        with World(clock) as world:
            for time_to_sleep in world.interleave(loop_a, loop_b):
                clock.advance(time_to_sleep.seconds)
            original_order = execution_order.copy()

            assert original_order == ['a_0', 'b_0', 'a_1', 'b_1', 'a_2', 'b_2', 'a_3', 'b_3']
