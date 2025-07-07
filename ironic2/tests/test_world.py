import pytest
import multiprocessing as mp
from queue import Empty, Full
from unittest.mock import Mock, patch

from ironic2.core import Clock, Message
from ironic2.world import (QueueEmitter, QueueReader, EventReader, SystemClock, World)


class MockClock(Clock):
    """Mock clock that can be controlled for testing."""

    def __init__(self, start_time: float = 0.0):
        self._time = start_time

    def now(self) -> float:
        return self._time

    def now_ns(self) -> int:
        return int(self._time * 1e9)

    def advance(self, delta: float):
        """Advance the clock by delta seconds."""
        self._time += delta


def dummy_process(stop_signal):
    """A simple background process that runs until stopped."""
    while not stop_signal.read().data:
        yield 0.01


class TestQueueEmitter:
    """Test the QueueEmitter class."""

    def test_queue_emitter_emit_success(self):
        """Test successful emission to queue."""
        queue = mp.Manager().Queue()
        emitter = QueueEmitter(queue, SystemClock())

        result = emitter.emit("test_data")
        assert result is True

        # Verify the message was added to the queue
        message = queue.get_nowait()
        assert isinstance(message, Message)
        assert message.data == "test_data"
        assert isinstance(message.ts, int)

    def test_queue_emitter_emit_with_timestamp(self):
        """Test emission with explicit timestamp."""
        queue = mp.Manager().Queue()
        emitter = QueueEmitter(queue, SystemClock())
        timestamp = 1234567890

        result = emitter.emit("test_data", ts=timestamp)
        assert result is True

        message = queue.get_nowait()
        assert message.data == "test_data"
        assert message.ts == timestamp

    def test_queue_emitter_full_queue_removes_old_message(self):
        """Test that full queue removes old message before adding new one."""
        queue = mp.Manager().Queue(maxsize=1)
        emitter = QueueEmitter(queue, SystemClock())

        # Fill the queue
        emitter.emit("old_data")

        # Add another message (should remove old one)
        result = emitter.emit("new_data")
        assert result is True

        # Only new message should be in queue
        message = queue.get_nowait()
        assert message.data == "new_data"

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
        result = emitter.emit("test_data")

        assert result is False
        mock_queue.put_nowait.assert_called_once()  # Only called once since get_nowait fails


class TestQueueReader:
    """Test the QueueReader class."""

    def test_queue_reader_initial_state(self):
        """Test that QueueReader initially returns None."""
        queue = mp.Manager().Queue()
        reader = QueueReader(queue)

        result = reader.read()
        assert result is None

    def test_queue_reader_reads_message(self):
        """Test reading a message from the queue."""
        manager = mp.Manager()
        queue = manager.Queue()
        reader = QueueReader(queue)

        # Put a message in the queue
        test_message = Message("test_data", 123)
        queue.put_nowait(test_message)

        result = reader.read()
        assert result == test_message
        assert result.data == "test_data"
        assert result.ts == 123

    def test_queue_reader_returns_last_value_when_empty(self):
        """Test that reader returns last value when queue is empty."""
        manager = mp.Manager()
        queue = manager.Queue()
        reader = QueueReader(queue)

        # Put and read a message
        test_message = Message("test_data", 123)
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
        reader = QueueReader(queue)

        # Put first message
        message1 = Message("data1", 100)
        queue.put_nowait(message1)
        result1 = reader.read()
        assert result1 == message1

        # Put second message
        message2 = Message("data2", 200)
        queue.put_nowait(message2)
        result2 = reader.read()
        assert result2 == message2

        # Should still return latest message when queue is empty
        result3 = reader.read()
        assert result3 == message2


class TestEventReader:
    """Test the EventReader class."""

    def test_event_reader_unset_event(self):
        """Test reading from an unset event."""
        event = mp.Event()
        reader = EventReader(event, SystemClock())

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is False
        assert isinstance(result.ts, int)

    def test_event_reader_set_event(self):
        """Test reading from a set event."""
        event = mp.Event()
        event.set()
        reader = EventReader(event, SystemClock())

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is True
        assert isinstance(result.ts, int)

    def test_event_reader_uses_clock(self):
        """Test that EventReader uses clocks for timestamps."""
        class MockClock(Clock):
            def now(self) -> float:
                return 0.987654321

            def now_ns(self) -> int:
                return 987654321

        event = mp.Event()
        reader = EventReader(event, MockClock())

        result = reader.read()
        assert result.ts == 987654321


class TestWorld:
    """Test the World class."""

    def test_world_pipe_creation(self):
        """Test that World.pipe creates emitter and reader pair."""
        world = World()
        emitter, reader = world.mp_pipe()

        assert isinstance(emitter, QueueEmitter)
        assert isinstance(reader, QueueReader)

    def test_world_pipe_communication(self):
        """Test that pipe emitter and reader can communicate."""
        world = World()
        emitter, reader = world.mp_pipe()

        # Initially reader should return None
        assert reader.read() is None

        # Emit a message
        emitter.emit("test_message")

        # Reader should now have the message
        result = reader.read()
        assert isinstance(result, Message)
        assert result.data == "test_message"

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
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # After exiting context (even with exception), should_stop should be True
        assert world.should_stop is True

        # Background processes should be terminated and cleaned up
        # We can't check is_alive() after exit because processes are closed
        assert len(world.background_processes) == 1


# Integration tests
class TestIntegration:
    """Integration tests for world components."""

    def test_full_pipeline(self):
        """Test a complete pipeline with World, emitters, and readers."""
        world = World()

        # Create communication channels
        emitter1, reader1 = world.mp_pipe()
        emitter2, reader2 = world.mp_pipe()

        # Test data flow
        emitter1.emit("message1")
        emitter2.emit("message2")

        result1 = reader1.read()
        result2 = reader2.read()

        assert result1.data == "message1"
        assert result2.data == "message2"

    def test_event_reader_integration(self):
        """Test EventReader with actual multiprocessing Event."""
        event = mp.Event()
        reader = EventReader(event, SystemClock())

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
                    execution_order.append(f"single_{i}")
                    yield 0.1

            sleep_times = list(world.interleave(single_loop))

            # Should have 2 sleep times - one after each execution
            assert len(sleep_times) == 2
            assert execution_order == ["single_0", "single_1"]

            # Stop event should be set after loop completes
            assert world.should_stop

    def test_two_loops(self):
        clock = MockClock(0.0)
        execution_order = []

        with World(clock) as world:
            def loop_a(stop_reader, clock):
                for i in range(2):
                    execution_order.append(f"a_{i}")
                    yield 0.1

            def loop_b(stop_reader, clock):
                for i in range(2):
                    execution_order.append(f"b_{i}")
                    yield 0.1

            sleep_times = list(world.interleave(loop_a, loop_b))

            assert len(sleep_times) == 4

            # Both loops should have executed all their steps
            assert len([item for item in execution_order if item.startswith("a_")]) == 2
            assert len([item for item in execution_order if item.startswith("b_")]) == 2

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
                execution_order.append("before_exception")
                raise ValueError("Test exception")
                yield 0.1  # This should never be reached

            # The exception should be raised and stop the interleave
            with pytest.raises(ValueError, match="Test exception"):
                list(world.interleave(failing_loop))

            assert "before_exception" in execution_order

    def test_interleave_stop_behavior(self):
        """Test stop event behavior: early stopping and completion detection."""
        clock = MockClock(0.0)

        with World(clock) as world:
            execution_order = []

            def stop_checking_loop(stop_reader, clock):
                """Loop that checks stop signal and exits early."""
                for i in range(10):  # Would run 10 times if not stopped
                    if stop_reader.value:
                        execution_order.append(f"stopped_at_{i}")
                        return
                    execution_order.append(f"step_{i}")
                    yield 0.1

            def short_loop(stop_reader, clock):
                """Short loop that completes quickly."""
                for i in range(2):
                    execution_order.append(f"short_{i}")
                    yield 0.1

            # The short loop should complete first and set the stop event
            sleep_times = list(world.interleave(stop_checking_loop, short_loop))

            # Both loops should run some steps
            assert len(sleep_times) >= 4
            assert world.should_stop

            # Should have some execution from both loops
            assert any(item.startswith("step_") for item in execution_order)
            assert any(item.startswith("short_") for item in execution_order)

            # The stop_checking_loop should detect the stop event and exit early
            assert any(item.startswith("stopped_at_") for item in execution_order)

    def test_interleave_scheduling_order(self):
        """Test that loops are scheduled in the correct order based on their sleep times."""
        clock = MockClock(0.0)

        with World(clock) as world:
            execution_order = []

            def loop_a(stop_reader, clock):
                """Loop A with specific timing."""
                execution_order.append("a_0")
                yield 0.3  # Will run next at time 0.3
                execution_order.append("a_1")
                yield 0.1  # Will run next at time 0.4

            def loop_b(stop_reader, clock):
                """Loop B with different timing."""
                execution_order.append("b_0")
                yield 0.1  # Will run next at time 0.1
                execution_order.append("b_1")
                yield 0.1  # Will run next at time 0.2

            sleep_times = list(world.interleave(loop_a, loop_b))

            # Expected execution order based on scheduling:
            # t=0.0: a_0, b_0 (both start simultaneously)
            # t=0.1: b_1 (loop_b scheduled first)
            # t=0.2: (no loops ready)
            # t=0.3: a_1 (loop_a scheduled next)
            # Should have 4 steps total
            assert len(sleep_times) == 4
            assert execution_order == ["a_0", "b_0", "b_1", "a_1"]
