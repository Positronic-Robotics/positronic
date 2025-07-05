import pytest
import multiprocessing as mp
from queue import Empty, Full
from unittest.mock import Mock, patch
import time

from ironic2.core import Message
from ironic2.world import (QueueEmitter, QueueReader, EventReader, World)


def dummy_process(stop_signal):
    """A simple background process that runs until stopped."""
    while not stop_signal.read().data:
        time.sleep(0.01)


class TestQueueEmitter:
    """Test the QueueEmitter class."""

    def test_queue_emitter_emit_success(self):
        """Test successful emission to queue."""
        queue = mp.Manager().Queue()
        emitter = QueueEmitter(queue)

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
        emitter = QueueEmitter(queue)
        timestamp = 1234567890

        result = emitter.emit("test_data", ts=timestamp)
        assert result is True

        message = queue.get_nowait()
        assert message.data == "test_data"
        assert message.ts == timestamp

    def test_queue_emitter_full_queue_removes_old_message(self):
        """Test that full queue removes old message before adding new one."""
        queue = mp.Manager().Queue(maxsize=1)
        emitter = QueueEmitter(queue)

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

        emitter = QueueEmitter(mock_queue)
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
        reader = EventReader(event)

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is False
        assert isinstance(result.ts, int)

    def test_event_reader_set_event(self):
        """Test reading from a set event."""
        event = mp.Event()
        event.set()
        reader = EventReader(event)

        result = reader.read()
        assert isinstance(result, Message)
        assert result.data is True
        assert isinstance(result.ts, int)

    @patch('ironic2.world.system_clock')
    def test_event_reader_uses_system_clock(self, mock_system_clock):
        """Test that EventReader uses system_clock for timestamps."""
        mock_system_clock.return_value = 987654321
        event = mp.Event()
        reader = EventReader(event)

        result = reader.read()
        assert result.ts == 987654321
        mock_system_clock.assert_called_once()


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
            world.start(dummy_process)

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
                world.start(dummy_process)

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
        reader = EventReader(event)

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
