import pytest
import multiprocessing as mp
from queue import Empty, Full
from unittest.mock import Mock, patch

from ironic2.core import Message, NoValue
from ironic2.world import (QueueEmitter, QueueReader, EventReader, World)


class TestQueueEmitter:
    """Test the QueueEmitter class."""

    def test_queue_emitter_emit_success(self):
        """Test successful emission to queue."""
        queue = mp.Queue()
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
        queue = mp.Queue()
        emitter = QueueEmitter(queue)
        timestamp = 1234567890

        result = emitter.emit("test_data", ts=timestamp)
        assert result is True

        message = queue.get_nowait()
        assert message.data == "test_data"
        assert message.ts == timestamp

    def test_queue_emitter_full_queue_removes_old_message(self):
        """Test that full queue removes old message before adding new one."""
        queue = mp.Queue(maxsize=1)
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
        mock_queue.full.return_value = False
        mock_queue.put_nowait.side_effect = Full()
        mock_queue_class.return_value = mock_queue

        emitter = QueueEmitter(mock_queue)
        result = emitter.emit("test_data")

        assert result is False
        mock_queue.put_nowait.assert_called_once()


class TestQueueReader:
    """Test the QueueReader class."""

    def test_queue_reader_initial_state(self):
        """Test that QueueReader initially returns NoValue."""
        queue = mp.Queue()
        reader = QueueReader(queue)

        result = reader.value()
        assert result is NoValue

    def test_queue_reader_reads_message(self):
        """Test reading a message from the queue."""
        queue = mp.Queue()
        reader = QueueReader(queue)

        # Put a message in the queue
        test_message = Message("test_data", 123)
        queue.put_nowait(test_message)

        result = reader.value()
        assert result == test_message
        assert result.data == "test_data"
        assert result.ts == 123

    def test_queue_reader_returns_last_value_when_empty(self):
        """Test that reader returns last value when queue is empty."""
        queue = mp.Queue()
        reader = QueueReader(queue)

        # Put and read a message
        test_message = Message("test_data", 123)
        queue.put_nowait(test_message)
        first_result = reader.value()
        assert first_result == test_message

        # Queue is now empty, should return same message
        second_result = reader.value()
        assert second_result == test_message

    def test_queue_reader_updates_with_new_messages(self):
        """Test that reader updates with new messages."""
        queue = mp.Queue()
        reader = QueueReader(queue)

        # Put first message
        message1 = Message("data1", 100)
        queue.put_nowait(message1)
        result1 = reader.value()
        assert result1 == message1

        # Put second message
        message2 = Message("data2", 200)
        queue.put_nowait(message2)
        result2 = reader.value()
        assert result2 == message2

        # Should still return latest message when queue is empty
        result3 = reader.value()
        assert result3 == message2


class TestEventReader:
    """Test the EventReader class."""

    def test_event_reader_unset_event(self):
        """Test reading from an unset event."""
        event = mp.Event()
        reader = EventReader(event)

        result = reader.value()
        assert isinstance(result, Message)
        assert result.data is False
        assert isinstance(result.ts, int)

    def test_event_reader_set_event(self):
        """Test reading from a set event."""
        event = mp.Event()
        event.set()
        reader = EventReader(event)

        result = reader.value()
        assert isinstance(result, Message)
        assert result.data is True
        assert isinstance(result.ts, int)

    @patch('ironic2.world.system_clock')
    def test_event_reader_uses_system_clock(self, mock_system_clock):
        """Test that EventReader uses system_clock for timestamps."""
        mock_system_clock.return_value = 987654321
        event = mp.Event()
        reader = EventReader(event)

        result = reader.value()
        assert result.ts == 987654321
        mock_system_clock.assert_called_once()


class TestWorld:
    """Test the World class."""

    def test_world_pipe_creation(self):
        """Test that World.pipe creates emitter and reader pair."""
        world = World()
        emitter, reader = world.pipe()

        assert isinstance(emitter, QueueEmitter)
        assert isinstance(reader, QueueReader)

    def test_world_pipe_communication(self):
        """Test that pipe emitter and reader can communicate."""
        world = World()
        emitter, reader = world.pipe()

        # Initially reader should return NoValue
        assert reader.value() is NoValue

        # Emit a message
        emitter.emit("test_message")

        # Reader should now have the message
        result = reader.value()
        assert isinstance(result, Message)
        assert result.data == "test_message"


# Integration tests
class TestIntegration:
    """Integration tests for world components."""

    def test_full_pipeline(self):
        """Test a complete pipeline with World, emitters, and readers."""
        world = World()

        # Create communication channels
        emitter1, reader1 = world.pipe()
        emitter2, reader2 = world.pipe()

        # Test data flow
        emitter1.emit("message1")
        emitter2.emit("message2")

        result1 = reader1.value()
        result2 = reader2.value()

        assert result1.data == "message1"
        assert result2.data == "message2"

    def test_event_reader_integration(self):
        """Test EventReader with actual multiprocessing Event."""
        event = mp.Event()
        reader = EventReader(event)

        # Initially event is not set
        result = reader.value()
        assert result.data is False

        # Set the event
        event.set()
        result = reader.value()
        assert result.data is True

        # Clear the event
        event.clear()
        result = reader.value()
        assert result.data is False
