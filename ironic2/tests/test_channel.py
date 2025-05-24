import pytest
from typing import List

from ironic2 import Message, NoValue, Channel, LastValueChannel, DuplicateChannel


class MockChannel(Channel):
    """Mock channel implementation for testing purposes."""

    def __init__(self):
        self.written_messages: List[Message] = []
        self.read_queue: List[Message] = []
        self.read_index = 0

    def write(self, message: Message):
        """Store written messages for verification."""
        self.written_messages.append(message)

    def read(self):
        """Return messages from read_queue or NoValue if empty."""
        if self.read_index < len(self.read_queue):
            message = self.read_queue[self.read_index]
            self.read_index += 1
            return message
        return NoValue

    def set_read_queue(self, messages: List[Message]):
        """Set up messages to be returned by read()."""
        self.read_queue = messages
        self.read_index = 0


class TestLastValueChannel:
    """Tests for LastValueChannel class."""

    def test_init(self):
        """Test LastValueChannel initialization."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        assert last_value_channel.base == mock_channel
        assert last_value_channel.last_value is NoValue

    def test_write_delegates_to_base_channel(self):
        """Test that write operations are delegated to the base channel."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        message = Message("test_data", 12345)
        last_value_channel.write(message)

        assert len(mock_channel.written_messages) == 1
        assert mock_channel.written_messages[0] == message

    def test_read_returns_no_value_initially(self):
        """Test that reading returns NoValue when no messages have been read."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        result = last_value_channel.read()
        assert result is NoValue

    def test_read_stores_and_returns_last_value(self):
        """Test that reading stores the last value and returns it."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        message1 = Message("first", 1000)
        message2 = Message("second", 2000)

        # Set up the mock to return messages
        mock_channel.set_read_queue([message1, message2])

        # First read should return and store first message
        result1 = last_value_channel.read()
        assert result1 == message1
        assert last_value_channel.last_value == message1

        # Second read should return and store second message
        result2 = last_value_channel.read()
        assert result2 == message2
        assert last_value_channel.last_value == message2

    def test_read_returns_last_value_when_base_has_no_value(self):
        """Test that reading returns the last stored value when base channel has no new data."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        message = Message("stored_value", 1000)

        # Set up one message to be read
        mock_channel.set_read_queue([message])

        # First read stores the message
        first_result = last_value_channel.read()
        assert first_result == message

        # Second read should return the same message since base has no new data
        second_result = last_value_channel.read()
        assert second_result == message
        assert last_value_channel.last_value == message

    def test_read_continues_returning_last_value_after_no_value(self):
        """Test that once a value is stored, it continues to be returned even after NoValue from base."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        message = Message("persistent_value", 1000)
        mock_channel.set_read_queue([message])

        # Read the message
        result1 = last_value_channel.read()
        assert result1 == message

        # Read again (base will return NoValue)
        result2 = last_value_channel.read()
        assert result2 == message

        # Read multiple times
        for _ in range(5):
            result = last_value_channel.read()
            assert result == message

    def test_read_updates_last_value_with_new_messages(self):
        """Test that last_value is updated when new messages are available."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        message1 = Message("first", 1000)
        message2 = Message("second", 2000)
        message3 = Message("third", 3000)

        # Read first message
        mock_channel.set_read_queue([message1])
        result1 = last_value_channel.read()
        assert result1 == message1

        # Read when no new messages (should return last)
        result2 = last_value_channel.read()
        assert result2 == message1

        # Add more messages and continue reading
        mock_channel.set_read_queue([message1, message2, message3])
        mock_channel.read_index = 1  # Skip the first message we already read

        result3 = last_value_channel.read()
        assert result3 == message2
        assert last_value_channel.last_value == message2

        result4 = last_value_channel.read()
        assert result4 == message3
        assert last_value_channel.last_value == message3


class TestDuplicateChannel:
    """Tests for DuplicateChannel class."""

    def test_init_with_multiple_channels(self):
        """Test DuplicateChannel initialization with multiple channels."""
        channel1 = MockChannel()
        channel2 = MockChannel()
        channel3 = MockChannel()

        duplicate_channel = DuplicateChannel(channel1, channel2, channel3)

        assert duplicate_channel.channels == (channel1, channel2, channel3)

    def test_init_with_no_channels(self):
        """Test DuplicateChannel initialization with no channels."""
        duplicate_channel = DuplicateChannel()
        assert duplicate_channel.channels == ()

    def test_write_forwards_to_all_channels(self):
        """Test that write operation forwards messages to all channels."""
        channel1 = MockChannel()
        channel2 = MockChannel()
        channel3 = MockChannel()

        duplicate_channel = DuplicateChannel(channel1, channel2, channel3)
        message = Message("broadcast_data", 5000)

        duplicate_channel.write(message)

        # Check that all channels received the message
        assert len(channel1.written_messages) == 1
        assert len(channel2.written_messages) == 1
        assert len(channel3.written_messages) == 1

        assert channel1.written_messages[0] == message
        assert channel2.written_messages[0] == message
        assert channel3.written_messages[0] == message

    def test_write_with_no_channels(self):
        """Test that write operation works even with no channels."""
        duplicate_channel = DuplicateChannel()
        message = Message("no_recipients", 6000)

        # Should not raise an exception
        duplicate_channel.write(message)

    def test_write_multiple_messages(self):
        """Test writing multiple messages to multiple channels."""
        channel1 = MockChannel()
        channel2 = MockChannel()

        duplicate_channel = DuplicateChannel(channel1, channel2)

        message1 = Message("first_message", 1000)
        message2 = Message("second_message", 2000)
        message3 = Message("third_message", 3000)

        duplicate_channel.write(message1)
        duplicate_channel.write(message2)
        duplicate_channel.write(message3)

        # Check that both channels received all messages
        assert len(channel1.written_messages) == 3
        assert len(channel2.written_messages) == 3

        assert channel1.written_messages[0] == message1
        assert channel1.written_messages[1] == message2
        assert channel1.written_messages[2] == message3

        assert channel2.written_messages[0] == message1
        assert channel2.written_messages[1] == message2
        assert channel2.written_messages[2] == message3

    def test_read_raises_value_error(self):
        """Test that read operation raises ValueError."""
        channel1 = MockChannel()
        channel2 = MockChannel()

        duplicate_channel = DuplicateChannel(channel1, channel2)

        with pytest.raises(ValueError, match="Duplicate Channel is write only"):
            duplicate_channel.read()

    def test_write_with_single_channel(self):
        """Test writing to a single channel."""
        channel = MockChannel()
        duplicate_channel = DuplicateChannel(channel)

        message = Message("single_channel_data", 7000)
        duplicate_channel.write(message)

        assert len(channel.written_messages) == 1
        assert channel.written_messages[0] == message


class TestMessage:
    """Additional tests for Message class behavior with channels."""

    def test_message_with_explicit_timestamp(self):
        """Test that Message preserves explicit timestamps through channels."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        explicit_timestamp = 999999
        message = Message("test_data", explicit_timestamp)

        last_value_channel.write(message)

        written_message = mock_channel.written_messages[0]
        assert written_message.timestamp == explicit_timestamp
        assert written_message.data == "test_data"

    def test_message_auto_timestamp_through_channels(self):
        """Test that Message auto-generates timestamp when passed through channels."""
        mock_channel = MockChannel()
        last_value_channel = LastValueChannel(mock_channel)

        # Message without explicit timestamp
        message = Message("auto_timestamp_data")

        last_value_channel.write(message)

        written_message = mock_channel.written_messages[0]
        assert written_message.timestamp is not None
        assert isinstance(written_message.timestamp, int)
        assert written_message.data == "auto_timestamp_data"
