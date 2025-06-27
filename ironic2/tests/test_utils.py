from unittest.mock import Mock

from ironic2.core import Message, SignalReader
from ironic2.utils import ValueUpdated, DefaultReader


class TestValueUpdated:
    """Test the ValueUpdated class."""

    def test_first_read_with_data_returns_updated(self):
        """Test that first read with data returns updated=True."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        mock_reader.read.return_value = Message(data=test_data, ts=123)

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.read()

        assert result is not None
        assert result.data == (test_data, True)
        assert result.ts == 123

    def test_first_read_with_none_returns_none(self):
        """Test that first read with None returns None."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = None

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.read()

        assert result is None

    def test_same_timestamp_returns_not_updated(self):
        """Test that reading the same timestamp returns updated=False."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        message = Message(data=test_data, ts=123)
        mock_reader.read.return_value = message

        value_updated = ValueUpdated(mock_reader)

        # First read
        result1 = value_updated.read()
        assert result1.data[1] is True  # First read should be updated

        # Second read with same timestamp
        result2 = value_updated.read()
        assert result2.data[1] is False  # Same timestamp should not be updated

    def test_different_timestamp_returns_updated(self):
        """Test that reading different timestamps returns updated=True."""
        mock_reader = Mock(spec=SignalReader)
        test_data1 = "test_data1"
        test_data2 = "test_data2"

        value_updated = ValueUpdated(mock_reader)

        # First read
        mock_reader.read.return_value = Message(data=test_data1, ts=123)
        result1 = value_updated.read()
        assert result1.data == (test_data1, True)

        # Second read with different timestamp
        mock_reader.read.return_value = Message(data=test_data2, ts=456)
        result2 = value_updated.read()
        assert result2.data == (test_data2, True)

    def test_timestamp_tracking_persists_across_calls(self):
        """Test that timestamp tracking persists across multiple calls."""
        mock_reader = Mock(spec=SignalReader)

        value_updated = ValueUpdated(mock_reader)

        # Multiple reads with same timestamp
        mock_reader.read.return_value = Message(data="data1", ts=100)
        result1 = value_updated.read()
        assert result1.data == ("data1", True)
        assert result1.ts == 100

        result2 = value_updated.read()
        assert result2.data == ("data1", False)
        assert result2.ts == 100

        result3 = value_updated.read()
        assert result3.data == ("data1", False)
        assert result3.ts == 100

        # New timestamp
        mock_reader.read.return_value = Message(data="data2", ts=200)
        result4 = value_updated.read()
        assert result4.data == ("data2", True)
        assert result4.ts == 200

    def test_message_timestamp_is_preserved(self):
        """Test that the original message timestamp is preserved in the returned message."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        original_ts = 123456789

        value_updated = ValueUpdated(mock_reader)
        mock_reader.read.return_value = Message(data=test_data, ts=original_ts)

        result = value_updated.read()
        assert result.ts == original_ts


class TestDefaultReader:
    """Test the DefaultReader class."""

    def test_returns_reader_message_when_available(self):
        """Test that original reader's message is returned when available."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "actual_data"
        test_ts = 123
        mock_reader.read.return_value = Message(data=test_data, ts=test_ts)

        default_reader = DefaultReader(mock_reader, "default_data", 456)
        result = default_reader.read()

        assert result is not None
        assert result.data == test_data
        assert result.ts == test_ts

    def test_returns_default_when_reader_returns_none(self):
        """Test that default message is returned when reader returns None."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = None

        default_data = "default_value"
        default_ts = 789
        default_reader = DefaultReader(mock_reader, default_data, default_ts)
        result = default_reader.read()

        assert result is not None
        assert result.data == default_data
        assert result.ts == default_ts