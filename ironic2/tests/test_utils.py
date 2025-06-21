from unittest.mock import Mock

from ironic2.core import Message, SignalReader
from ironic2.utils import ValueUpdated


class TestValueUpdated:
    """Test the ValueUpdated class."""

    def test_first_read_with_data_returns_updated(self):
        """Test that first read with data returns updated=True."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        mock_reader.value.return_value = Message(data=test_data, ts=123)

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.value()

        assert result is not None
        assert result.data[0].data == test_data
        assert result.data[1] is True  # is_updated should be True
        assert result.ts == 123

    def test_first_read_with_none_returns_none(self):
        """Test that first read with None returns None (no default value)."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = None

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.value()

        assert result is None

    def test_first_read_with_none_and_default_returns_default_not_updated(self):
        """Test that first read with None and default value returns default with updated=False."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = None
        def_value = "default"

        value_updated = ValueUpdated(mock_reader, default_value=def_value)
        result = value_updated.value()

        assert result == (def_value, False)

    def test_same_timestamp_returns_not_updated(self):
        """Test that reading the same timestamp returns updated=False."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        message = Message(data=test_data, ts=123)
        mock_reader.value.return_value = message

        value_updated = ValueUpdated(mock_reader)

        # First read
        result1 = value_updated.value()
        assert result1.data[1] is True  # First read should be updated

        # Second read with same timestamp
        result2 = value_updated.value()
        assert result2.data[1] is False  # Same timestamp should not be updated

    def test_different_timestamp_returns_updated(self):
        """Test that reading different timestamps returns updated=True."""
        mock_reader = Mock(spec=SignalReader)
        test_data1 = "test_data1"
        test_data2 = "test_data2"

        value_updated = ValueUpdated(mock_reader)

        # First read
        mock_reader.value.return_value = Message(data=test_data1, ts=123)
        result1 = value_updated.value()
        assert result1.data[1] is True
        assert result1.data[0].data == test_data1

        # Second read with different timestamp
        mock_reader.value.return_value = Message(data=test_data2, ts=456)
        result2 = value_updated.value()
        assert result2.data[1] is True  # Different timestamp should be updated
        assert result2.data[0].data == test_data2

    def test_timestamp_tracking_persists_across_calls(self):
        """Test that timestamp tracking persists across multiple calls."""
        mock_reader = Mock(spec=SignalReader)

        value_updated = ValueUpdated(mock_reader)

        # Multiple reads with same timestamp
        mock_reader.value.return_value = Message(data="data1", ts=100)
        result1 = value_updated.value()
        assert result1.data[1] is True  # First time seeing ts=100

        result2 = value_updated.value()
        assert result2.data[1] is False  # Second time seeing ts=100

        result3 = value_updated.value()
        assert result3.data[1] is False  # Third time seeing ts=100

        # New timestamp
        mock_reader.value.return_value = Message(data="data2", ts=200)
        result4 = value_updated.value()
        assert result4.data[1] is True  # First time seeing ts=200

    def test_none_as_explicit_default_value(self):
        """Test that None can be used as an explicit default value."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = None

        value_updated = ValueUpdated(mock_reader, default_value=None)
        result = value_updated.value()

        assert result == (None, False)

    def test_message_timestamp_is_preserved(self):
        """Test that the original message timestamp is preserved in the returned message."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        original_ts = 123456789

        value_updated = ValueUpdated(mock_reader)
        mock_reader.value.return_value = Message(data=test_data, ts=original_ts)

        result = value_updated.value()
        assert result.ts == original_ts
