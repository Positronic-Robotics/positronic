from unittest.mock import Mock

from ironic2.core import Message, SignalReader, is_true


class TestIsTrue:
    """Test the is_true utility function."""

    def test_is_true_with_true_signal(self):
        """Test is_true with a signal containing True."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = Message(data=True, ts=123)

        result = is_true(mock_reader)
        assert result is True

    def test_is_true_with_false_signal(self):
        """Test is_true with a signal containing False."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = Message(data=False, ts=123)

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_no_value_signal(self):
        """Test is_true with a signal containing None."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = None

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_non_boolean_signal(self):
        """Test is_true with a signal containing non-boolean data."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = Message(data="some_string", ts=123)

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_truthy_signal(self):
        """Test is_true with a signal containing truthy but not True data."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.read.return_value = Message(data=1, ts=123)

        result = is_true(mock_reader)
        assert result is False  # Only True should return True
