from unittest.mock import Mock

import pytest

from ironic2.core import (Message, NoValue, NoValueException, SignalReader,
                          is_true, signal_value)


class TestIsTrue:
    """Test the is_true utility function."""

    def test_is_true_with_true_signal(self):
        """Test is_true with a signal containing True."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = Message(data=True, ts=123)

        result = is_true(mock_reader)
        assert result is True

    def test_is_true_with_false_signal(self):
        """Test is_true with a signal containing False."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = Message(data=False, ts=123)

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_no_value_signal(self):
        """Test is_true with a signal containing NoValue."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = NoValue

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_non_boolean_signal(self):
        """Test is_true with a signal containing non-boolean data."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = Message(data="some_string", ts=123)

        result = is_true(mock_reader)
        assert result is False

    def test_is_true_with_truthy_signal(self):
        """Test is_true with a signal containing truthy but not True data."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = Message(data=1, ts=123)

        result = is_true(mock_reader)
        assert result is False  # Only True should return True


class TestSignalValue:
    """Test the signal_value utility function."""

    def test_signal_value_with_data(self):
        """Test signal_value with a signal containing data."""
        mock_reader = Mock(spec=SignalReader)
        test_data = "test_data"
        mock_reader.value.return_value = Message(data=test_data, ts=123)

        result = signal_value(mock_reader)
        assert result == test_data

    def test_signal_value_with_no_value_and_no_default(self):
        """Test signal_value with NoValue and no default raises exception."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = NoValue

        with pytest.raises(NoValueException):
            signal_value(mock_reader)

    def test_signal_value_with_no_value_and_default(self):
        """Test signal_value with NoValue and default returns default."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = NoValue
        default_value = "default"

        result = signal_value(mock_reader, default=default_value)
        assert result == default_value

    def test_signal_value_with_none_data(self):
        """Test signal_value with None as data."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = Message(data=None, ts=123)

        result = signal_value(mock_reader)
        assert result is None

    def test_signal_value_with_complex_data(self):
        """Test signal_value with complex data structures."""
        mock_reader = Mock(spec=SignalReader)
        complex_data = {"key": "value", "list": [1, 2, 3]}
        mock_reader.value.return_value = Message(data=complex_data, ts=123)

        result = signal_value(mock_reader)
        assert result == complex_data

    def test_signal_value_with_none_as_default(self):
        """Test signal_value with None as explicit default."""
        mock_reader = Mock(spec=SignalReader)
        mock_reader.value.return_value = NoValue

        result = signal_value(mock_reader, default=None)
        assert result is None
