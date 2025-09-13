from unittest.mock import Mock

from pimm.core import Message, SignalReceiver, Clock
from pimm.utils import ValueUpdated, DefaultReceiver, RateLimiter, is_any_updated


class TestValueUpdated:
    """Test the ValueUpdated class."""

    def test_first_read_with_data_returns_updated(self):
        """Test that first read with data returns updated=True."""
        mock_reader = Mock(spec=SignalReceiver)
        test_data = "test_data"
        mock_reader.read.return_value = Message(data=test_data, ts=123)

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.read()

        assert result is not None
        assert result.data == (test_data, True)
        assert result.ts == 123

    def test_first_read_with_none_returns_none(self):
        """Test that first read with None returns None."""
        mock_reader = Mock(spec=SignalReceiver)
        mock_reader.read.return_value = None

        value_updated = ValueUpdated(mock_reader)
        result = value_updated.read()

        assert result is None

    def test_same_timestamp_returns_not_updated(self):
        """Test that reading the same timestamp returns updated=False."""
        mock_reader = Mock(spec=SignalReceiver)
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
        mock_reader = Mock(spec=SignalReceiver)
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
        mock_reader = Mock(spec=SignalReceiver)

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
        mock_reader = Mock(spec=SignalReceiver)
        test_data = "test_data"
        original_ts = 123456789

        value_updated = ValueUpdated(mock_reader)
        mock_reader.read.return_value = Message(data=test_data, ts=original_ts)

        result = value_updated.read()
        assert result.ts == original_ts


class TestDefaultReceiver:
    """Test the DefaultReceiver class."""

    def test_returns_reader_message_when_available(self):
        """Test that original reader's message is returned when available."""
        mock_reader = Mock(spec=SignalReceiver)
        test_data = "actual_data"
        test_ts = 123
        mock_reader.read.return_value = Message(data=test_data, ts=test_ts)

        default_reader = DefaultReceiver(mock_reader, "default_data", 456)
        result = default_reader.read()

        assert result is not None
        assert result.data == test_data
        assert result.ts == test_ts

    def test_returns_default_when_reader_returns_none(self):
        """Test that default message is returned when reader returns None."""
        mock_reader = Mock(spec=SignalReceiver)
        mock_reader.read.return_value = None

        default_data = "default_value"
        default_ts = 789
        default_reader = DefaultReceiver(mock_reader, default_data, default_ts)
        result = default_reader.read()

        assert result is not None
        assert result.data == default_data
        assert result.ts == default_ts


class TestRateLimiter:
    """Test the RateLimiter class."""

    def test_initialisation(self):
        """Test that providing both every_sec and hz raises AssertionError."""
        mock_clock = Mock(spec=Clock)

        RateLimiter(mock_clock, every_sec=0.5)  # Should not raise
        RateLimiter(mock_clock, hz=10)  # Should not raise

        try:
            RateLimiter(mock_clock, every_sec=0.5, hz=10)
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert "Exactly one of every_sec or hz must be provided" in str(e)

        try:
            RateLimiter(mock_clock)
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert "Exactly one of every_sec or hz must be provided" in str(e)

    def test_first_call_returns_zero_wait(self):
        """Test that first call to wait_time returns 0."""
        mock_clock = Mock(spec=Clock)
        mock_clock.now.return_value = 100.0

        rate_limiter = RateLimiter(mock_clock, every_sec=0.5)
        wait_time = rate_limiter.wait_time()

        assert wait_time == 0.0

    def test_call_within_interval_returns_correct_wait_time(self):
        """Test that calls within interval return correct wait time."""
        mock_clock = Mock(spec=Clock)
        rate_limiter = RateLimiter(mock_clock, every_sec=0.5)

        # First call at time 100
        mock_clock.now.return_value = 100.0
        wait_time1 = rate_limiter.wait_time()
        assert wait_time1 == 0.0

        # Second call at time 100.2 (within 0.5 second interval)
        mock_clock.now.return_value = 100.2
        wait_time2 = rate_limiter.wait_time()
        assert abs(wait_time2 - 0.3) < 1e-10  # 0.5 - (100.2 - 100.0)

    def test_call_after_interval_returns_zero_wait(self):
        """Test that calls after interval return 0 wait time."""
        mock_clock = Mock(spec=Clock)
        rate_limiter = RateLimiter(mock_clock, every_sec=0.5)

        # First call at time 100
        mock_clock.now.return_value = 100.0
        wait_time1 = rate_limiter.wait_time()
        assert wait_time1 == 0.0

        # Second call at time 100.6 (after 0.5 second interval)
        mock_clock.now.return_value = 100.6
        wait_time2 = rate_limiter.wait_time()
        assert wait_time2 == 0.0

    def test_multiple_calls_timing_behavior(self):
        """Test multiple calls and their timing behavior."""
        mock_clock = Mock(spec=Clock)
        rate_limiter = RateLimiter(mock_clock, every_sec=1.0)

        # First call
        mock_clock.now.return_value = 100.0
        assert rate_limiter.wait_time() == 0.0

        # Second call - within interval
        mock_clock.now.return_value = 100.3
        assert abs(rate_limiter.wait_time() - 0.7) < 1e-10

        # Third call - still within interval from first call
        mock_clock.now.return_value = 100.5
        assert abs(rate_limiter.wait_time() - 0.5) < 1e-10

        # Fourth call - after interval
        mock_clock.now.return_value = 101.2
        assert rate_limiter.wait_time() == 0.0

    def test_hz_parameter_different_frequencies(self):
        """Test that hz parameter works with different frequencies."""
        mock_clock = Mock(spec=Clock)

        # Test high frequency rate limiting
        rate_limiter_high = RateLimiter(mock_clock, hz=100)
        mock_clock.now.return_value = 0.0
        assert rate_limiter_high.wait_time() == 0.0
        mock_clock.now.return_value = 0.005  # Half the interval
        assert rate_limiter_high.wait_time() > 0.0

        # Test low frequency rate limiting
        rate_limiter_low = RateLimiter(mock_clock, hz=2)
        mock_clock.now.return_value = 10.0
        assert rate_limiter_low.wait_time() == 0.0
        mock_clock.now.return_value = 10.25  # Half the interval
        assert rate_limiter_low.wait_time() > 0.0

    def test_precise_timing_with_hz(self):
        """Test precise timing behavior with hz parameter."""
        mock_clock = Mock(spec=Clock)
        rate_limiter = RateLimiter(mock_clock, hz=100)  # 0.01 second interval

        # First call
        mock_clock.now.return_value = 50.0
        assert rate_limiter.wait_time() == 0.0

        # Second call - just before interval
        mock_clock.now.return_value = 50.009
        wait_time = rate_limiter.wait_time()
        assert abs(wait_time - 0.001) < 1e-10  # Very close to 0.001

        # Third call - just after interval
        mock_clock.now.return_value = 50.011
        assert rate_limiter.wait_time() == 0.0


class TestIsAnyUpdated:
    def test_returns_false_if_no_values_updated(self):
        reader1 = Mock(spec=SignalReceiver)
        reader1.read.return_value = Message(data=1, ts=1)
        reader2 = Mock(spec=SignalReceiver)
        reader2.read.return_value = Message(data=2, ts=2)

        vu_reader1 = ValueUpdated(reader1)
        vu_reader2 = ValueUpdated(reader2)

        # read messages once
        vu_reader1.read()
        vu_reader2.read()

        messages, is_updated = is_any_updated({'reader1': vu_reader1, 'reader2': vu_reader2})

        assert messages == {'reader1': Message(data=1, ts=1), 'reader2': Message(data=2, ts=2)}
        assert not is_updated

    def test_returns_true_if_all_values_updated(self):
        reader1 = Mock(spec=SignalReceiver)
        reader1.read.return_value = Message(data=1, ts=1)
        reader2 = Mock(spec=SignalReceiver)
        reader2.read.return_value = Message(data=2, ts=2)

        vu_reader1 = ValueUpdated(reader1)
        vu_reader2 = ValueUpdated(reader2)

        messages, is_updated = is_any_updated({'reader1': vu_reader1, 'reader2': vu_reader2})

        assert messages == {'reader1': Message(data=1, ts=1), 'reader2': Message(data=2, ts=2)}
        assert is_updated

    def test_returns_true_if_any_value_updated(self):
        reader1 = Mock(spec=SignalReceiver)
        reader1.read.return_value = Message(data=1, ts=1)
        reader2 = Mock(spec=SignalReceiver)
        reader2.read.return_value = Message(data=2, ts=2)

        vu_reader1 = ValueUpdated(reader1)
        vu_reader2 = ValueUpdated(reader2)

        # only one value is updated
        vu_reader1.read()

        messages, is_updated = is_any_updated({'reader1': vu_reader1, 'reader2': vu_reader2})

        assert messages == {'reader1': Message(data=1, ts=1), 'reader2': Message(data=2, ts=2)}
        assert is_updated

    def test_returns_false_and_empty_dict_if_all_readers_has_no_data(self):
        reader1 = Mock(spec=SignalReceiver)
        reader1.read.return_value = None
        reader2 = Mock(spec=SignalReceiver)
        reader2.read.return_value = None

        vu_reader1 = ValueUpdated(reader1)
        vu_reader2 = ValueUpdated(reader2)

        messages, is_updated = is_any_updated({'reader1': vu_reader1, 'reader2': vu_reader2})

        assert messages == {}
        assert not is_updated

    def test_returns_true_and_present_data_if_some_readers_has_no_data(self):
        reader1 = Mock(spec=SignalReceiver)
        reader1.read.return_value = None
        reader2 = Mock(spec=SignalReceiver)
        reader2.read.return_value = Message(data=2, ts=2)

        vu_reader1 = ValueUpdated(reader1)
        vu_reader2 = ValueUpdated(reader2)

        messages, is_updated = is_any_updated({'reader1': vu_reader1, 'reader2': vu_reader2})

        assert messages == {'reader2': Message(data=2, ts=2)}
        assert is_updated
