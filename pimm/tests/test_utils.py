from unittest.mock import Mock

from pimm.core import Clock, Message, SignalEmitter, SignalReceiver
from pimm.utils import DefaultReceiver, MapSignalEmitter, MapSignalReceiver, RateLimiter


class TestMapSignalReceiver:
    """Test the MapSignalReceiver class."""

    def test_transforms_data_with_mapping_function(self):
        """Test that data is transformed by the mapping function."""
        mock_reader = Mock(spec=SignalReceiver)
        test_data = 10
        test_ts = 123
        mock_reader.read.return_value = Message(data=test_data, ts=test_ts)

        # Double the value
        map_reader = MapSignalReceiver(mock_reader, lambda x: x * 2)
        result = map_reader.read()

        assert result is not None
        assert result.data == 20
        assert result.ts == test_ts
        assert result.updated is True

    def test_preserves_timestamp(self):
        """Test that original timestamp is preserved."""
        mock_reader = Mock(spec=SignalReceiver)
        test_ts = 987654321
        mock_reader.read.return_value = Message(data='test', ts=test_ts)

        map_reader = MapSignalReceiver(mock_reader, lambda x: x.upper())
        result = map_reader.read()

        assert result is not None
        assert result.ts == test_ts
        assert result.updated is True

    def test_returns_none_when_reader_returns_none(self):
        """Test that None is returned when underlying reader returns None."""
        mock_reader = Mock(spec=SignalReceiver)
        mock_reader.read.return_value = None

        map_reader = MapSignalReceiver(mock_reader, lambda x: x * 2)
        result = map_reader.read()

        assert result is None

    def test_filters_when_func_returns_none(self):
        """Test that returning None from func filters out the value."""
        mock_reader = Mock(spec=SignalReceiver)
        test_ts = 456
        mock_reader.read.return_value = Message(data=10, ts=test_ts)

        # Filter function that returns None
        map_reader = MapSignalReceiver(mock_reader, lambda x: None)
        result = map_reader.read()

        assert result is None

    def test_filters_preserve_last_non_none_message(self):
        """Test that when func returns None, we return the last non-None message."""
        mock_reader = Mock(spec=SignalReceiver)

        # First read: return a valid value
        mock_reader.read.return_value = Message(data=10, ts=100)
        map_reader = MapSignalReceiver(mock_reader, lambda x: x * 2 if x < 20 else None)

        result1 = map_reader.read()
        assert result1 is not None
        assert result1.data == 20
        assert result1.ts == 100
        assert result1.updated is True

        # Second read: func returns None, should return last message
        mock_reader.read.return_value = Message(data=30, ts=200)
        result2 = map_reader.read()
        assert result2 is not None
        assert result2.data == 20  # Last non-None value
        assert result2.ts == 100  # Last non-None timestamp
        assert result2.updated is False

    def test_filtering_with_conditional_logic(self):
        """Test filtering with conditional logic."""
        mock_reader = Mock(spec=SignalReceiver)

        # Filter only even numbers
        def filter_even(x):
            return x if x % 2 == 0 else None

        map_reader = MapSignalReceiver(mock_reader, filter_even)

        # Test with even number
        mock_reader.read.return_value = Message(data=10, ts=100)
        result1 = map_reader.read()
        assert result1 is not None
        assert result1.data == 10

        # Test with odd number (should return last message)
        mock_reader.read.return_value = Message(data=11, ts=200)
        result2 = map_reader.read()
        assert result2 is not None
        assert result2.data == 10  # Last even number
        assert result2.ts == 100
        assert result2.updated is False

        # Test with another even number
        mock_reader.read.return_value = Message(data=12, ts=300)
        result3 = map_reader.read()
        assert result3 is not None
        assert result3.data == 12
        assert result3.ts == 300
        assert result3.updated is True

    def test_initial_state_has_no_last_message(self):
        """Test that initial state returns None when func returns None."""
        mock_reader = Mock(spec=SignalReceiver)
        mock_reader.read.return_value = Message(data=10, ts=100)

        # First read with func returning None should return None
        map_reader = MapSignalReceiver(mock_reader, lambda x: None)
        result = map_reader.read()

        assert result is None

    def test_propagates_updated_flag(self):
        """Test that updated flag from upstream reader is preserved."""
        mock_reader = Mock(spec=SignalReceiver)
        msg = Message(data=42, ts=100, updated=False)
        mock_reader.read.return_value = msg

        map_reader = MapSignalReceiver(mock_reader, lambda x: x)
        result = map_reader.read()

        assert result is not None
        assert result.updated is False


class TestMapSignalEmitter:
    """Test the MapSignalEmitter class."""

    def test_transforms_data_before_emitting(self):
        """Test that data is transformed before being emitted."""
        mock_emitter = Mock(spec=SignalEmitter)
        map_emitter = MapSignalEmitter(mock_emitter, lambda x: x * 2)
        map_emitter.emit(10, ts=123)
        mock_emitter.emit.assert_called_once_with(20, 123)

    def test_preserves_timestamp(self):
        """Test that timestamp is preserved when emitting."""
        mock_emitter = Mock(spec=SignalEmitter)
        map_emitter = MapSignalEmitter(mock_emitter, lambda x: x.upper())
        test_ts = 987654321
        map_emitter.emit('test', ts=test_ts)
        mock_emitter.emit.assert_called_once_with('TEST', test_ts)

    def test_filters_when_func_returns_none(self):
        """Test that returning None from func prevents emission."""
        mock_emitter = Mock(spec=SignalEmitter)
        map_emitter = MapSignalEmitter(mock_emitter, lambda _: None)
        map_emitter.emit(10, ts=123)
        mock_emitter.emit.assert_not_called()

    def test_conditional_filtering(self):
        """Test filtering with conditional logic."""
        mock_emitter = Mock(spec=SignalEmitter)

        def filter_even(x):
            return x if x % 2 == 0 else None

        map_emitter = MapSignalEmitter(mock_emitter, filter_even)

        map_emitter.emit(10, ts=100)
        mock_emitter.emit.assert_called_once_with(10, 100)

        mock_emitter.reset_mock()
        map_emitter.emit(11, ts=200)
        mock_emitter.emit.assert_not_called()

        map_emitter.emit(12, ts=300)
        mock_emitter.emit.assert_called_once_with(12, 300)

    def test_transform_and_filter_combination(self):
        """Test combining transformation and filtering."""
        mock_emitter = Mock(spec=SignalEmitter)

        def transform_and_filter(x):
            squared = x * x
            return squared if squared < 100 else None

        map_emitter = MapSignalEmitter(mock_emitter, transform_and_filter)

        map_emitter.emit(5, ts=100)
        mock_emitter.emit.assert_called_once_with(25, 100)

        mock_emitter.reset_mock()
        map_emitter.emit(15, ts=200)
        mock_emitter.emit.assert_not_called()


class TestDefaultReceiver:
    """Test the DefaultReceiver class."""

    def test_returns_reader_message_when_available(self):
        """Test that original reader's message is returned when available."""
        mock_reader = Mock(spec=SignalReceiver)
        test_data = 'actual_data'
        test_ts = 123
        mock_reader.read.return_value = Message(data=test_data, ts=test_ts)

        default_reader = DefaultReceiver(mock_reader, 'default_data', 456)
        result = default_reader.read()

        assert result is not None
        assert result.data == test_data
        assert result.ts == test_ts
        assert result.updated is True

    def test_returns_default_when_reader_returns_none(self):
        """Test that default message is returned when reader returns None."""
        mock_reader = Mock(spec=SignalReceiver)
        mock_reader.read.return_value = None

        default_data = 'default_value'
        default_ts = 789
        default_reader = DefaultReceiver(mock_reader, default_data, default_ts)
        result = default_reader.read()

        assert result is not None
        assert result.data == default_data
        assert result.ts == default_ts
        assert result.updated is False


class TestRateLimiter:
    """Test the RateLimiter class."""

    def test_initialisation(self):
        """Test that providing both every_sec and hz raises AssertionError."""
        mock_clock = Mock(spec=Clock)

        RateLimiter(mock_clock, every_sec=0.5)  # Should not raise
        RateLimiter(mock_clock, hz=10)  # Should not raise

        try:
            RateLimiter(mock_clock, every_sec=0.5, hz=10)
            raise AssertionError('Expected AssertionError')
        except AssertionError as e:
            assert 'Exactly one of every_sec or hz must be provided' in str(e)

        try:
            RateLimiter(mock_clock)
            raise AssertionError('Expected AssertionError')
        except AssertionError as e:
            assert 'Exactly one of every_sec or hz must be provided' in str(e)

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
