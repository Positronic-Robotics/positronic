import time
from typing import Callable, Any

from ironic2 import SignalReader, SignalEmitter, Message
from ironic2.core import Clock


class MapSignalReader(SignalReader):

    def __init__(self, reader: SignalReader, func: Callable[[Any], Any]):
        self.reader = reader
        self.func = func

    def read(self):
        orig_message = self.reader.read()
        if orig_message is None:
            return None
        return Message(self.func(orig_message.data), orig_message.ts)


class MapSignalEmitter(SignalEmitter):

    def __init__(self, emitter: SignalEmitter, func: Callable[[Any], Any]):
        self.emitter = emitter
        self.func = func

    def emit(self, data: Any, ts: int | None = None) -> bool:
        return self.emitter.emit(self.func(data), ts)


def map(reader: SignalReader | SignalEmitter, func: Callable[[Any], Any]) -> SignalReader | SignalEmitter:
    if isinstance(reader, SignalReader):
        return MapSignalReader(reader, func)
    elif isinstance(reader, SignalEmitter):
        return MapSignalEmitter(reader, func)
    else:
        raise ValueError(f"Invalid reader type: {type(reader)}")


class ValueUpdated(SignalReader):
    """Wrapper around reader to signal whether the value we read is 'new'."""

    def __init__(self, reader: SignalReader):
        """By default, if original reader returns None, we return None."""
        self.reader = reader
        self.last_ts = None

    def read(self) -> Message | None:
        orig_message = self.reader.read()

        if orig_message is None:
            return None

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return Message((orig_message.data, is_updated), self.last_ts)


class DefaultReader(SignalReader):
    """Signal reader that returns a default value if no value is available."""

    def __init__(self, reader: SignalReader, default: Any, default_ts: int = 0):
        self.reader = reader
        self.default_msg = Message(default, default_ts)

    def read(self) -> Message | None:
        msg = self.reader.read()
        if msg is None:
            return self.default_msg
        return msg


class RateLimiter:
    """Rate limiter that enforces a minimum interval between calls."""

    def __init__(self, clock: Clock, *, every_sec=None, hz=None) -> None:
        """
        One of every_sec or hz must be provided.
        """
        assert (every_sec is None) ^ (hz is None), "Exactly one of every_sec or hz must be provided"
        self._clock = clock
        self._last_time = None
        self._interval = every_sec if every_sec is not None else 1.0 / hz

    def wait_time(self) -> float:
        """Wait if necessary to enforce the rate limit."""
        now = self._clock.now()
        if self._last_time is not None and now - self._last_time < self._interval:
            return self._interval - (now - self._last_time)
        self._last_time = now
        return 0.0


class FPSCounter:
    """Utility class for tracking and reporting frames per second (FPS).

    Counts frames and periodically reports the average FPS over the reporting interval.

    Args:
        prefix (str): Prefix string to use in FPS report messages
        report_every_sec (float): How often to report FPS, in seconds (default: 10.0)
    """

    def __init__(self, prefix: str, report_every_sec: float = 10.0):
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.reset()

    def reset(self):
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def report(self):
        fps = self.frame_count / (time.monotonic() - self.last_report_time)
        print(f"{self.prefix}: {fps:.2f} fps")
        self.last_report_time = time.monotonic()
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()
