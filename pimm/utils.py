import time
from typing import Callable, Mapping, Tuple, overload, TypeVar

from pimm import SignalReceiver, SignalEmitter, Message
from pimm.core import Clock


T = TypeVar('T', covariant=True)
K = TypeVar('K', covariant=True)


class MapSignalReceiver(SignalReceiver[T]):
    def __init__(self, reader: SignalReceiver[T], func: Callable[[T], T]):
        self.reader = reader
        self.func = func

    def read(self):
        orig_message = self.reader.read()
        if orig_message is None:
            return None
        return Message(self.func(orig_message.data), orig_message.ts)


class MapSignalEmitter(SignalEmitter[T]):

    def __init__(self, emitter: SignalEmitter[T], func: Callable[[T], T]):
        self.emitter = emitter
        self.func = func

    def emit(self, data: T, ts: int = -1) -> bool:
        return self.emitter.emit(self.func(data), ts)


@overload
def map(signal: SignalReceiver[T], func: Callable[[T], T]) -> SignalReceiver[T]:
    ...


@overload
def map(signal: SignalEmitter[T], func: Callable[[T], T]) -> SignalEmitter[T]:
    ...


def map(signal: SignalReceiver[T] | SignalEmitter[T], func: Callable[[T], T]) -> SignalReceiver[T] | SignalEmitter[T]:
    if isinstance(signal, SignalReceiver):
        return MapSignalReceiver(signal, func)
    elif isinstance(signal, SignalEmitter):
        return MapSignalEmitter(signal, func)
    else:
        raise ValueError(f"Invalid signal type: {type(signal)}")


class ValueUpdated(SignalReceiver[Tuple[T, bool]]):
    """Wrapper around reader to signal whether the value we read is 'new'."""

    def __init__(self, reader: SignalReceiver[T]):
        """By default, if original reader returns None, we return None."""
        self.reader = reader
        self.last_ts = None

    def read(self) -> Message[Tuple[T, bool]] | None:
        orig_message = self.reader.read()

        if orig_message is None:
            return None

        is_updated = orig_message.ts != self.last_ts
        self.last_ts = orig_message.ts

        return Message((orig_message.data, is_updated), self.last_ts)


def is_any_updated(readers: Mapping[str, SignalReceiver[Tuple[T, bool]]]) -> Tuple[dict[str, Message[T]], bool]:
    """Get the latest value of all readers and whether any of them are updated.

    In case some of the readers return None, this keys will be omitted from the returned dict.

    Args:
        readers: A mapping of reader names to readers. Typically a dict of ValueUpdated readers.

    Returns:
        (dict[str, Message[T]], bool): Dict with latest values and a bool indicating whether any of the readers
        are updated.
    """
    messages = {k: reader.read() for k, reader in readers.items()}
    is_updated = {k: msg.data[1] for k, msg in messages.items() if msg is not None}
    is_any_updated = any(is_updated.values())

    messages = {k: Message(msg.data[0], msg.ts) for k, msg in messages.items() if msg is not None}

    return messages, is_any_updated


class DefaultReceiver(SignalReceiver[T | K]):
    """Signal reader that returns a default value if no value is available."""

    def __init__(self, reader: SignalReceiver[T], default: K, default_ts: int = 0):
        self.reader = reader
        self.default_msg = Message(default, default_ts)

    def read(self) -> Message[T | K] | None:
        msg = self.reader.read()
        if msg is None:
            return self.default_msg
        return msg


class RateLimiter:
    """Rate limiter that enforces a minimum interval between calls."""

    def __init__(self, clock: Clock, *, every_sec: float | None = None, hz: float | None = None) -> None:
        """
        One of every_sec or hz must be provided.
        """
        assert (every_sec is None) ^ (hz is None), "Exactly one of every_sec or hz must be provided"
        self._clock = clock
        self._last_time = None
        self._interval = every_sec if every_sec is not None else 1.0 / hz  # type: ignore

    def wait_time(self) -> float:
        """Wait if necessary to enforce the rate limit."""
        now = self._clock.now()
        if self._last_time is not None and now - self._last_time < self._interval:
            return self._interval - (now - self._last_time)
        self._last_time = now
        return 0.0


class RateCounter:
    """Utility class for tracking and reporting call rate.

    Counts events and periodically reports the average rate over the reporting interval.

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
        self.tick_count = 0

    def report(self):
        rate = self.tick_count / (time.monotonic() - self.last_report_time)
        print(f"{self.prefix}: {rate:.2f} Hz")
        self.last_report_time = time.monotonic()
        self.tick_count = 0

    def tick(self):
        self.tick_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()
