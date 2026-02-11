import logging
import time
from collections.abc import Callable
from typing import TypeVar, overload

from pimm import Message, SignalEmitter, SignalReceiver
from pimm.core import Clock

T = TypeVar('T', covariant=True)
K = TypeVar('K', covariant=True)


def identity(x):
    return x


class MapSignalReceiver(SignalReceiver[T]):
    """Transform and filter signal data on read.

    If func returns None, the value is filtered out and receiver behaves
    like it didn't see the filtered message.
    """

    def __init__(self, receiver: SignalReceiver[T], func: Callable[[T], T | None]):
        self.receiver = receiver
        self.func = func

        self.last_message: Message[T] | None = None

    def read(self):
        orig_message = self.receiver.read()
        if orig_message is None:
            return None

        transformed_data = self.func(orig_message.data)
        if transformed_data is None:
            if self.last_message is not None:
                self.last_message.updated = False
            return self.last_message

        self.last_message = Message(transformed_data, orig_message.ts, orig_message.updated)
        return self.last_message

    def _bind(self, receiver: SignalReceiver[T]):
        self.receiver = receiver


class MapSignalEmitter(SignalEmitter[T]):
    """Transform and filter signal data on emit.

    If func returns None, the value is filtered out and nothing is emitted.
    This enables conditional filtering at the emission point.
    """

    def __init__(self, emitter: SignalEmitter[T], func: Callable[[T], T | None]):
        self.emitter = emitter
        self.func = func

    def emit(self, data: T, ts: int = -1):
        transformed_data = self.func(data)
        if transformed_data is not None:
            self.emitter.emit(transformed_data, ts)


@overload
def map(signal: SignalReceiver[T], func: Callable[[T], T | None]) -> SignalReceiver[T]: ...


@overload
def map(signal: SignalEmitter[T], func: Callable[[T], T | None]) -> SignalEmitter[T]: ...


def map(
    func: Callable[[T], T | None],
) -> Callable[[SignalReceiver[T] | SignalEmitter[T]], SignalReceiver[T] | SignalEmitter[T]]:
    """Transform or filter values passing through a signal.

    Returns a wrapper that applies func to all values. If func returns None,
    the value is filtered: receivers return the last valid message, emitters
    skip emission entirely.

    Args:
        func: Callable that transforms type T to T, or returns None to filter.

    Returns:
        A function that wraps a SignalReceiver or SignalEmitter with the transform.

    Raises:
        ValueError: If the provided signal is not a SignalReceiver or SignalEmitter.
    """

    def wrapper(signal: SignalReceiver[T] | SignalEmitter[T]) -> SignalReceiver[T] | SignalEmitter[T]:
        if isinstance(signal, SignalReceiver):
            return MapSignalReceiver(signal, func)
        elif isinstance(signal, SignalEmitter):
            return MapSignalEmitter(signal, func)
        else:
            raise ValueError(f'Invalid signal type: {type(signal)}')

    return wrapper


class RateLimiter:
    """Rate limiter that enforces a minimum interval between calls."""

    def __init__(self, clock: Clock, *, every_sec: float | None = None, hz: float | None = None) -> None:
        """
        One of every_sec or hz must be provided.
        """
        assert (every_sec is None) ^ (hz is None), 'Exactly one of every_sec or hz must be provided'
        self._clock = clock
        self._next_time = None
        self._interval = every_sec if every_sec is not None else 1.0 / hz  # type: ignore

    def reset(self):
        """Reset the rate limiter."""
        self._next_time = None

    def wait_time(self) -> float:
        """Return seconds to sleep before the next tick.

        Advances the internal deadline on every call so that each call
        consumes exactly one interval slot.  If the caller falls behind
        (e.g. work took longer than the interval), the deadline is
        fast-forwarded to the next future slot.
        """
        now = self._clock.now()
        if self._next_time is None:
            self._next_time = now + self._interval
            return 0.0
        wait = max(0.0, self._next_time - now)
        self._next_time += self._interval
        if self._next_time < now:
            self._next_time = now + self._interval
        return wait


class RateCounter:
    """Utility class for tracking and reporting call rate.

    Counts events and periodically reports the average rate over the reporting interval.

    Args:
        prefix (str): Prefix string to use in FPS report messages
        report_every_sec (float): How often to report FPS, in seconds (default: 10.0)
    """

    def __init__(self, prefix: str, report_every_sec: float = 10.0, level: int | None = logging.DEBUG):
        # None means print to stdout
        self.prefix = prefix
        self.report_every_sec = report_every_sec
        self.reset()
        self.level = level

    def reset(self):
        self.last_report_time = time.monotonic()
        self.tick_count = 0

    def report(self):
        rate = self.tick_count / (time.monotonic() - self.last_report_time)
        if self.level is None:
            print(f'{self.prefix}: {rate:.2f} Hz')
        else:
            logging.log(self.level, '%s: %.2f Hz', self.prefix, rate)
        self.last_report_time = time.monotonic()
        self.tick_count = 0

    def tick(self):
        self.tick_count += 1
        if time.monotonic() - self.last_report_time >= self.report_every_sec:
            self.report()
