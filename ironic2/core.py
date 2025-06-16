from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable

# Internal sentinel object to distinguish between no default and explicit None default
_RAISE_EXCEPTION_SENTINEL = object()


def system_clock() -> int:
    """Get current timestamp in nanoseconds."""
    return time.monotonic_ns()


class NoValueException(Exception):
    pass


@dataclass
class Message:
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: Any
    ts: int | None = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = system_clock()


class SignalEmitter(ABC):
    """Write a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def emit(self, data: Any, ts: int | None = None) -> bool:
        """Emit a message with the given data and timestamp. Returns True if successful."""
        pass


class NoOpEmitter(SignalEmitter):

    def emit(self, data: Any, ts: int | None = None) -> bool:
        return True


class SignalReader(ABC):
    """Read a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def value(self) -> Message | None:
        """Returns next message, otherwise last value. None if nothing was read yet."""
        pass


class NoOpReader(SignalReader):

    def value(self) -> Message | None:
        return None


def is_true(signal: SignalReader) -> bool:
    value = signal.value()
    if value is None:
        return False
    return value.data is True


def signal_value(signal: SignalReader, default: Any = _RAISE_EXCEPTION_SENTINEL) -> Any:
    value = signal.value()
    if value is None:
        if default is _RAISE_EXCEPTION_SENTINEL:
            raise NoValueException
        return default

    return value.data


ControlSystem = Callable[[SignalReader], None]
