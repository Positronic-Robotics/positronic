from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any, Callable, ContextManager, final

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
    ts: int = -1  # -1 means no value

    def __post_init__(self):
        if self.ts == -1:
            self.ts = system_clock()


class SignalEmitter(ABC):
    """Write a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def emit(self, data: Any, ts: int = -1) -> bool:
        """Emit a message with the given data and timestamp. Returns True if successful."""
        pass

    def zc_lock(self) -> ContextManager[None]:
        """Some emitter/reader pairs can implement zero-copy operations.
        Zero-copy means that writing and reading code work with the physically same memory.
        You want to avoid reading simultaneously with writing, as the data will appear to be corrupted.

        This method returns a context manager that writing code should enter before modifying the data.
        If reader code respects the similar lock, you won't have data races.

        If emitter/reader pair does not support zero-copy, this is a harmless no-op.
        """
        return nullcontext()


class NoOpEmitter(SignalEmitter):

    def emit(self, data: Any, ts: int = -1) -> bool:
        return True


class SignalReader(ABC):
    """Read a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def read(self) -> Message | None:
        """Returns next message, otherwise last value. None if nothing was read yet."""
        pass

    @final
    @property
    def value(self) -> Any:
        """Returns the current value of the signal."""
        msg = self.read()
        if msg is None:
            raise NoValueException()
        return msg.data


class NoOpReader(SignalReader):

    def read(self) -> Message | None:
        return None


def is_true(signal: SignalReader) -> bool:
    value = signal.read()
    if value is None:
        return False
    return value.data is True


ControlSystem = Callable[[SignalReader], None]
