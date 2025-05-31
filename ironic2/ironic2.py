from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable


def system_clock() -> int:
    """Get current timestamp in nanoseconds."""
    return time.monotonic_ns()


# Object that represents no value
# It is used to make a distinction between a value that is not set and a value that is set to None
class NoValueType:

    def __str__(self):
        return "ironic.NoValue"

    def __repr__(self):
        return str(self)


NoValue = NoValueType()


@dataclass
class Message:
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: Any
    ts: int = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = system_clock()


class SignalEmitter(ABC):
    @abstractmethod
    def emit(self, message: Message) -> bool:
        pass


class NoOpEmitter(SignalEmitter):
    def emit(self, message: Message) -> bool:
        return True


class SignalReader(ABC):
    @abstractmethod
    def value(self) -> Message | NoValueType:
        """Returns next message, otherwise last value. NoValue if nothing was read yet."""
        pass


class NoOpReader(SignalReader):
    def value(self) -> Message | NoValueType:
        return NoValue


def signal_value(signal: SignalReader, default: Any) -> Any:
    value = signal.value()
    if value is NoValue:
        return default
    return value.data


ControlSystem = Callable[[SignalReader], None]
