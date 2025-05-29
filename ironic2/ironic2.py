from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any, Dict


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


class SignalReader(ABC):
    @abstractmethod
    def value(self) -> Message | NoValueType:
        """Returns next message, otherwise last value. NoValue if nothing was read yet."""
        pass


def signal_is_true(signal: SignalReader) -> bool:
    value = signal.value()
    return value is not NoValue and value.data is True


class CommunicationProvider(ABC):
    @abstractmethod
    def emitter(self, name: str, **kwargs: Dict[str, Any]) -> SignalEmitter:
        pass

    @abstractmethod
    def reader(self, name: str, **kwargs: Dict[str, Any]) -> SignalReader:
        pass

    @abstractmethod
    def should_stop(self) -> SignalReader:
        pass

    @abstractmethod
    def interface(self) -> SimpleNamespace:
        pass


class ControlSystem(ABC):
    def __init__(self, comms: CommunicationProvider):
        pass

    @abstractmethod
    def run(self):
        pass
