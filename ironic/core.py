from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, final, TypeVar


T = TypeVar('T', covariant=True)


class NoValueException(Exception):
    pass


@dataclass
class Message(Generic[T]):
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: T
    ts: int = -1  # -1 means no value


class SignalEmitter(ABC, Generic[T]):
    """Write a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def emit(self, data: T, ts: int = -1) -> bool:
        """
        Emit a message with the given data and timestamp. Returns True if successful.
        Must overwrite ts with current clock time if negative.
        """
        pass


class NoOpEmitter(SignalEmitter[T]):

    def emit(self, data: T, ts: int = -1) -> bool:
        return True


class SignalReader(ABC, Generic[T]):
    """Read a signal value. All implementations must be non-blocking."""

    @abstractmethod
    def read(self) -> Message[T] | None:
        """Returns next message, otherwise last value. None if nothing was read yet."""
        pass

    @final
    @property
    def value(self) -> T:
        """Returns the current value of the signal."""
        msg = self.read()
        if msg is None:
            raise NoValueException
        return msg.data


class NoOpReader(SignalReader[T]):

    def read(self) -> None:
        return None


class Clock(ABC):
    """A clock is a source of timestamps. It can be system clock, or a more precise clock."""

    @abstractmethod
    def now(self) -> float:
        """Get current timestamp in seconds."""
        pass

    def now_ns(self) -> int:
        """Get current timestamp in nanoseconds."""
        return int(self.now() * 1e9)


@dataclass
class Sleep:
    seconds: float


def Pass() -> Sleep:
    return Sleep(0.0)


# In Ironic a control loop is a main abstraction. This is a code that manages a particular piece of robotic system.
# It can be camera, sensor, gripper, robotic arm, inference loop, etc. A robotic system then is a collection of
# control loops that communicate with each other.
ControlLoop = Callable[[SignalReader, Clock], Iterator[Sleep]]
