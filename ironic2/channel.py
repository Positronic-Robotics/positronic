from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any


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


class Channel(ABC):
    """The way to commmunicate between different parts of robotic system

    Implementations can rely on different mechanics, inlcuding the actual channels, queues, shared memory, RPCs, etc.
    Implementations can drop values, transform them, do anything they want. It's up to the one who construct the whole
    system to decide which particular implementations to use.

    Please note that write and read may be called from different processes.
    """

    @abstractmethod
    def write(self, message: Message) -> bool:
        """Write new value into channel. Implementations must be non-blocking.

        Returns False if the value was not written. Lossy implementations may return True even in this case.
        """
        pass

    @abstractmethod
    def value(self) -> Message | NoValueType:
        """The current value in the channel. Returns NoValue only if nothing was written yet.

        Does not have to be the same as the last value written.
        """
        # There are two use cases, either I want to read just the last value, or I want to read "next" value.
        pass


# TODO: Should we define EventLike type?
