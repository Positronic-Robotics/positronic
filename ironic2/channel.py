from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Any, Union


def system_clock() -> int:
    """Get current timestamp in nanoseconds."""
    return time.monotonic_ns()


# Object that represents no value
# It is used to make a distinction between a value that is not set and a value that is set to None
class NoValue:

    def __str__(self):
        return "ironic.NoValue"

    def __repr__(self):
        return str(self)


NoValue = NoValue()


@dataclass
class Message:
    """
    Contains some data and a timestamp for this data. Timestamps are integers,
    to avoid floating point precision issues. It can be related to epoch or
    to anything else, depending on the context.

    If no timestamp is provided, the current system time is used.
    """
    data: Any
    timestamp: int = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = system_clock()


class Channel(ABC):
    """The way to commmunicate between different parts of robotic system

    Implementations can rely on different mechanics, inlcuding the actual channels, queues, shared memory, RPCs, etc.
    Implementations can drop values, transform them, do anything they want. It's up to the one who construct the whole
    system to decide which particular implementations to use.

    Please note that write and read may be read from different processes.
    """

    @abstractmethod
    def write(self, message: Message):
        """Write new value into channel. Must be non-blocking."""
        pass

    @abstractmethod
    def read(self) -> Union[Message, NoValue]:
        """Must return NoValue when there is nothing to read."""
        pass


# TODO: define EventLike type


class LastValueChannel(Channel):
    """Wrapper around Channel that keeps last value."""

    def __init__(self, base_channel):
        super().__init__()
        self.base = base_channel
        self.last_value = NoValue

    def write(self, message):
        return self.base.write(message)

    def read(self):
        value = self.base.read()
        while value is not NoValue:
            self.last_value = value
        return self.last_value


class DuplicateChannel(Channel):
    """The channel that forwards data into multiple channels."""

    def __init__(self, *channels):
        super().__init__()
        self.channels = channels

    def write(self, message):
        for c in self.channels:
            c.write()

    def read(self):
        raise ValueError('Duplicate Channel is write only')
