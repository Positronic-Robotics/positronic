from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, final, TypeVar


T = TypeVar('T', covariant=True)
U = TypeVar('U')


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


class SignalReceiver(ABC, Generic[T]):
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


class NoOpEmitter(SignalEmitter[T]):

    def emit(self, data: T, ts: int = -1) -> bool:
        return True


class NoOpReceiver(SignalReceiver[T]):

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


# In pimm a control loop is a main abstraction. This is a code that manages a particular piece of robotic system.
# It can be camera, sensor, gripper, robotic arm, inference loop, etc. A robotic system then is a collection of
# control loops that communicate with each other.
ControlLoop = Callable[[SignalReceiver, Clock], Iterator[Sleep]]


class ControlSystem(ABC):
    """Composable unit of runtime that cooperates with the world scheduler.

    A control system owns the emitters and receivers that make up its external
    interface. The world supplies two utilities when running a system:

    - ``should_stop``: a ``SignalReceiver`` that becomes true when the system
      should shut down.
    - ``clock``: the ``Clock`` instance the world uses for timestamping messages.

    Implementations must advance their internal work by yielding ``Sleep``
    instances, allowing the ``World`` interleaver to sequence multiple systems.
    """

    @abstractmethod
    def run(self, should_stop: SignalReceiver, clock: Clock) -> Iterator[Sleep]:
        pass


class ControlSystemEmitter(SignalEmitter[T]):
    """Emitter adaptor that keeps track of its owning control system."""

    def __init__(self, owner: ControlSystem):
        self._owner = owner
        self._internal: list[SignalEmitter[T]] = []

    @property
    def owner(self) -> ControlSystem:
        return self._owner

    def _bind(self, emitter: SignalEmitter[T]):
        self._internal.append(emitter)

    def emit(self, data: T, ts: int = -1) -> bool:
        for emitter in self._internal:
            emitter.emit(data, ts)
        # TODO: Remove bool as return type from all Emitters
        return True


class ControlSystemReceiver(SignalReceiver[T]):
    """Receiver adaptor bound to a single upstream signal on behalf of a system."""

    def __init__(self, owner: ControlSystem):
        self._owner = owner
        self._internal: SignalReceiver[T] | None = None

    @property
    def owner(self) -> ControlSystem:
        return self._owner

    def _bind(self, receiver: SignalReceiver[T]):
        assert self._internal is None, "Receiver can be connected only to one Emitter"
        self._internal = receiver

    def read(self) -> Message[T] | None:
        return self._internal.read() if self._internal is not None else None


class ReceiverDict(dict[str, ControlSystemReceiver[U]]):
    """Dictionary that lazily allocates receivers owned by a control system."""

    def __init__(self, owner: ControlSystem):
        super().__init__()
        self._owner = owner

    def __missing__(self, key: str) -> ControlSystemReceiver[U]:
        receiver = ControlSystemReceiver(self._owner)
        self[key] = receiver
        return receiver


class EmitterDict(dict[str, ControlSystemEmitter[U]]):
    """Dictionary that lazily allocates emitters owned by a control system."""

    def __init__(self, owner: ControlSystem):
        super().__init__()
        self._owner = owner

    def __missing__(self, key: str) -> ControlSystemEmitter[U]:
        emitter = ControlSystemEmitter(self._owner)
        self[key] = emitter
        return emitter
