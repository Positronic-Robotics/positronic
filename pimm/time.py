"""Named timestamps and clocks used by messages and the scheduler."""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from operator import index
from types import MappingProxyType
from typing import SupportsIndex, overload

import numpy as np

EMITTED_PREFIX = 'emitted.'
RECEIVED_PREFIX = 'received.'
EMITTED_WALL = f'{EMITTED_PREFIX}wall'
EMITTED_WORLD = f'{EMITTED_PREFIX}world'
RECEIVED_WALL = f'{RECEIVED_PREFIX}wall'
RECEIVED_WORLD = f'{RECEIVED_PREFIX}world'


def validate_timeline(timeline: str) -> None:
    if not isinstance(timeline, str) or not timeline.strip():
        raise ValueError('A timeline name must be a non-empty string')


def validate_timelines(timelines: tuple[str, ...], *, allow_empty: bool = False) -> None:
    if not isinstance(timelines, tuple):
        raise TypeError('Timeline selection must be a tuple of names')
    if (not timelines and not allow_empty) or len(set(timelines)) != len(timelines):
        raise ValueError('Select a nonempty tuple of unique timeline names')
    for name in timelines:
        validate_timeline(name)


class Time(Mapping[str, int]):
    """Immutable integer coordinates on a nonempty set of named timelines."""

    __slots__ = ('_coordinates',)

    def __init__(self, /, **timestamps: SupportsIndex):
        if not timestamps:
            raise ValueError('Time must contain at least one timeline')
        coordinates = {}
        for name, value in timestamps.items():
            validate_timeline(name)
            if isinstance(value, bool | np.bool_):
                raise TypeError('Timeline coordinates must be integers, not booleans')
            coordinates[name] = index(value)
        self._coordinates: Mapping[str, int] = MappingProxyType(coordinates)

    @property
    def timelines(self) -> tuple[str, ...]:
        return tuple(self._coordinates)

    @overload
    def __getitem__(self, key: str) -> int: ...

    @overload
    def __getitem__(self, key: tuple[str, ...]) -> 'Time': ...

    def __getitem__(self, key: str | tuple[str, ...]) -> 'int | Time':
        if isinstance(key, str):
            return self._coordinates[key]
        if not isinstance(key, tuple):
            raise TypeError('Select a timeline name or a tuple of timeline names')
        if not key or len(set(key)) != len(key):
            raise ValueError('Select a nonempty tuple of unique timeline names')
        return Time(**{name: self._coordinates[name] for name in key})

    def __iter__(self) -> Iterator[str]:
        return iter(self._coordinates)

    def __len__(self) -> int:
        return len(self._coordinates)

    def __contains__(self, key: object) -> bool:
        return key in self._coordinates

    def __repr__(self) -> str:
        return f'Time(**{dict(self._coordinates)!r})'

    def __getstate__(self) -> dict[str, int]:
        return dict(self._coordinates)

    def __setstate__(self, coordinates: dict[str, int]) -> None:
        self.__init__(**coordinates)

    def _validate_timelines(self, other: 'Time') -> None:
        if self._coordinates.keys() != other._coordinates.keys():
            raise ValueError('Timestamp operations require the same timeline names')

    def __le__(self, other: 'Time') -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        self._validate_timelines(other)
        return all(value <= other[name] for name, value in self.items())

    def __lt__(self, other: 'Time') -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: 'Time') -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return other <= self

    def __gt__(self, other: 'Time') -> bool:
        if not isinstance(other, Time):
            return NotImplemented
        return other < self

    def __add__(self, other: 'Time') -> 'Time':
        if not isinstance(other, Time):
            return NotImplemented
        self._validate_timelines(other)
        return Time(**{name: value + other[name] for name, value in self.items()})

    def __sub__(self, other: 'Time') -> 'Time':
        if not isinstance(other, Time):
            return NotImplemented
        self._validate_timelines(other)
        return Time(**{name: value - other[name] for name, value in self.items()})


class Clock(ABC):
    """A clock is a source of timestamps. It can be system clock, or a more precise clock."""

    @abstractmethod
    def now(self) -> float:
        """Get current timestamp in seconds."""
        pass

    def time(self) -> Time:
        """Wall and world coordinates available in this process."""
        return Time(wall=time.monotonic_ns(), world=self.now_ns())

    def now_ns(self) -> int:
        """Get current timestamp in nanoseconds."""
        return int(self.now() * 1e9)


class SystemClock(Clock):
    def time(self) -> Time:
        return Time(wall=self.now_ns())

    def now(self) -> float:
        return time.monotonic()

    def now_ns(self) -> int:
        return time.monotonic_ns()


class VirtualClock(Clock):
    """Simulated-time clock owned and advanced by the World.

    Time does not pass on its own. As the scheduler works through its timeline it
    moves this clock forward to the next scheduled event, so simulated time runs as
    fast as the machine allows and is decoupled from any engine's internal time.
    The clock is kept in integer nanoseconds — the resolution recorded timestamps use —
    so the scheduler reasons on one exact grid. Only the World advances it; control
    systems just read ``now()``/``now_ns()``.
    """

    def __init__(self):
        self._time_ns = 0

    def now(self) -> float:
        return self._time_ns / 1e9

    def now_ns(self) -> int:
        return self._time_ns

    def advance_to_ns(self, target_ns: int) -> None:
        self._time_ns = max(self._time_ns, target_ns)
