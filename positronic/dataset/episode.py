import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar, overload

from pimm.time import RECEIVED_WORLD

from .signal import RECORDED_TIME, Signal
from .time import HARNESS_WORLD, Time, TimeBounds, TimeGrid, validate_queries, validate_timeline, validate_timelines

EPISODE_SCHEMA_VERSION = 1
# Where the episode is written, in the meta of both the episode and the writer that made it.
META_PATH = 'path'
# The episode's identity, in the meta of both the episode and the writer that made it.
META_UID = 'uid'
# When the episode was opened, in the meta of both the episode and the writer that made it.
META_CREATED_TS_NS = 'created_ts_ns'
# The video encoder spec, under the ``writer`` entry of the episode meta.
META_WRITER_VIDEO_ENCODER = 'video_encoder'
T = TypeVar('T')
SIGNAL_FACTORY_T = Callable[[], Signal[Any]]

_BYTES_TAG = '__bytes_b64__'


class _StaticEncoder(json.JSONEncoder):
    """JSON encoder that handles ``bytes`` values via base64."""

    def default(self, o):
        if isinstance(o, bytes):
            return {_BYTES_TAG: base64.b64encode(o).decode('ascii')}
        return super().default(o)


def _static_decode_hook(obj: dict) -> Any:
    if _BYTES_TAG in obj and len(obj) == 1:
        return base64.b64decode(obj[_BYTES_TAG])
    return obj


def _is_valid_static_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool | bytes):
        return True
    if isinstance(value, list | tuple):
        return all(_is_valid_static_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_valid_static_value(v) for k, v in value.items())
    return False


class _EpisodeTimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'Episode') -> None:
        self.episode = episode

    def __getitem__(self, request: Time | slice | Sequence[Time]):
        if isinstance(request, Time):
            sampled = {
                name: signal.time[request][0] for name, signal in self.episode._signals_on(request.timelines).items()
            }
            return {**self.episode.static, **sampled}
        if isinstance(request, slice):
            if request.step is None:
                raise KeyError('Episode.time[start:stop] is not supported; use a step or explicit timestamps')
            if request.start is None:
                raise ValueError('Slice start is required when step is provided')
            start, stop, step = request.start, request.stop, request.step
            if any(not isinstance(time, Time) for time in (start, stop, step) if time is not None):
                raise TypeError('Time slice endpoints and step must be Time values')
            stop = stop if stop is not None else self.episode.bounds(start.timelines).finish
            request = TimeGrid(start, stop, step, inclusive=request.stop is None)
            names = start.timelines
        elif isinstance(request, Sequence):
            validate_queries(request)
            if not len(request):
                return {
                    **self.episode.static,
                    **{name: signal[:0].values() for name, signal in self.episode.signals.items()},
                }
            names = request[0].timelines
        else:
            raise TypeError('Expected Time, a time slice, or a sequence of Time')
        signals = self.episode._signals_on(names)
        return {**self.episode.static, **{name: signal.time[request].values() for name, signal in signals.items()}}


class Episode(ABC, Mapping[str, Any]):
    """Abstract base class for an Episode (core concept).

    Subclasses must implement the following methods:
    - __getitem__ - return a Signal or static value by name
    - __iter__ - return an iterator over the keys
    - __len__ - return the number of keys
    """

    @property
    @abstractmethod
    def meta(self) -> dict:
        pass

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        out: dict[str, Signal[Any]] = {}
        for k in self:
            v = self[k]
            if isinstance(v, Signal):
                out[k] = v
        return out

    @property
    def timelines(self) -> tuple[str, ...]:
        """All timeline names present in at least one signal."""
        return tuple(dict.fromkeys(name for signal in self.signals.values() for name in signal.timelines))

    @property
    def static(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in self:
            v = self[k]
            if not isinstance(v, Signal):
                out[k] = v
        return out

    def _signals_on(self, timelines: tuple[str, ...]) -> dict[str, Signal[Any]]:
        validate_timelines(timelines)
        return {name: signal for name, signal in self.signals.items() if set(timelines).issubset(signal.timelines)}

    @overload
    def bounds(self, timelines: str) -> TimeBounds[int]: ...

    @overload
    def bounds(self, timelines: tuple[str, ...]) -> TimeBounds[Time]: ...

    def bounds(self, timelines: str | tuple[str, ...]) -> TimeBounds[int] | TimeBounds[Time]:
        """Coordinatewise latest starts and finishes of signals on every selected timeline."""
        names = (timelines,) if isinstance(timelines, str) else timelines
        bounds = [signal.bounds(names) for signal in self._signals_on(names).values()]
        if not bounds:
            raise ValueError('Episode has no signals on the requested timelines')
        start = Time(**{name: max(bound.start[name] for bound in bounds) for name in names})
        finish = Time(**{name: max(bound.finish[name] for bound in bounds) for name in names})
        if isinstance(timelines, str):
            return TimeBounds(start[timelines], finish[timelines])
        return TimeBounds(start, finish)

    @property
    def time(self):
        return _EpisodeTimeIndexer(self)


class EpisodeContainer(Episode):
    """In-memory view over an Episode's items."""

    def __init__(self, data: dict[str, Signal[Any] | Any], meta: dict[str, Any] | None = None) -> None:
        self._data = data
        self._meta = meta or {}

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        return self._data[name]

    @property
    def meta(self) -> dict:
        return self._meta.copy()


class EpisodeWriter(AbstractContextManager, ABC, Generic[T]):
    """Abstract interface for recording an episode's dynamic and static data."""

    @abstractmethod
    def append(self, signal_name: str, data: T, timestamps: Time) -> None:
        """Append a sample for the named signal."""
        pass

    @abstractmethod
    def set_static(self, name: str, data: Any) -> None:
        """Record a static (per-episode) item by key."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize resources on context-manager exit."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """Abort the write and discard any partially written data."""
        pass

    @property
    def meta(self) -> dict:
        """Metadata for the episode, known at the time of request."""
        return {}


def select_timeline(timelines: Iterable[str], *, timeline: str | None = None) -> str:
    """Select an explicit timeline or prefer Harness, receipt, then legacy recorded world time."""
    available = set(timelines)
    if timeline is not None:
        validate_timeline(timeline)
        if timeline not in available:
            raise KeyError(timeline)
        return timeline
    for name in (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME):
        if name in available:
            return name
    raise ValueError(f'Select an explicit timeline from {sorted(available)}')
