import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar

from .signal import Signal
from .time import Time, TimeGrid, as_time, validate_queries, validate_timelines

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

    def __getitem__(self, request):
        if isinstance(request, Mapping):
            query = as_time(request)
            sampled = {
                name: signal.time[query][0] for name, signal in self.episode._signals_on(query.timelines).items()
            }
            return {**self.episode.static, **sampled}
        if isinstance(request, slice):
            if request.step is None:
                raise KeyError('Episode.time[start:stop] is not supported; use a step or explicit timestamps')
            if request.start is None:
                raise ValueError('Slice start is required when step is provided')
            start, step = as_time(request.start), as_time(request.step)
            stop = as_time(request.stop) if request.stop is not None else self.episode.bounds(start.timelines)[1]
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
            raise TypeError('Expected named coordinates or a sequence of Time')
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

    def bounds(self, timelines: tuple[str, ...]) -> tuple[Time, Time]:
        bounds = [signal.bounds(timelines) for signal in self._signals_on(timelines).values()]
        if not bounds:
            raise ValueError('Episode has no signals on the requested timelines')
        start = Time(**{name: max(first[name] for first, _ in bounds) for name in timelines})
        stop = Time(**{name: max(last[name] for _, last in bounds) for name in timelines})
        return start, stop

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
