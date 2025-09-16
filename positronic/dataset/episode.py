import json
import platform
import shutil
import sys
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from functools import lru_cache, partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np

from positronic.utils.git import get_git_state

from .signal import Signal
from .vector import SimpleSignal, SimpleSignalWriter
from .video import VideoSignal, VideoSignalWriter

EPISODE_SCHEMA_VERSION = 1
T = TypeVar('T')
SIGNAL_FACTORY_T = Callable[[], Signal[Any]]


class _EpisodeTimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'Episode') -> None:
        self.episode = episode

    def __getitem__(self, index_or_slice):
        match index_or_slice:
            case int() | np.integer() | float() | np.floating() as ts:
                # For a single timestamp, return static items and only the values for signals
                sampled = {key: sig.time[ts][0] for key, sig in self.episode.signals.items()}
                return {**self.episode.static, **sampled}
            case slice() as sl if sl.step is None:
                raise KeyError("Episode.time[start:stop] is not supported; use a step or explicit timestamps")
            case slice() | list() | tuple() | np.ndarray() as req:
                # For slice or sequence of timestamps, return a dict:
                # - static items as-is
                # - dynamic signals mapped to sequences of values sampled at requested timestamps
                # If slice with step but no stop provided, default stop to episode.last_ts (+1 for end-exclusive)
                if isinstance(req, slice) and req.step is not None and req.stop is None:
                    req = slice(req.start, self.episode.last_ts + 1, req.step)
                result: dict[str, Any] = self.episode.static.copy()
                for key, sig in self.episode.signals.items():
                    view = sig.time[req]
                    # Extract the full sequence of values corresponding to the time selection
                    result[key] = view._values_at(slice(None))
                return {**self.episode.static, **result}
            case _:
                raise TypeError(f"Invalid index type: {type(index_or_slice)}")


class Episode(ABC):
    """Abstract base class for an Episode (core concept)."""

    @property
    @abstractmethod
    def keys(self) -> Sequence[str]:
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> Signal[Any] | Any:
        pass

    @property
    @abstractmethod
    def meta(self) -> dict:
        pass

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        out: dict[str, Signal[Any]] = {}
        for k in self.keys:
            v = self[k]
            if isinstance(v, Signal):
                out[k] = v
        return out

    @property
    def static(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in self.keys:
            v = self[k]
            if not isinstance(v, Signal):
                out[k] = v
        return out

    @property
    def start_ts(self):
        values = [sig.start_ts for sig in self.signals.values()]
        if not values:
            raise ValueError("Episode has no signals")
        return max(values)

    @property
    def last_ts(self):
        values = [sig.last_ts for sig in self.signals.values()]
        if not values:
            raise ValueError("Episode has no signals")
        return max(values)

    @property
    def duraion_ns(self):
        return self.last_ts - self.start_ts

    @property
    def time(self):
        return _EpisodeTimeIndexer(self)


class EpisodeContainer(Episode):
    """In-memory view over an Episode's items."""

    def __init__(self,
                 signals: dict[str, Signal[Any]],
                 static: dict[str, Any] | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self._signals = signals
        self._static = static or {}
        self._meta = meta or {}

    @property
    def start_ts(self) -> int:
        return max([signal.start_ts for signal in self._signals.values()]) if self._signals else 0

    @property
    def last_ts(self) -> int:
        return max([signal.last_ts for signal in self._signals.values()]) if self._signals else 0

    @property
    def keys(self):
        return {**{k: True for k in self._signals.keys()}, **{k: True for k in self._static.keys()}}.keys()

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name in self._signals:
            return self._signals[name]
        if name in self._static:
            return self._static[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        return dict(self._signals)

    @property
    def static(self) -> dict[str, Any]:
        return dict(self._static)


def _is_valid_static_value(value: Any) -> bool:
    """Validate that a value conforms to our restricted JSON static schema.

    Allowed:
      - dict with string keys and values that are valid static values
      - list with elements that are valid static values
      - leaf values: str, int, float, bool
    """
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_valid_static_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_valid_static_value(v) for k, v in value.items())
    return False


class EpisodeWriter(AbstractContextManager, ABC, Generic[T]):
    """Abstract interface for recording an episode's dynamic and static data."""

    @abstractmethod
    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        """Append a sample for the named signal."""
        pass

    @abstractmethod
    def set_signal_meta(self, signal_name: str, *, names: Sequence[str] | None = None) -> None:
        """Declare metadata for a pending signal before writing samples."""
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


class DiskEpisodeWriter(EpisodeWriter):
    """Writer for recording episode data containing multiple signals."""

    def __init__(self, directory: Path, *, on_close: Callable[['DiskEpisodeWriter'], None] | None = None) -> None:
        """Initialize episode writer.

        Args:
            directory: Directory to write episode data to (must not exist)
        """
        self._path = directory
        assert not self._path.exists(), f"Writing to existing directory {self._path}"
        # Create the episode directory for output files
        self._path.mkdir(parents=True, exist_ok=False)

        self._writers: dict[str, SimpleSignalWriter | VideoSignalWriter] = {}
        self._signal_names: dict[str, list[str] | None] = {}
        # Accumulated static items to be stored in a single static.json
        self._static_items: dict[str, Any] = {}
        self._finished = False
        self._aborted = False
        self._on_close = on_close

        # Write system metadata immediately
        meta = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "created_ts_ns": time.time_ns(),
        }
        meta["writer"] = _cached_env_writer_info()
        meta["writer"]["name"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        with (self._path / "meta.json").open('w', encoding='utf-8') as f:
            json.dump(meta, f)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        """Append data to a named signal.

        Args:
            signal_name: Name of the signal to append to
            data: Data to append
            ts_ns: Timestamp in nanoseconds
        """
        if self._finished:
            raise RuntimeError("Cannot append to a finished writer")
        if self._aborted:
            raise RuntimeError("Cannot append to an aborted writer")
        if signal_name in self._static_items:
            raise ValueError(f"Static item '{signal_name}' already set for this episode")

        # Create writer on first append, choosing vector vs video based on data shape/dtype
        if signal_name not in self._writers:
            names = self._signal_names.get(signal_name)
            if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 3 and data.shape[2] == 3:
                if names is not None:
                    raise ValueError("Custom names are not supported for video signals")
                # Image signal -> route to video writer
                video_path = self._path / f"{signal_name}.mp4"
                frames_index = self._path / f"{signal_name}.frames.parquet"
                self._writers[signal_name] = VideoSignalWriter(video_path, frames_index)
                self._signal_names[signal_name] = None
            else:
                # Scalar/vector signal
                self._writers[signal_name] = SimpleSignalWriter(self._path / f"{signal_name}.parquet",
                                                                drop_equal_bytes_threshold=128,
                                                                names=names)
                self._signal_names[signal_name] = names

        self._writers[signal_name].append(data, ts_ns)

    def set_signal_meta(self, signal_name: str, *, names: Sequence[str] | None = None) -> None:
        if self._finished:
            raise RuntimeError("Cannot set signal meta on a finished writer")
        if self._aborted:
            raise RuntimeError("Cannot set signal meta on an aborted writer")
        if signal_name in self._writers:
            raise ValueError(f"Signal '{signal_name}' already has data written")
        if signal_name in self._signal_names:
            raise ValueError(f"Signal '{signal_name}' already has metadata set")
        self._signal_names[signal_name] = names

    def set_static(self, name: str, data: Any) -> None:
        """Set a static (non-time-varying) item by key for this episode.

        All static items are persisted together into a single 'static.json'.

        Args:
            name: The key name for the static item
            data: JSON-serializable value to store

        Raises:
            ValueError: If the key has already been set or value not serializable
        """
        if self._finished:
            raise RuntimeError("Cannot set static on a finished writer")
        if self._aborted:
            raise RuntimeError("Cannot set static on an aborted writer")
        if name in self._writers:
            raise ValueError(f"Signal '{name}' already exists for this episode")

        if name in self._static_items:
            raise ValueError(f"Static item '{name}' already set for this episode")

        # Validate restricted JSON structure
        if not _is_valid_static_value(data):
            raise ValueError("Static item must be JSON-serializable: dict/list over numbers and strings")
        self._static_items[name] = data

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize all signal writers and persist static items on context exit."""
        # Always try to close all signal writers
        for writer in self._writers.values():
            try:
                writer.__exit__(exc_type, exc, tb)
            except Exception:
                # Do not suppress exceptions, just ensure we attempt to close others
                if exc is None:
                    raise

        if self._aborted:
            return
        self._finished = True

        # Write all static items into a single static.json
        episode_json = self._path / "static.json"
        if self._static_items or not episode_json.exists():
            with episode_json.open('w', encoding='utf-8') as f:
                json.dump(self._static_items, f)

        if exc_type is None and not self._aborted and self._on_close is not None:
            self._on_close(self)

    def abort(self) -> None:
        """Abort writing: close resources and remove episode directory."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError("Cannot abort a finished writer")

        for w in list(self._writers.values()):
            w.abort()

        shutil.rmtree(self._path, ignore_errors=True)
        self._aborted = True


@lru_cache(maxsize=1)
def _cached_env_writer_info() -> dict:
    info = {
        "python": sys.version.split(" ")[0],
        "platform": platform.platform(),
    }
    try:
        info["version"] = importlib_metadata.version("positronic")
    except Exception:
        info["version"] = ''
    git_state = get_git_state()
    if git_state is not None:
        info['git'] = git_state
    return info


class DiskEpisode(Episode):
    """Reader for episode data containing multiple signals.

    An Episode represents a collection of signals recorded together,
    typically during a single robotic episode or data collection session.
    All signals in an episode share a common timeline.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize episode reader from a directory (lazy).

        - Defers loading of signal data, static items, and meta until accessed.
        - Prepares lightweight factories for signals discovered on disk.
        """
        self._dir = directory
        # Lazy containers
        self._signals: dict[str, Signal[Any]] = {}
        self._signal_factories: dict[str, SIGNAL_FACTORY_T] = {}
        self._static: dict[str, Any] | None = None
        self._meta: dict[str, Any] | None = None

        # Discover available signal files but do not instantiate readers yet
        used_names: set[str] = set()
        for video_file in self._dir.glob('*.mp4'):
            name = video_file.stem
            frames_idx = self._dir / f"{name}.frames.parquet"
            if not frames_idx.exists():
                raise ValueError(f"Video file {video_file} has no frames index {frames_idx}")
            self._signal_factories[name] = partial(VideoSignal, video_file, frames_idx)
            used_names.add(name)

        for file in self._dir.glob('*.parquet'):
            fname = file.name
            if fname.endswith('.frames.parquet'):
                continue
            key = fname[:-len('.parquet')]
            if key in used_names:
                continue
            self._signal_factories[key] = partial(SimpleSignal, file)

    # ---- lazy loaders ----
    def _ensure_signal(self, name: str) -> Signal[Any]:
        if name in self._signals:
            return self._signals[name]
        if name not in self._signal_factories:
            raise KeyError(name)
        factory = self._signal_factories[name]
        sig = factory()
        self._signals[name] = sig
        return sig

    def _iter_signals(self):
        for name in self._signal_factories.keys():
            yield name, self._ensure_signal(name)

    @property
    def _static_data(self) -> dict[str, Any]:
        if self._static is None:
            self._static = {}
            ep_json = self._dir / 'static.json'
            if ep_json.exists():
                with ep_json.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._static.update(data)
                else:
                    raise ValueError("static.json must contain a JSON object (mapping)")
        return self._static

    @property
    def keys(self):
        """Return the names of all items in this episode.

        Returns:
            dict_keys: Item names (both dynamic signals and static items)
        """
        dyn = {k: True for k in self._signal_factories.keys()}
        stat = {k: True for k in self._static_data.keys()}
        return {**dyn, **stat}.keys()

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        """Get an item (dynamic Signal or static value) by name.

        Args:
            name: Name of the item to retrieve

        Returns:
            - Signal[Any]: if the name corresponds to a dynamic signal
            - Any: the static value itself if the name corresponds to a static item

        Raises:
            KeyError: If name is not found in the episode
        """
        if name in self._signal_factories:
            return self._ensure_signal(name)
        if name in self._static_data:
            return self._static_data[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict:
        if self._meta is None:
            meta: dict[str, Any] = {}
            meta_json = self._dir / 'meta.json'
            if meta_json.exists():
                with meta_json.open('r', encoding='utf-8') as f:
                    try:
                        meta_data = json.load(f)
                        if isinstance(meta_data, dict):
                            meta.update(meta_data)
                    except Exception:
                        pass
            self._meta = meta
        return dict(self._meta)

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        out: dict[str, Signal[Any]] = {}
        for name in self._signal_factories.keys():
            out[name] = self._ensure_signal(name)
        return out

    @property
    def static(self) -> dict[str, Any]:
        return dict(self._static_data)
