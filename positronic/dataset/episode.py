import platform
import shutil
import stat
import sys
import time
from functools import lru_cache, partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, TypeVar

import json
import numpy as np

from positronic.utils.git import get_git_state

from .core import BaseEpisode, EpisodeWriter, Signal
from .vector import SimpleSignal, SimpleSignalWriter
from .video import VideoSignal, VideoSignalWriter

EPISODE_SCHEMA_VERSION = 1
T = TypeVar('T')
SIGNAL_FACTORY_T = Callable[[], Signal[Any]]


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


class DiskEpisodeWriter(EpisodeWriter):
    """Writer for recording episode data containing multiple signals."""

    def __init__(self, directory: Path) -> None:
        """Initialize episode writer.

        Args:
            directory: Directory to write episode data to (must not exist)
        """
        self._path = directory
        assert not self._path.exists(), f"Writing to existing directory {self._path}"
        # Create the episode directory for output files
        self._path.mkdir(parents=True, exist_ok=False)

        self._writers = {}
        # Accumulated static items to be stored in a single static.json
        self._static_items: dict[str, Any] = {}
        self._finished = False
        self._aborted = False

        # Write system metadata immediately
        meta = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "created_ts_ns": time.time_ns(),
        }
        meta["writer"] = _cached_env_writer_info()
        meta["writer"]["name"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        with (self._path / "meta.json").open('w', encoding='utf-8') as f:
            json.dump(meta, f)

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
            if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 3 and data.shape[2] == 3:
                # Image signal -> route to video writer
                video_path = self._path / f"{signal_name}.mp4"
                frames_index = self._path / f"{signal_name}.frames.parquet"
                self._writers[signal_name] = VideoSignalWriter(video_path, frames_index)
            else:
                # Scalar/vector signal
                self._writers[signal_name] = SimpleSignalWriter(self._path / f"{signal_name}.parquet")

        self._writers[signal_name].append(data, ts_ns)

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
            raise ValueError(
                "Static item must be JSON-serializable: dict/list over numbers and strings")
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


class _TimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'BaseEpisode') -> None:
        self.episode = episode

    def __getitem__(self, index_or_slice):
        """Access all items by timestamp or time selection.

        - Integer timestamp: returns a mapping of all dynamic signals sampled at
          or before the timestamp, merged with all static items.
        - Slice/list/ndarray: returns a new Episode with each dynamic signal
          sliced/sampled accordingly and all static items preserved.
        """
        if isinstance(index_or_slice, int):
            # Merge sampled dynamic values with static items
            sampled = {key: self.episode[key].time[index_or_slice] for key in self.episode.signal_keys()}
            return {**self.episode._static, **sampled}
        elif isinstance(index_or_slice, (list, tuple, np.ndarray)) or isinstance(index_or_slice, slice):
            # Return a view with sliced/sampled signals and preserved static
            signals = {key: self.episode[key].time[index_or_slice] for key in self.episode.signal_keys()}
            return Episode(signals, dict(self.episode._static), dict(self.episode.meta))
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    @property
    def start_ts(self) -> int:
        """Return the start timestamp of the episode in nanoseconds."""
        return self.episode.start_ts

    @property
    def last_ts(self) -> int:
        """Return the last timestamp of the episode in nanoseconds."""
        return self.episode.last_ts

    @property
    def duration(self) -> int:
        """Return the duration of the episode in nanoseconds."""
        return self.last_ts - self.start_ts


class Episode(BaseEpisode):
    """Reader for episode data containing multiple signals.

    An Episode represents a collection of signals recorded together,
    typically during a single robotic episode or data collection session.
    All signals in an episode share a common timeline.
    """

    def __init__(self, signals: dict[str, Signal[Any]], static: dict[str, Any], meta: dict[str, Any]) -> None:
        """Initialize episode reader from a directory (lazy).

        - Defers loading of signal data, static items, and meta until accessed.
        - Prepares lightweight factories for signals discovered on disk.
        """
        self._signals = signals
        self._static = static
        self._meta = meta

    @property
    def start_ts(self):
        """Return the latest start timestamp across all signals in the episode.

        Returns:
            int: Start timestamp in nanoseconds
        """
        values = [self[name].start_ts for name in self.signal_keys()]
        if not values:
            raise ValueError("Episode has no signals")
        return max(values)

    @property
    def last_ts(self):
        """Return the latest end timestamp across all signals in the episode.

        Returns:
            int: Last timestamp in nanoseconds
        """
        values = [self[name].last_ts for name in self.signal_keys()]
        if not values:
            raise ValueError("Episode has no signals")
        return max(values)

    @property
    def time(self):
        """Return a time-based indexer for accessing all signals by timestamp.

        Returns:
            _TimeIndexer: Indexer that allows time-based access to all signals
        """
        return _TimeIndexer(self)

    def keys(self) -> list[str]:
        """Return the names of all items in this episode.

        Returns:
            dict_keys: Item names (both dynamic signals and static items)
        """
        return self.signal_keys() + self.static_keys()

    def signal_keys(self) -> list[str]:
        return list(self._signals.keys())

    def static_keys(self) -> list[str]:
        return list(self._static.keys())

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
        if name in self._signals:
            return self._signals[name]
        if name in self._static:
            return self._static[name]
        raise KeyError(name)

    @property
    def meta(self) -> dict:
        return dict(self._meta)


def load_episode_from_disk(directory: Path | str) -> BaseEpisode:
    signals = {}
    static = {}
    meta = {}
    used_names: set[str] = set()

    directory = Path(directory)

    # Load video signals
    for video_file in directory.glob('*.mp4'):
        name = video_file.stem
        frames_idx = directory / f"{name}.frames.parquet"
        if not frames_idx.exists():
            raise ValueError(f"Video file {video_file} has no frames index {frames_idx}")
        signals[name] = VideoSignal(video_file, frames_idx)
        used_names.add(name)

    # Load scalar/vector signals
    for file in directory.glob('*.parquet'):
        fname = file.name
        if fname.endswith('.frames.parquet'):
            continue
        key = fname[:-len('.parquet')]
        if key in used_names:
            continue
        signals[key] = SimpleSignal(file)

    # Load static items
    ep_json = directory / 'static.json'
    if ep_json.exists():
        with ep_json.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            static.update(data)
        else:
            raise ValueError("static.json must contain a JSON object (mapping)")

    # Load meta
    meta_json = directory / 'meta.json'
    if meta_json.exists():
        with meta_json.open('r', encoding='utf-8') as f:
            try:
                meta_data = json.load(f)
                if isinstance(meta_data, dict):
                    meta.update(meta_data)
            except Exception:
                pass

    return Episode(signals, static, meta)