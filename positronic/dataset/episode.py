from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from json_tricks import dumps as json_dumps, loads as json_loads

from .core import Signal, Episode, EpisodeWriter
from .vector import SimpleSignal, SimpleSignalWriter
from .video import VideoSignal, VideoSignalWriter

T = TypeVar('T')


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
        # Accumulated static items to be stored in a single episode.json
        self._static_items: dict[str, Any] = {}

    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        """Append data to a named signal.

        Args:
            signal_name: Name of the signal to append to
            data: Data to append
            ts_ns: Timestamp in nanoseconds
        """
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

        All static items are persisted together into a single 'episode.json'.

        Args:
            name: The key name for the static item
            data: JSON-serializable value to store

        Raises:
            ValueError: If the key has already been set or value not serializable
        """
        if name in self._writers:
            raise ValueError(f"Signal '{name}' already exists for this episode")

        if name in self._static_items:
            raise ValueError(f"Static item '{name}' already set for this episode")

        self._static_items[name] = data

    def finish(self):
        """Finish writing all signals."""
        for writer in self._writers.values():
            writer.finish()
        # Write all static items into a single episode.json
        episode_json = self._path / "episode.json"
        if self._static_items or not episode_json.exists():
            with episode_json.open('w', encoding='utf-8') as f:
                f.write(json_dumps(self._static_items))


class _TimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'Episode') -> None:
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
            sampled = {key: signal.time[index_or_slice] for key, signal in self.episode._signals.items()}
            return {**self.episode._static, **sampled}
        elif isinstance(index_or_slice, (list, tuple, np.ndarray)) or isinstance(index_or_slice, slice):
            # Return a view with sliced/sampled signals and preserved static
            signals = {key: signal.time[index_or_slice] for key, signal in self.episode._signals.items()}
            return EpisodeView(signals, dict(self.episode._static))
        else:
            raise TypeError(f"Invalid index type: {type(index_or_slice)}")


class EpisodeView(Episode):
    """In-memory view over an Episode's items.

    Provides the same read API as Episode (keys, __getitem__, start_ts, last_ts, time),
    but does not load from disk. Chained time indexing returns another EpisodeView.
    """

    def __init__(self, signals: dict[str, Signal[Any]], static: dict[str, Any]) -> None:
        self._signals = signals
        self._static = static

    @property
    def start_ts(self) -> int:
        return max([signal.start_ts for signal in self._signals.values()]) if self._signals else 0

    @property
    def last_ts(self) -> int:
        return max([signal.last_ts for signal in self._signals.values()]) if self._signals else 0

    @property
    def time(self):
        return _TimeIndexer(self)

    @property
    def keys(self):
        return {**{k: True for k in self._signals.keys()}, **{k: True for k in self._static.keys()}}.keys()

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name in self._signals:
            return self._signals[name]
        if name in self._static:
            return self._static[name]
        raise KeyError(name)


class DiskEpisode(Episode):
    """Reader for episode data containing multiple signals.

    An Episode represents a collection of signals recorded together,
    typically during a single robotic episode or data collection session.
    All signals in an episode share a common timeline.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize episode reader from a directory.

        Args:
            directory: Directory containing signal files (*.parquet)
        """
        self._signals = {}
        self._static = {}
        # Build video signals first: pair *.mp4 with *.frames.parquet
        used_names: set[str] = set()
        for video_file in directory.glob('*.mp4'):
            name = video_file.stem
            frames_idx = directory / f"{name}.frames.parquet"
            if frames_idx.exists():
                self._signals[name] = VideoSignal(video_file, frames_idx)
                used_names.add(name)

        # Load remaining parquet files as vector/scalar signals, skipping frame index files
        for file in directory.glob('*.parquet'):
            fname = file.name
            if fname.endswith('.frames.parquet'):
                # handled as part of a video signal above
                continue
            key = fname[:-len('.parquet')]
            if key in used_names:
                continue
            self._signals[key] = SimpleSignal(file)
        # Load all static JSON items from the single episode.json file
        ep_json = directory / 'episode.json'
        if ep_json.exists():
            with ep_json.open('r', encoding='utf-8') as f:
                data = json_loads(f.read())
            if isinstance(data, dict):
                self._static.update(data)
            else:
                raise ValueError("episode.json must contain a JSON object (mapping)")

    @property
    def start_ts(self):
        """Return the latest start timestamp across all signals in the episode.

        Returns:
            int: Start timestamp in nanoseconds
        """
        return max([signal.start_ts for signal in self._signals.values()])

    @property
    def last_ts(self):
        """Return the latest end timestamp across all signals in the episode.

        Returns:
            int: Last timestamp in nanoseconds
        """
        return max([signal.last_ts for signal in self._signals.values()])

    @property
    def time(self):
        """Return a time-based indexer for accessing all signals by timestamp.

        Returns:
            _TimeIndexer: Indexer that allows time-based access to all signals
        """
        return _TimeIndexer(self)

    @property
    def keys(self):
        """Return the names of all items in this episode.

        Returns:
            dict_keys: Item names (both dynamic signals and static items)
        """
        return {**{k: True for k in self._signals.keys()}, **{k: True for k in self._static.keys()}}.keys()

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
