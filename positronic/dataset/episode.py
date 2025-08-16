from pathlib import Path
from typing import Any, TypeVar

from .core import Signal
from .vector import SimpleSignal, SimpleSignalWriter

T = TypeVar('T')


class EpisodeWriter:
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

    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        """Append data to a named signal.

        Args:
            signal_name: Name of the signal to append to
            data: Data to append
            ts_ns: Timestamp in nanoseconds
        """
        if signal_name not in self._writers:
            self._writers[signal_name] = SimpleSignalWriter(self._path / f"{signal_name}.parquet")

        self._writers[signal_name].append(data, ts_ns)

    def finish(self):
        """Finish writing all signals."""
        for writer in self._writers.values():
            writer.finish()


class _TimeIndexer:
    """Time-based indexer for Episode signals."""

    def __init__(self, episode: 'Episode') -> None:
        self.episode = episode

    def __getitem__(self, index_or_slice):
        """Access all signals by timestamp or time slice."""
        if isinstance(index_or_slice, int):
            return {key: signal.time[index_or_slice] for key, signal in self.episode._signals.items()}
        else:
            res = Episode.__new__(Episode)
            res._signals = {key: signal.time[index_or_slice] for key, signal in self.episode._signals.items()}
            return res


class Episode:
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
        for file in directory.glob('*.parquet'):
            # TODO: Support video, when it's here
            key = file.name[:-len('.parquet')]
            self._signals[key] = SimpleSignal(file)

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
        """Return the names of all signals in this episode.

        Returns:
            dict_keys: Signal names
        """
        return self._signals.keys()

    def __getitem__(self, signal_name: str) -> Signal[Any]:
        """Get a signal by name.

        Args:
            signal_name: Name of the signal to retrieve

        Returns:
            Signal[Any]: The requested signal

        Raises:
            KeyError: If signal_name is not found in the episode
        """
        return self._signals[signal_name]
