from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import lru_cache, partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from positronic.utils.git import get_git_state
from positronic.utils.lazy_dict import LazyDict

from .dataset import ConcatDataset, Dataset, DatasetWriter
from .episode import EPISODE_SCHEMA_VERSION, SIGNAL_FACTORY_T, Episode, EpisodeWriter, T
from .signal import Signal
from .vector import SimpleSignal, SimpleSignalWriter
from .video import VideoSignal, VideoSignalWriter

UNFINISHED_MARKER = '.unfinished'


def _is_valid_static_value(value: Any) -> bool:
    if isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list | tuple):
        return all(_is_valid_static_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_valid_static_value(v) for k, v in value.items())
    return False


def _is_numeric_dir(p: Path) -> bool:
    """Return True if path is a directory named as zero-padded 12-digit number."""
    name = p.name
    return p.is_dir() and name.isdigit() and len(name) == 12


def _ensure_block_dir(root: Path, episode_id: int) -> Path:
    block_start = (episode_id // 1000) * 1000
    block_dir = root / f'{block_start:012d}'
    block_dir.mkdir(parents=True, exist_ok=True)
    return block_dir


def _mark_unfinished(path: Path) -> None:
    marker = path / UNFINISHED_MARKER
    marker.write_text('unfinished', encoding='utf-8')


def _clear_unfinished(path: Path) -> None:
    marker = path / UNFINISHED_MARKER
    with suppress(FileNotFoundError):
        marker.unlink()


@lru_cache(maxsize=1)
def _cached_env_writer_info() -> dict:
    info = {'python': sys.version.split(' ')[0], 'platform': platform.platform()}
    try:
        info['version'] = importlib_metadata.version('positronic')
    except Exception:
        info['version'] = ''
    git_state = get_git_state()
    if git_state is not None:
        info['git'] = git_state
    return info


class DiskEpisodeWriter(EpisodeWriter):
    """Writer for recording episode data containing multiple signals."""

    def __init__(
        self,
        directory: Path,
        *,
        on_close: Callable[[DiskEpisodeWriter], None] | None = None,
        created_ts_ns: int | None = None,
    ) -> None:
        """Initialize episode writer.

        Args:
            directory: Directory to write episode data to (must not exist)
            on_close: Optional callback invoked after successful episode close
            created_ts_ns: Optional creation timestamp (defaults to current time).
                Use this to preserve original creation time during migration.
        """
        self._path = directory
        assert not self._path.exists(), f'Writing to existing directory {self._path}'
        # Create the episode directory for output files
        self._path.mkdir(parents=True, exist_ok=False)
        _mark_unfinished(self._path)

        self._writers: dict[str, SimpleSignalWriter | VideoSignalWriter] = {}
        # Accumulated static items to be stored in a single static.json
        self._static_items: dict[str, Any] = {}
        self._finished = False
        self._aborted = False
        self._on_close = on_close

        # Write system metadata immediately
        # NB: falsy created_ts_ns (including 0) defaults to current time — epoch 0 is not a valid episode timestamp
        self._meta = {'schema_version': EPISODE_SCHEMA_VERSION, 'created_ts_ns': created_ts_ns or time.time_ns()}
        self._meta['writer'] = _cached_env_writer_info()
        self._meta['writer']['name'] = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        self._meta['path'] = str(self._path.resolve(strict=True))

    @property
    def path(self) -> Path:
        return self._path

    def append(self, signal_name: str, data: T, ts_ns: int, extra_ts: dict[str, int] | None = None) -> None:
        """Append data to a named signal.

        Args:
            signal_name: Name of the signal to append to
            data: Data to append
            ts_ns: Timestamp in nanoseconds
            extra_ts: Optional dict of extra timeline names to timestamps
        """
        if self._finished:
            raise RuntimeError(f'Cannot append to a finished writer {self._path}')
        if self._aborted:
            raise RuntimeError(f'Cannot append to an aborted writer {self._path}')
        if signal_name in self._static_items:
            raise ValueError(f"Static item '{signal_name}' already set for this episode {self._path}")

        # Create writer on first append, choosing vector vs video based on data shape/dtype
        if signal_name not in self._writers:
            if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 3 and data.shape[2] == 3:
                # Image signal -> route to video writer
                video_path = self._path / f'{signal_name}.mp4'
                frames_index = self._path / f'{signal_name}.frames.parquet'
                self._writers[signal_name] = VideoSignalWriter(video_path, frames_index)
            else:
                # Scalar/vector signal
                self._writers[signal_name] = SimpleSignalWriter(self._path / f'{signal_name}.parquet')

        self._writers[signal_name].append(data, ts_ns, extra_ts)

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
            raise RuntimeError('Cannot set static on a finished writer')
        if self._aborted:
            raise RuntimeError('Cannot set static on an aborted writer')
        if name in self._writers:
            raise ValueError(f"Signal '{name}' already exists for this episode")

        if name in self._static_items:
            raise ValueError(f"Static item '{name}' already set for this episode")

        # Validate restricted JSON structure
        if not _is_valid_static_value(data):
            raise ValueError(
                f'Static item must be JSON-serializable: dict/list over numbers and strings, but got {name}\n{data=!r}'
            )
        self._static_items[name] = data

    def _scan_timestamps(self) -> tuple[int | None, int | None]:
        """Scan parquet files for min/max timestamps."""
        first_ts: int | None = None
        last_ts: int | None = None

        for parquet_file in self._path.glob('*.parquet'):
            try:
                schema = pq.read_schema(parquet_file)
                # Vector signals use 'timestamp', video frames use 'ts_ns'
                if 'timestamp' in schema.names:
                    col_name = 'timestamp'
                elif 'ts_ns' in schema.names:
                    col_name = 'ts_ns'
                else:
                    continue

                table = pq.read_table(parquet_file, columns=[col_name])
                timestamps = table[col_name].to_pylist()
                if not timestamps:
                    continue

                file_first, file_last = min(timestamps), max(timestamps)
                if first_ts is None or file_first > first_ts:
                    first_ts = file_first
                if last_ts is None or file_last > last_ts:
                    last_ts = file_last
            except Exception:
                continue

        return first_ts, last_ts

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize all signal writers and persist static items on context exit."""
        if exc_type is not None and not self._aborted:
            with suppress(Exception):
                self.abort()
            return

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

        # Compute duration by scanning parquet files
        first_ts, last_ts = self._scan_timestamps()
        if first_ts is not None and last_ts is not None:
            self._meta['duration_ns'] = int(last_ts - first_ts)

        # Write all static items into a single static.json
        episode_json = self._path / 'static.json'
        if self._static_items or not episode_json.exists():
            with episode_json.open('w', encoding='utf-8') as f:
                json.dump(self._static_items, f, indent=2)

        with (self._path / 'meta.json').open('w', encoding='utf-8') as f:
            json.dump(self._meta, f, indent=2)

        _clear_unfinished(self._path)

        if exc_type is None and self._on_close is not None:
            self._on_close(self)

    def abort(self) -> None:
        """Abort writing: close resources and remove episode directory."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError('Cannot abort a finished writer')

        for w in list(self._writers.values()):
            w.abort()

        shutil.rmtree(self._path, ignore_errors=True)
        self._aborted = True

    @property
    def meta(self) -> dict:
        """Metadata for the episode, known at the time of request."""
        return self._meta.copy()


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
        if (directory / UNFINISHED_MARKER).exists():
            raise ValueError(f'Cannot read unfinished episode at {directory}')
        self._dir = directory
        # Lazy containers
        self._signals: dict[str, Signal[Any]] = {}
        self._signal_factories: dict[str, SIGNAL_FACTORY_T] = {}
        self._static: dict[str, Any] | None = None
        self._meta: dict[str, Any] | None = None
        self._cached_duration_ns: int | None = None

        # Discover available signal files but do not instantiate readers yet
        used_names: set[str] = set()
        for video_file in self._dir.glob('*.mp4'):
            name = video_file.stem
            frames_idx = self._dir / f'{name}.frames.parquet'
            if not frames_idx.exists():
                raise ValueError(f'Video file {video_file} has no frames index {frames_idx}')
            self._signal_factories[name] = partial(VideoSignal, video_file, frames_idx)
            used_names.add(name)

        for file in self._dir.glob('*.parquet'):
            fname = file.name
            if fname.endswith('.frames.parquet'):
                continue
            key = fname[: -len('.parquet')]
            if key in used_names:
                continue
            self._signal_factories[key] = partial(SimpleSignal, file)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(path={str(self._dir)!r})'

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
                    raise ValueError('static.json must contain a JSON object (mapping)')
        return self._static

    def __iter__(self) -> Iterator[str]:
        yield from self._signal_factories.keys()
        yield from self._static_data.keys()

    def __len__(self) -> int:
        return len(self._signal_factories) + len(self._static_data)

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
        available = list(self._signal_factories.keys()) + list(self._static_data.keys())
        raise KeyError(f"'{name}' not found in episode {self._dir}. Available keys: {', '.join(available)}")

    def _compute_size_mb(self) -> float:
        """Compute total size of episode directory in MB."""
        size_bytes = 0
        with suppress(OSError):
            for entry in self._dir.rglob('*'):
                if entry.is_file():
                    size_bytes += entry.stat().st_size
        return size_bytes / (1024 * 1024)

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

            # duration_ns is a first-class Episode property, not meta.
            # Extract it as a private cache for DiskEpisode.duration_ns.
            self._cached_duration_ns = meta.pop('duration_ns', None)

            meta['path'] = str(self._dir.expanduser().resolve(strict=False))

            lazy_getters: dict[str, Any] = {}
            if 'size_mb' not in meta:
                lazy_getters['size_mb'] = self._compute_size_mb

            self._meta = LazyDict(meta, lazy_getters)
        return self._meta.copy()

    @property
    def duration_ns(self):
        # Fast path: use cached value from meta.json (written at recording time)
        _ = self.meta  # ensure meta is loaded
        if self._cached_duration_ns is not None:
            return self._cached_duration_ns
        # Fallback: compute from signals (expensive, for old episodes without cached value)
        return super().duration_ns

    @property
    def signals(self) -> dict[str, Signal[Any]]:
        out: dict[str, Signal[Any]] = {}
        for name in self._signal_factories.keys():
            out[name] = self._ensure_signal(name)
        return out

    @property
    def static(self) -> dict[str, Any]:
        return dict(self._static_data)


class LocalDataset(Dataset):
    """Filesystem-backed dataset of Episodes.

    Layout:
      root/
        000000000000/      # block for episodes [0..999]
          000000000000/    # episode 0 (full 12-digit id)
          000000000001/    # episode 1
          ...
        000000001000/      # block for episodes [1000..1999]
          000000001000/    # episode 1000

    Each episode directory is readable by DiskEpisode.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        if not self.root.exists():
            raise FileNotFoundError(
                f'Dataset directory {self.root} does not exist. Check that the path is correct and accessible.'
            )
        self._episodes: list[tuple[int, Path]] = []
        self._build_episode_list()

    def _build_episode_list(self) -> None:
        self._episodes.clear()
        if not self.root.exists():
            return
        for block_dir in sorted([p for p in self.root.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
            for ep_dir in sorted([p for p in block_dir.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
                if (ep_dir / UNFINISHED_MARKER).exists():
                    continue
                ep_id = int(ep_dir.name)
                self._episodes.append((ep_id, ep_dir))

        # Ensure episodes are sorted by id
        self._episodes.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._episodes)

    def _get_episode(self, index: int) -> DiskEpisode:
        if not (0 <= index < len(self)):
            raise IndexError(f'Index {index} out of range for {self.root} dataset with {len(self)} episodes')
        return DiskEpisode(self._episodes[index][1])


class LocalDatasetWriter(DatasetWriter):
    """Writer that appends Episodes into a local directory structure.

    - Stores episodes under root / {block:012d} / {episode_id:012d}
    - Scans existing structure on init to continue episode numbering safely.
    - `new_episode()` allocates a new episode directory and returns a
      DiskEpisodeWriter.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_episode_id = self._compute_next_episode_id()

    def _compute_next_episode_id(self) -> int:
        max_id = -1
        for block_dir in self.root.iterdir():
            if not _is_numeric_dir(block_dir):
                continue
            for ep_dir in block_dir.iterdir():
                if not _is_numeric_dir(ep_dir):
                    continue
                eid = int(ep_dir.name)
                if eid > max_id:
                    max_id = eid
        return max_id + 1

    def new_episode(self, *, created_ts_ns: int | None = None) -> DiskEpisodeWriter:
        """Create a new episode writer.

        Args:
            created_ts_ns: Optional creation timestamp (defaults to current time).
                Use this to preserve original creation time during migration.
        """
        eid = self._next_episode_id
        self._next_episode_id += 1  # Reserve id immediately

        block_dir = _ensure_block_dir(self.root, eid)
        # Do NOT create the episode directory here; DiskEpisodeWriter is
        # responsible for creating it and expects it to not exist yet.
        ep_dir = block_dir / f'{eid:012d}'

        writer = DiskEpisodeWriter(ep_dir, created_ts_ns=created_ts_ns)
        return writer

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


def load_all_datasets(root: Path) -> Dataset:
    """Load and concatenate all LocalDatasets found in root directory tree.

    Uses breadth-first search to find datasets. For each directory, tries to load it
    as a LocalDataset. If successful, that branch is complete. If not, expands into
    subdirectories and tries them. Continues until all branches either resolve to
    datasets or have no more subdirectories to explore.

    Args:
        root: Path to directory that may be a dataset and/or contain dataset subdirectories

    Returns:
        Dataset: A ConcatDataset combining all found datasets, or a single LocalDataset
                 if only one is found

    Raises:
        FileNotFoundError: If root does not exist
        ValueError: If no valid datasets are found in root or if root is not a directory
    """

    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f'Directory {root} does not exist')

    if not root.is_dir():
        raise ValueError(f'{root} is not a directory')

    datasets: list[Dataset] = []
    # Queue of directories to explore
    to_explore: deque[Path] = deque([root])

    while to_explore:
        current_path = to_explore.popleft()

        # Try to load this directory as a dataset
        try:
            dataset = LocalDataset(current_path)
            if len(dataset) > 0:
                datasets.append(dataset)
                continue
        except (FileNotFoundError, ValueError):
            pass

        subdirs = sorted([p for p in current_path.iterdir() if p.is_dir()])
        to_explore.extend(subdirs)

    if len(datasets) == 0:
        raise ValueError(f'No valid datasets found in {root}')

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(*datasets)
