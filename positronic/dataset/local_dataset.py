from pathlib import Path
from typing import Sequence

import numpy as np

from .core import Dataset, DatasetWriter
from .episode import Episode, DiskEpisodeWriter, load_episode_from_disk


def _is_numeric_dir(p: Path) -> bool:
    """Return True if path is a directory named as zero-padded 12-digit number."""
    name = p.name
    return p.is_dir() and name.isdigit() and len(name) == 12


def _ensure_block_dir(root: Path, episode_id: int) -> Path:
    block_start = (episode_id // 1000) * 1000
    block_dir = root / f"{block_start:012d}"
    block_dir.mkdir(parents=True, exist_ok=True)
    return block_dir


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

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._episode_paths: list[tuple[int, Path]] = []
        self._episodes: dict[Path, Episode] = {}
        self._build_episode_list()

    def _build_episode_list(self) -> None:
        episode_ids_paths: list[tuple[int, Path]] = []
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist")
        for block_dir in sorted([p for p in self.root.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
            for ep_dir in sorted([p for p in block_dir.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
                episode_ids_paths.append((int(ep_dir.name), ep_dir))

        # Ensure episodes are sorted by id
        episode_ids_paths.sort(key=lambda x: x[0])
        self._episode_paths = [x[1] for x in episode_ids_paths]
        self._episodes = {}

    def __len__(self) -> int:
        return len(self._episode_paths)

    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray):
        if isinstance(index_or_slice, slice):
            # Return a list of Episodes for slices
            start, stop, step = index_or_slice.indices(len(self))
            return [self.get_episode_by_path(self._episode_paths[i]) for i in range(start, stop, step)]

        if isinstance(index_or_slice, (list, tuple, np.ndarray)):
            idxs = np.asarray(index_or_slice)
            if idxs.dtype == bool:
                raise TypeError("Boolean indexing is not supported")
            result = []
            for i in idxs:
                ii = int(i)
                if ii < 0:
                    ii += len(self)
                if not (0 <= ii < len(self)):
                    raise IndexError("Index out of range")
                result.append(self.get_episode_by_path(self._episode_paths[ii]))
            return result

        # Integer index
        i = int(index_or_slice)
        if i < 0:
            i += len(self)
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return self.get_episode_by_path(self._episode_paths[i])

    def get_episode_by_path(self, path: Path) -> Episode:
        if path in self._episodes:
            return self._episodes[path]
        episode = load_episode_from_disk(path)
        self._episodes[path] = episode
        return episode

class LocalDatasetWriter(DatasetWriter):
    """Writer that appends Episodes into a local directory structure.

    - Stores episodes under root / {block:012d} / {episode_id:012d}
    - Scans existing structure on init to continue episode numbering safely.
    - `new_episode()` allocates a new episode directory and returns a
      DiskEpisodeWriter.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
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

    def new_episode(self) -> DiskEpisodeWriter:
        eid = self._next_episode_id
        self._next_episode_id += 1  # Reserve id immediately

        block_dir = _ensure_block_dir(self.root, eid)
        # Do NOT create the episode directory here; DiskEpisodeWriter is
        # responsible for creating it and expects it to not exist yet.
        ep_dir = block_dir / f"{eid:012d}"

        writer = DiskEpisodeWriter(ep_dir)
        return writer
