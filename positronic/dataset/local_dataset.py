from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .core import Dataset, DatasetWriter
from .episode import DiskEpisode, DiskEpisodeWriter


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

    def __init__(self, root: Path) -> None:
        self.root = root
        self._episodes: list[tuple[int, Path]] = []
        self._build_episode_list()

    def _build_episode_list(self) -> None:
        self._episodes.clear()
        if not self.root.exists():
            return
        for block_dir in sorted([p for p in self.root.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
            for ep_dir in sorted([p for p in block_dir.iterdir() if _is_numeric_dir(p)], key=lambda p: p.name):
                ep_id = int(ep_dir.name)
                self._episodes.append((ep_id, ep_dir))

        # Ensure episodes are sorted by id
        self._episodes.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray):
        if isinstance(index_or_slice, slice):
            # Return a list of Episodes for slices
            start, stop, step = index_or_slice.indices(len(self))
            return [DiskEpisode(self._episodes[i][1]) for i in range(start, stop, step)]

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
                result.append(DiskEpisode(self._episodes[ii][1]))
            return result

        # Integer index
        i = int(index_or_slice)
        if i < 0:
            i += len(self)
        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")
        return DiskEpisode(self._episodes[i][1])


class LocalDatasetWriter(DatasetWriter):
    """Writer that appends Episodes into a local directory structure.

    - Stores episodes under root / {block:012d} / {episode_id:012d}
    - Scans existing structure on init to continue episode numbering safely.
    - `new_episode()` allocates a new episode directory and returns a
      DiskEpisodeWriter.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
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
