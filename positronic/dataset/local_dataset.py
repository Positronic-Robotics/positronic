from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydoc import locate

from .dataset import Dataset, DatasetWriter
from .episode import DiskEpisode, DiskEpisodeWriter
from .signal import Kind, SignalMeta

_META_TAG = "__meta__"


def _encode_meta_value(value: Any) -> Any:
    if isinstance(value, np.dtype):
        return {_META_TAG: 'np_dtype', 'value': str(value)}
    if isinstance(value, type):
        return {_META_TAG: 'py_type', 'value': f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, tuple):
        return {_META_TAG: 'tuple', 'items': [_encode_meta_value(v) for v in value]}
    if isinstance(value, list):
        return {_META_TAG: 'list', 'items': [_encode_meta_value(v) for v in value]}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return {_META_TAG: 'repr', 'value': repr(value)}


def _decode_meta_value(value: Any) -> Any:
    if isinstance(value, dict) and _META_TAG in value:
        kind = value[_META_TAG]
        match kind:
            case 'np_dtype':
                return np.dtype(value['value'])
            case 'py_type':
                located = locate(value['value'])
                if located is None:
                    raise ValueError(f"Unable to locate type {value['value']}")
                return located
            case 'tuple':
                return tuple(_decode_meta_value(v) for v in value['items'])
            case 'list':
                return [_decode_meta_value(v) for v in value['items']]
            case 'repr':
                return value['value']
            case _:
                raise ValueError(f"Unsupported encoded meta kind: {kind}")
    return value


def _encode_signal_meta(meta: SignalMeta) -> dict[str, Any]:
    return {
        'dtype': _encode_meta_value(meta.dtype),
        'shape': _encode_meta_value(meta.shape),
        'kind': meta.kind.value,
        'names': list(meta.names) if meta.names is not None else None,
    }


def _decode_signal_meta(data: dict[str, Any]) -> SignalMeta:
    dtype = _decode_meta_value(data['dtype'])
    shape = _decode_meta_value(data['shape'])
    kind = Kind(data['kind'])
    return SignalMeta(dtype=dtype, shape=shape, kind=kind, names=data.get('names'))


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
        self._signals_meta: dict[str, SignalMeta] | None = None

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

    def _get_episode(self, index: int) -> DiskEpisode:
        if not (0 <= index < len(self)):
            raise IndexError("Index out of range")
        return DiskEpisode(self._episodes[index][1])

    @property
    def signals_meta(self) -> dict[str, SignalMeta]:
        if self._signals_meta is None:
            meta_path = self.root / "signals_meta.json"
            if meta_path.exists():
                with meta_path.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
                self._signals_meta = {name: _decode_signal_meta(meta_dict) for name, meta_dict in payload.items()}
            else:
                if len(self._episodes) == 0:
                    self._signals_meta = {}
                else:
                    signals = self[0].signals
                    self._signals_meta = {name: sig.meta if len(sig) > 0 else None for name, sig in signals.items()}
        return self._signals_meta


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
        self._signals_meta: dict[str, SignalMeta] = self._load_signals_meta()

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

    @property
    def _signals_meta_path(self) -> Path:
        return self.root / "signals_meta.json"

    def _load_signals_meta(self) -> dict[str, SignalMeta]:
        path = self._signals_meta_path
        if not path.exists():
            return {}
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return {name: _decode_signal_meta(meta_dict) for name, meta_dict in payload.items()}

    def _save_signals_meta(self) -> None:
        path = self._signals_meta_path
        data = {name: _encode_signal_meta(meta) for name, meta in self._signals_meta.items()}
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def new_episode(self) -> DiskEpisodeWriter:
        eid = self._next_episode_id
        self._next_episode_id += 1  # Reserve id immediately

        block_dir = _ensure_block_dir(self.root, eid)
        # Do NOT create the episode directory here; DiskEpisodeWriter is
        # responsible for creating it and expects it to not exist yet.
        ep_dir = block_dir / f"{eid:012d}"

        writer = DiskEpisodeWriter(ep_dir, on_close=self._handle_episode_closed)
        return writer

    def __exit__(self, exc_type, exc, tb) -> None:
        self._save_signals_meta()

    def _handle_episode_closed(self, writer: DiskEpisodeWriter) -> None:
        episode = DiskEpisode(writer.path)
        for name, signal in episode.signals.items():
            try:
                meta = signal.meta
            except ValueError:  # The signal is empty, ignore
                continue
            if name not in self._signals_meta:
                self._signals_meta[name] = meta
        self._save_signals_meta()
