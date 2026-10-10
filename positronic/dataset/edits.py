"""Edit logs over immutable recordings.

Episodes are never modified after recording. Post-hoc facts — operator verdicts, analysis scores — are *edits*:
declarative records appended to an `edits.jsonl` file in the episode's own directory and applied as a view on read.
`EditedDataset` is that view and the handle that amends it: it reads the logs and applies the edits, and its methods
append new ones.

Each line is one JSON record carrying its op and version, so a log replays forever. The ops, targeting episodes by
`meta['uid']`:

- `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` — merge static items over the episode's recorded ones
  (log order, last write per key wins).
- `{"op": "drop", "v": 1, "ep": "<uid>"}` — remove the episode from the view; the recording stays on disk.
- `{"op": "undrop", "v": 1, "ep": "<uid>"}` — restore a dropped episode (the last drop/undrop per episode wins).

A dataset directory can hold an `edits.jsonl` with records for any episode under it. That log is read and never
written. It applies to the episodes under its directory, before their own logs.

The format is plain appendable JSON so external tools can write it; an episode directory has a single writer.
"""

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from .dataset import Dataset, FilterDataset
from .episode import META_PATH, META_UID, Episode, _is_valid_static_value, _static_decode_hook, _StaticEncoder
from .signal import Signal

EDITS_FILE = 'edits.jsonl'

# One edit log's records by episode uid, each uid's in log order.
_Log = dict[str, list[dict[str, Any]]]


def _read_log(edits_path: Path) -> _Log:
    """The records of the edit log at `edits_path`. A log that does not exist holds none."""
    records: _Log = {}
    if not edits_path.exists():
        return records
    with edits_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            try:
                record = json.loads(line, object_hook=_static_decode_hook)
            except json.JSONDecodeError as e:
                raise ValueError(f'Corrupt edit record at {edits_path}:{line_no}') from e
            match record:
                case {'op': 'set_static', 'v': 1, 'ep': str(), 'data': dict(data)}:
                    if not _is_valid_static_value(data):
                        raise ValueError(
                            f'Invalid static values in edit record at {edits_path}:{line_no}: {line.strip()}'
                        )
                case {'op': 'drop' | 'undrop', 'v': 1, 'ep': str()}:
                    pass
                case _:
                    raise ValueError(f'Unsupported edit record at {edits_path}:{line_no}: {line.strip()}')
            records.setdefault(record['ep'], []).append(record)
    return records


def _apply_records(records: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """The static edits `records` merge, the last write per key winning, and whether the last drop/undrop drops."""
    statics: dict[str, Any] = {}
    dropped = False
    for record in records:
        if record['op'] == 'set_static':
            statics.update(record['data'])
        else:
            dropped = record['op'] == 'drop'
    return statics, dropped


class _EpisodeKey(NamedTuple):
    """An episode's identity, and the directory that holds its own edit log where it has one."""

    uid: str
    directory: Path | None


def _episode_key(episode: Episode) -> _EpisodeKey:
    meta = episode.meta
    path = meta.get(META_PATH)
    return _EpisodeKey(meta[META_UID], Path(path) if path is not None else None)


def _validate_uid(uid: str) -> None:
    if not (isinstance(uid, str) and uid):
        raise ValueError(f'Episode uid must be a non-empty string, got {uid!r}')


class EditedEpisode(Episode):
    """View of an episode with edited static items merged over the recorded ones.

    A static edit that shadows a recorded signal is ambiguous; it raises when that key is *read*, not at
    construction — so identity (`meta`/`uid`) stays readable for filtering and dropping even when an edit collides.
    """

    def __init__(self, episode: Episode, edits: dict[str, Any]) -> None:
        self._episode = episode
        self._edits = edits

    def __iter__(self) -> Iterator[str]:
        yield from self._episode
        yield from (k for k in self._edits if k not in self._episode)

    def __len__(self) -> int:
        return len(self._episode) + sum(1 for k in self._edits if k not in self._episode)

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name in self._edits:
            if name in self._episode and isinstance(self._episode[name], Signal):
                raise ValueError(f'Edited static item {name!r} collides with a signal in {self._episode!r}')
            return self._edits[name]
        return self._episode[name]

    @property
    def meta(self) -> dict:
        return self._episode.meta


class EditedDataset(Dataset):
    """A dataset with its edit logs applied as a view, and the handle that amends the logs.

    Reads are curated: `len()`/indexing apply the edit logs, so consumers see the corrected dataset. The same object
    amends them — its edit methods (`set_static`/`drop`/`undrop`) append a record to the own log of every episode
    with that uid and return a *new* `EditedDataset` over the same recordings, so a held reference keeps its shape
    while the returned one reflects the edit.

    An episode's own log is in the directory its `meta['path']` names. The log in each of `dataset_log_dirs` applies,
    in that order, to the episodes under that directory, and to every episode that has no directory. Each episode's
    own log applies after them.
    """

    def __init__(self, base: Dataset, dataset_log_dirs: Sequence[Path] = ()):
        self._base = base
        self._dataset_log_dirs = tuple(Path(directory).expanduser().resolve() for directory in dataset_log_dirs)
        self._episode_dirs: dict[str, set[Path]] = {}
        self._statics: dict[_EpisodeKey, dict[str, Any]] = {}
        self._dropped: set[_EpisodeKey] = set()
        logs: dict[Path, _Log] = {}
        for key in map(_episode_key, base):
            if key.directory is not None:
                self._episode_dirs.setdefault(key.uid, set()).add(key.directory)
            records = (record for log in self._logs_of(key, logs) for record in log.get(key.uid, ()))
            statics, dropped = _apply_records(records)
            if statics:
                self._statics[key] = statics
            if dropped:
                self._dropped.add(key)
        self._kept = FilterDataset(base, lambda ep: _episode_key(ep) not in self._dropped)

    def _logs_of(self, key: _EpisodeKey, logs: dict[Path, _Log]) -> Iterator[_Log]:
        """The logs that apply to the episode `key`, in order, each read once into `logs`."""
        directories = [d for d in self._dataset_log_dirs if key.directory is None or key.directory.is_relative_to(d)]
        if key.directory is not None:
            directories.append(key.directory)
        for directory in directories:
            path = directory / EDITS_FILE
            if path not in logs:
                logs[path] = _read_log(path)
            yield logs[path]

    def __len__(self) -> int:
        return len(self._kept)

    def _get_episode(self, index: int) -> Episode:
        return self.overlay(self._kept[index])

    def overlay(self, episode: Episode) -> Episode:
        """Apply these logs' edits to an episode of the underlying recordings.

        A dropped episode is returned untouched — its edits are irrelevant to the curated view, and skipping the
        overlay lets the editor navigate to a dropped row (to undrop it) even when an edit collides with a signal.
        """
        key = _episode_key(episode)
        if key in self._dropped:
            return episode
        edits = self._statics.get(key)
        return EditedEpisode(episode, edits) if edits else episode

    @property
    def base(self) -> Dataset:
        """The underlying recordings, before edits are applied."""
        return self._base

    # TODO(#433): `dropped` (and `overlay` above) are public only for an episode editor's raw-indexed navigation
    # over all episodes, dropped ones included. The review-surface split in #433 should own that view and drop
    # these from the public edit API.
    @property
    def dropped(self) -> frozenset[str]:
        """Uids of the episodes the logs currently drop from the view."""
        return frozenset(key.uid for key in self._dropped)

    @property
    def meta(self) -> dict[str, Any]:
        return self._base.meta

    def set_static(self, uid: str, data: dict[str, Any]) -> 'EditedDataset':
        """Append a `set_static` edit merging `data` over the episode's recorded static items."""
        _validate_uid(uid)
        if not (isinstance(data, dict) and _is_valid_static_value(data)):
            raise ValueError(f'Edit data must be a mapping of static items to JSON-serializable values\n{data=!r}')
        return self._append('set_static', uid, data=data)

    def drop(self, uid: str) -> 'EditedDataset':
        """Append a `drop` edit removing the episode from the view; the recording stays on disk."""
        _validate_uid(uid)
        return self._append('drop', uid)

    def undrop(self, uid: str) -> 'EditedDataset':
        """Append an `undrop` edit restoring a previously dropped episode to the view."""
        _validate_uid(uid)
        return self._append('undrop', uid)

    def _append(self, op: str, uid: str, **fields: Any) -> 'EditedDataset':
        directories = self._episode_dirs.get(uid)
        if not directories:
            raise ValueError(f'No episode with uid {uid!r} and a directory for its edit log')
        line = json.dumps({'op': op, 'v': 1, 'ep': uid, **fields}, cls=_StaticEncoder) + '\n'
        for directory in sorted(directories):
            with (directory / EDITS_FILE).open('a', encoding='utf-8') as f:
                f.write(line)
        return EditedDataset(self._base, self._dataset_log_dirs)
