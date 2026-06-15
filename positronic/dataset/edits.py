"""Edit log over immutable recordings.

Episodes are never modified after recording. Post-hoc facts — operator verdicts, analysis scores — are *edits*:
declarative records appended to an `edits.jsonl` file beside a dataset and applied as a view on read. `EditedDataset`
is that view and the handle that amends it: it reads the log and applies the edits, and its methods append new ones.

Each line is one JSON record carrying its op and version, so a log replays forever. The ops, targeting episodes by
`meta['uid']`:

- `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` — merge static items over the episode's recorded ones
  (log order, last write per key wins).
- `{"op": "drop", "v": 1, "ep": "<uid>"}` — remove the episode from the view; the recording stays on disk.
- `{"op": "undrop", "v": 1, "ep": "<uid>"}` — restore a dropped episode (the last drop/undrop per episode wins).

The format is plain appendable JSON so external tools can write it; the directory has a single writer.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .dataset import Dataset, FilterDataset
from .episode import Episode, _is_valid_static_value, _static_decode_hook, _StaticEncoder
from .signal import Signal

EDITS_FILE = 'edits.jsonl'


def _load_edits(edits_dir: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Read the edit log and return merged static edits per episode uid plus the set of dropped uids.

    Records apply in log order: the last write per static key wins, and the last drop/undrop per episode wins.
    """
    edits_path = Path(edits_dir) / EDITS_FILE
    statics: dict[str, dict[str, Any]] = {}
    dropped: set[str] = set()
    if not edits_path.exists():
        return statics, dropped
    with edits_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            try:
                record = json.loads(line, object_hook=_static_decode_hook)
            except json.JSONDecodeError as e:
                raise ValueError(f'Corrupt edit record at {edits_path}:{line_no}') from e
            match record:
                case {'op': 'set_static', 'v': 1, 'ep': str(ep_uid), 'data': dict(data)}:
                    if not _is_valid_static_value(data):
                        raise ValueError(
                            f'Invalid static values in edit record at {edits_path}:{line_no}: {line.strip()}'
                        )
                    statics.setdefault(ep_uid, {}).update(data)
                case {'op': 'drop', 'v': 1, 'ep': str(ep_uid)}:
                    dropped.add(ep_uid)
                case {'op': 'undrop', 'v': 1, 'ep': str(ep_uid)}:
                    dropped.discard(ep_uid)
                case _:
                    raise ValueError(f'Unsupported edit record at {edits_path}:{line_no}: {line.strip()}')
    return statics, dropped


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
    """A dataset with its edit log applied as a view, and the handle that amends the log.

    Reads are curated: `len()`/indexing apply the edit log, so consumers see the corrected dataset. The same object
    amends the log — its edit methods (`set_static`/`drop`/`undrop`) append a record to `edits_dir` and return a *new*
    `EditedDataset` over the same recordings, so a held reference keeps its shape while the returned one reflects the
    edit. `edits_dir` is where the log lives; it can equal the recordings directory.
    """

    def __init__(self, base: Dataset, edits_dir: Path):
        self._base = base
        self._edits_dir = Path(edits_dir)
        self._statics, self._dropped = _load_edits(self._edits_dir)
        self._kept = FilterDataset(base, lambda ep: ep.meta['uid'] not in self._dropped)

    def __len__(self) -> int:
        return len(self._kept)

    def _get_episode(self, index: int) -> Episode:
        return self.overlay(self._kept[index])

    def overlay(self, episode: Episode) -> Episode:
        """Apply this log's edits to an episode of the underlying recordings, addressed by uid.

        A dropped episode is returned untouched — its edits are irrelevant to the curated view, and skipping the
        overlay lets the editor navigate to a dropped row (to undrop it) even when an edit collides with a signal.
        """
        uid = episode.meta.get('uid')
        if uid in self._dropped:
            return episode
        edits = self._statics.get(uid)
        return EditedEpisode(episode, edits) if edits else episode

    @property
    def base(self) -> Dataset:
        """The underlying recordings, before edits are applied."""
        return self._base

    # TODO(#433): `dropped` (and `overlay` above) are public only for the EvalUI editor's raw-indexed navigation
    # over all episodes, dropped ones included. The review-surface split in #433 should own that view and drop
    # these from the public edit API.
    @property
    def dropped(self) -> frozenset[str]:
        """Uids the log currently drops from the view."""
        return frozenset(self._dropped)

    @property
    def meta(self) -> dict[str, Any]:
        return self._base.meta

    def set_static(self, uid: str, data: dict[str, Any]) -> 'EditedDataset':
        """Append a `set_static` edit merging `data` over the episode's recorded static items."""
        _validate_uid(uid)
        if not (isinstance(data, dict) and _is_valid_static_value(data)):
            raise ValueError(f'Edit data must be a mapping of static items to JSON-serializable values\n{data=!r}')
        return self._append({'op': 'set_static', 'v': 1, 'ep': uid, 'data': data})

    def drop(self, uid: str) -> 'EditedDataset':
        """Append a `drop` edit removing the episode from the view; the recording stays on disk."""
        _validate_uid(uid)
        return self._append({'op': 'drop', 'v': 1, 'ep': uid})

    def undrop(self, uid: str) -> 'EditedDataset':
        """Append an `undrop` edit restoring a previously dropped episode to the view."""
        _validate_uid(uid)
        return self._append({'op': 'undrop', 'v': 1, 'ep': uid})

    def _append(self, record: dict[str, Any]) -> 'EditedDataset':
        with (self._edits_dir / EDITS_FILE).open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, cls=_StaticEncoder) + '\n')
        return EditedDataset(self._base, self._edits_dir)
