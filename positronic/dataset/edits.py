"""Edit log over immutable recordings.

Episodes are never modified after recording. Post-hoc facts — operator verdicts, analysis scores — are *edits*:
declarative records appended to an `edits.jsonl` file in the dataset directory and applied as a view on read.
Each line is one JSON record carrying its op: `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` merges
static items over the episode's recorded ones (log order, last write per key wins);
`{"op": "drop", "v": 1, "ep": "<uid>"}` removes the episode from the loaded view and
`{"op": "undrop", "v": 1, "ep": "<uid>"}` restores it (the last drop/undrop wins). Records target episodes by
`meta['uid']`. The format is plain appendable JSON so external tools can write it; the dataset directory assumes
a single writer.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .dataset import Dataset
from .episode import Episode, _is_valid_static_value, _static_decode_hook, _StaticEncoder
from .signal import Signal

EDITS_FILE = 'edits.jsonl'


def set_static(root: Path, uid: str, data: dict[str, Any]) -> None:
    """Append a `set_static` edit for the episode with the given uid.

    The edit applies when the dataset is loaded: the given items are merged over the episode's recorded
    static items. The episode recording itself is never modified.

    Args:
        root: The dataset directory holding the edit log
        uid: The target episode's `meta['uid']`
        data: Static items to set; values follow the same restrictions as `EpisodeWriter.set_static`
    """
    _validate_uid(uid)
    if not (isinstance(data, dict) and _is_valid_static_value(data)):
        raise ValueError(f'Edit data must be a mapping of static items to JSON-serializable values\n{data=!r}')
    _append_record(root, {'op': 'set_static', 'v': 1, 'ep': uid, 'data': data})


def drop(root: Path, uid: str) -> None:
    """Append a `drop` edit removing the episode with the given uid from the loaded view.

    The episode recording stays on disk; it just stops appearing in datasets opened through `load_dataset`.

    Args:
        root: The dataset directory holding the edit log
        uid: The target episode's `meta['uid']`
    """
    _validate_uid(uid)
    _append_record(root, {'op': 'drop', 'v': 1, 'ep': uid})


def undrop(root: Path, uid: str) -> None:
    """Append an `undrop` edit restoring a previously dropped episode to the loaded view.

    Args:
        root: The dataset directory holding the edit log
        uid: The target episode's `meta['uid']`
    """
    _validate_uid(uid)
    _append_record(root, {'op': 'undrop', 'v': 1, 'ep': uid})


def _validate_uid(uid: str) -> None:
    if not (isinstance(uid, str) and uid):
        raise ValueError(f'Episode uid must be a non-empty string, got {uid!r}')


def _append_record(root: Path, record: dict[str, Any]) -> None:
    with (root / EDITS_FILE).open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, cls=_StaticEncoder) + '\n')


def load_edits(root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Read the edit log and return merged static edits per episode uid plus the set of dropped uids.

    Records apply in log order: the last write per static key wins, and the last drop/undrop per episode wins.
    """
    edits_path = root / EDITS_FILE
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


class EditedEpisode(Episode):
    """View of an episode with edited static items merged over the recorded ones."""

    def __init__(self, episode: Episode, edits: dict[str, Any]) -> None:
        collisions = [k for k in edits if k in episode and isinstance(episode[k], Signal)]
        if collisions:
            raise ValueError(f'Edited static items {sorted(collisions)} collide with signals in {episode!r}')
        self._episode = episode
        self._edits = edits

    def __iter__(self) -> Iterator[str]:
        yield from self._episode
        yield from (k for k in self._edits if k not in self._episode)

    def __len__(self) -> int:
        return len(self._episode) + sum(1 for k in self._edits if k not in self._episode)

    def __getitem__(self, name: str) -> Signal[Any] | Any:
        if name in self._edits:
            return self._edits[name]
        return self._episode[name]

    @property
    def meta(self) -> dict:
        return self._episode.meta


class EditedDataset(Dataset):
    """View of a dataset with edits applied: per-episode static edits keyed by episode uid."""

    def __init__(self, dataset: Dataset, edits: dict[str, dict[str, Any]]):
        self._dataset = dataset
        self._edits = edits

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        episode = self._dataset[index]
        edits = self._edits.get(episode.meta.get('uid'))
        return EditedEpisode(episode, edits) if edits else episode

    @property
    def meta(self) -> dict[str, Any]:
        return self._dataset.meta
