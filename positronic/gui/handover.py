"""The file contract between the tooling that assigns a rollout batch and the operator console.

The launcher writes ``assignment.yaml`` into a handover directory before starting a run; the console
writes ``done-<batch_id>.yaml`` back into the same directory when the operator hands the batch in. Both
are YAML mappings, versioned by ``schema_version`` on the assignment::

    # assignment.yaml
    schema_version: 1
    batch_id: 2026-08-05-a
    task: Pick up objects from the red tote and place them in the green tote.
    episode_target: 20
    notes: tote on the left                 # optional
    created_at: 2026-08-05T09:14:03+00:00

    # done-2026-08-05-a.yaml
    batch_id: 2026-08-05-a
    finished_at: 2026-08-05T11:02:41+00:00
    operator_note: gripper slipped twice    # optional

An assignment names neither endpoint nor policy: the operator records and scores the batch without
knowing what is under test.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

SCHEMA_VERSION = 1
DEFAULT_DIR = Path('/home/rollout/handover')
ASSIGNMENT_NAME = 'assignment.yaml'


@dataclass(frozen=True)
class Assignment:
    """A batch of episodes to record, as the launcher described it.

    A ``schema_version`` other than ``SCHEMA_VERSION`` leaves every other field empty: the file follows a
    contract this console does not know, and the version is all it can report about it.
    """

    schema_version: int
    batch_id: str = ''
    task: str = ''
    episode_target: int = 0
    notes: str = ''
    created_at: str = ''

    @property
    def supported(self) -> bool:
        return self.schema_version == SCHEMA_VERSION


def read_assignment(handover_dir: Path) -> Assignment | None:
    """The assignment in ``handover_dir``, or None when it holds none."""
    path = handover_dir / ASSIGNMENT_NAME
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    if raw['schema_version'] != SCHEMA_VERSION:
        return Assignment(schema_version=raw['schema_version'])
    return Assignment(
        schema_version=SCHEMA_VERSION,
        batch_id=raw['batch_id'],
        task=raw['task'],
        episode_target=raw['episode_target'],
        notes=raw.get('notes', ''),
        created_at=raw['created_at'],
    )


def completion_path(handover_dir: Path, batch_id: str) -> Path:
    return handover_dir / f'done-{batch_id}.yaml'


def write_completion(handover_dir: Path, batch_id: str, operator_note: str = '') -> Path:
    """Mark the batch handed back, returning the marker path."""
    payload = {'batch_id': batch_id, 'finished_at': datetime.now(UTC).isoformat()}
    if operator_note:
        payload['operator_note'] = operator_note
    path = completion_path(handover_dir, batch_id)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path
