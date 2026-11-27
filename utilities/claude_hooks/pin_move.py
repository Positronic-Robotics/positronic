#!/usr/bin/env python3
"""Whether a pull request only moves a dependency pin, decided from its diff.

A pin move changes only three kinds of value:

- a quoted 40-hex revision, which must name a commit on the guarded branch;
- an exact requirement `"name==<version>"`, which must move to a version that a version field of
  the same pull request sets;
- a `version = "<version>"` field. In a lock file it too must move to such a version.

Each removed line pairs, in order, with an added line of the same block, and the two may differ
only in those values. A pull request that adds, deletes, renames or changes a binary file is not a
pin move. A pull request that moves no pin is not a pin move. Stdlib-only, as every hook in this
directory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import NamedTuple

# Field names of an entry of GitHub's `GET /repos/{owner}/{repo}/pulls/{number}/files`.
FILE_NAME = 'filename'
FILE_STATUS = 'status'
FILE_PATCH = 'patch'
MODIFIED = 'modified'

LOCK_FILES = frozenset({'uv.lock'})

_VERSION = r'\d[\w.+!-]*'
_VERSION_FIELD = re.compile(rf'^(\s*version\s*=\s*)(["\'])({_VERSION})\2(\s*,?\s*)$')
_EXACT_PIN = re.compile(rf'(["\'])([\w.\[\],-]+)==({_VERSION})\1')
_REVISION = re.compile(r'(["\'])([0-9a-f]{40})\1')


class _Values(NamedTuple):
    """A diff line with its pin values replaced by placeholders, and the values it carried."""

    shape: str
    versions: list[str]
    pins: list[str]
    revisions: list[str]


def _values(line: str) -> _Values:
    versions = []
    field = _VERSION_FIELD.match(line)
    if field:
        versions.append(field.group(3))
        line = f'{field.group(1)}{field.group(2)}<version>{field.group(2)}{field.group(4)}'
    pins = [m.group(3) for m in _EXACT_PIN.finditer(line)]
    line = _EXACT_PIN.sub(lambda m: f'{m.group(1)}{m.group(2)}==<pin>{m.group(1)}', line)
    revisions = [m.group(2) for m in _REVISION.finditer(line)]
    line = _REVISION.sub(r'\1<revision>\1', line)
    return _Values(line, versions, pins, revisions)


def _blocks(patch: str) -> Iterator[tuple[list[str], list[str]]]:
    """Each run of changed lines in a unified-diff patch, as its removed and its added lines."""
    removed: list[str] = []
    added: list[str] = []
    for line in [*patch.splitlines(), '']:
        if line.startswith('\\'):  # "\ No newline at end of file" belongs to the line before it
            continue
        if line.startswith('-'):
            removed.append(line[1:])
        elif line.startswith('+'):
            added.append(line[1:])
        else:
            if removed or added:
                yield removed, added
            removed, added = [], []


class _Moves(NamedTuple):
    versions: set[str]
    lock_versions: set[str]
    pins: set[str]
    revisions: set[str]


def _moves(files: list[dict]) -> _Moves | None:
    """The values that the files move, or None when a change is more than a moved value."""
    moves = _Moves(set(), set(), set(), set())
    for entry in files:
        patch = entry.get(FILE_PATCH)
        if entry.get(FILE_STATUS) != MODIFIED or not isinstance(patch, str):
            return None
        is_lock = str(entry.get(FILE_NAME) or '').rpartition('/')[2] in LOCK_FILES
        for removed, added in _blocks(patch):
            if len(removed) != len(added):
                return None
            for old, new in zip(map(_values, removed), map(_values, added), strict=True):
                if old.shape != new.shape:
                    return None
                (moves.lock_versions if is_lock else moves.versions).update(set(new.versions) - set(old.versions))
                moves.pins.update(n for o, n in zip(old.pins, new.pins, strict=True) if o != n)
                moves.revisions.update(n for o, n in zip(old.revisions, new.revisions, strict=True) if o != n)
    return moves


def is_pin_move(files: list[dict], on_guarded_branch: Callable[[str], bool]) -> bool:
    """Whether `files`, the changed files of one pull request, only move pins.

    `on_guarded_branch` answers whether a revision names a commit on the guarded branch.
    """
    moves = _moves(files)
    if moves is None or not (moves.pins or moves.revisions):
        return False
    if not (moves.pins | moves.lock_versions) <= moves.versions:
        return False
    return all(on_guarded_branch(revision) for revision in moves.revisions)
