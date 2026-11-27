#!/usr/bin/env python3
"""Whether a pull request only moves a dependency pin, decided from its diff.

A pin move changes only three kinds of value:

- a quoted 40-hex revision, in any file, which must name a commit on the guarded branch;
- the `version` field of a package in a `pyproject.toml`;
- an exact `"name==<version>"` pin in a `pyproject.toml`, or the `version` of a package in
  `uv.lock`. Each must move to the version that the same pull request gives that package.

A package is the `name` line that the diff shows above a version field, in the same table. Each
removed line pairs, in order, with an added line of the same block, and the two may differ only in
those values. A pull request that adds, deletes, renames or changes a binary file is not a pin
move. A pull request that moves no pin is not a pin move. Stdlib-only, as every hook in this
directory.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from typing import NamedTuple

# Field names of an entry of the `files` list that GitHub's pull request and compare reads return.
FILE_NAME = 'filename'
FILE_STATUS = 'status'
FILE_PATCH = 'patch'
MODIFIED = 'modified'


class FileKind(enum.Enum):
    PROJECT = 'pyproject.toml'
    LOCK = 'uv.lock'
    OTHER = ''


_VERSION = r'\d[\w.+!-]*'
_VERSION_FIELD = re.compile(rf'^(\s*version\s*=\s*)(["\'])({_VERSION})\2(\s*,?\s*)$')
_EXACT_PIN = re.compile(rf'(["\'])([\w.-]+)(\[[\w,.-]*\])?==({_VERSION})\1')
_REVISION = re.compile(r'(["\'])([0-9a-f]{40})\1')
_PACKAGE_NAME = re.compile(r'^\s*name\s*=\s*(["\'])([\w.-]+)\1\s*$')
_TABLE = re.compile(r'^\s*\[')
# Stand-ins for a moved value in the shape of a line. A line that holds one already is not a pin move,
# since its own text could then match the shape of another line.
_VERSION_MARK = '<version>'
_PIN_MARK = '<pin>'
_REVISION_MARK = '<revision>'
_MARKS = (_VERSION_MARK, _PIN_MARK, _REVISION_MARK)


def package_key(name: str) -> str:
    """The normalized form of a distribution name, so that `Foo_Bar` and `foo-bar` match."""
    return re.sub(r'[-_.]+', '-', name).lower()


class _Values(NamedTuple):
    """A diff line with its pin values replaced by placeholders, and the values it carried."""

    shape: str
    version: str
    pins: list[tuple[str, str]]
    revisions: list[str]


def _values(line: str, kind: FileKind) -> _Values:
    version = ''
    field = _VERSION_FIELD.match(line) if kind is not FileKind.OTHER else None
    if field:
        version = field.group(3)
        line = f'{field.group(1)}{field.group(2)}{_VERSION_MARK}{field.group(2)}{field.group(4)}'
    pins = []
    if kind is FileKind.PROJECT:
        pins = [(package_key(m.group(2)), m.group(4)) for m in _EXACT_PIN.finditer(line)]
        line = _EXACT_PIN.sub(lambda m: f'{m.group(1)}{m.group(2)}{m.group(3) or ""}=={_PIN_MARK}{m.group(1)}', line)
    revisions = [m.group(2) for m in _REVISION.finditer(line)]
    line = _REVISION.sub(rf'\1{_REVISION_MARK}\1', line)
    return _Values(line, version, pins, revisions)


class Moves(NamedTuple):
    """The values that a diff moves, each version beside the package it belongs to."""

    versions: set[tuple[str, str]]
    lock_versions: set[tuple[str, str]]
    pins: set[tuple[str, str]]
    revisions: set[str]


def _add_block(removed: list[str], added: list[str], kind: FileKind, package: str, moves: Moves) -> bool:
    """Record the values one block moves. False when the block changes more than a value."""
    if len(removed) != len(added) or any(mark in line for line in (*removed, *added) for mark in _MARKS):
        return False
    for old, new in zip((_values(r, kind) for r in removed), (_values(a, kind) for a in added), strict=True):
        if old.shape != new.shape:
            return False
        if new.version != old.version:
            if not package:
                return False
            (moves.lock_versions if kind is FileKind.LOCK else moves.versions).add((package, new.version))
        moves.pins.update(n for o, n in zip(old.pins, new.pins, strict=True) if o != n)
        moves.revisions.update(n for o, n in zip(old.revisions, new.revisions, strict=True) if o != n)
    return True


def _add_file(patch: str, kind: FileKind, moves: Moves) -> bool:
    """Record the values one file's patch moves. False when it changes more than a value."""
    removed: list[str] = []
    added: list[str] = []
    package = ''
    for line in [*patch.splitlines(), '']:
        if line.startswith('\\'):  # "\ No newline at end of file" belongs to the line before it
            continue
        if line.startswith('-'):
            removed.append(line[1:])
            continue
        if line.startswith('+'):
            added.append(line[1:])
            continue
        if (removed or added) and not _add_block(removed, added, kind, package, moves):
            return False
        removed, added = [], []
        text = line[1:]
        if line.startswith('@@') or _TABLE.match(text):
            package = ''
        elif name := _PACKAGE_NAME.match(text):
            package = package_key(name.group(2))
    return True


def moves_of(files: list[dict]) -> Moves | None:
    """The values that the files move, or None when a change is more than a moved value."""
    moves = Moves(set(), set(), set(), set())
    for entry in files:
        patch = entry.get(FILE_PATCH)
        if entry.get(FILE_STATUS) != MODIFIED or not isinstance(patch, str):
            return None
        base_name = str(entry.get(FILE_NAME) or '').rpartition('/')[2]
        kind = next((k for k in FileKind if k.value == base_name), FileKind.OTHER)
        if not _add_file(patch, kind, moves):
            return None
    return moves


def is_pin_move(files: list[dict], on_guarded_branch: Callable[[str], bool]) -> bool:
    """Whether `files`, the changed files of one pull request, only move pins.

    `on_guarded_branch` answers whether a revision names a commit on the guarded branch.
    """
    moves = moves_of(files)
    if moves is None or not (moves.pins or moves.revisions):
        return False
    if not (moves.pins | moves.lock_versions) <= moves.versions:
        return False
    return all(on_guarded_branch(revision) for revision in moves.revisions)
