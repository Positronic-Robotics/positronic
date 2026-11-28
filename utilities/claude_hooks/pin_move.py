#!/usr/bin/env python3
"""Whether a pull request only moves a dependency pin, decided from its diff.

A pin move changes only three kinds of value:

- a quoted 40-hex revision, in any file, which must name a commit on the guarded branch;
- `project.version` of a `pyproject.toml`;
- an exact `"name==<version>"` pin in `[project]`, `[project.optional-dependencies]` or
  `[dependency-groups]` of a `pyproject.toml`, or the `version` of a `[[package]]` in `uv.lock`.
  Each must move to the version that the same pull request gives that package.

The table of a changed `pyproject.toml` line comes from the file at the head commit, and the package
is its `project.name`. In `uv.lock`, a version counts only on the line after `[[package]]` and
`name`, the layout that uv writes. Each removed line pairs, in order, with an added line of the
same block, and the two may differ only in those values. A pull request that adds, deletes,
renames or changes a binary file is not a pin move. A pull request that moves no pin is not a pin
move. Stdlib-only, as every hook in this directory.
"""

from __future__ import annotations

import enum
import re
import tomllib
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


PROJECT_TABLE = 'project'
PIN_TABLES = frozenset({PROJECT_TABLE, 'project.optional-dependencies', 'dependency-groups'})
LOCK_PACKAGE_HEADER = '[[package]]'

_VERSION = r'\d[\w.+!-]*'
_VERSION_FIELD = re.compile(rf'^(\s*version\s*=\s*)(["\'])({_VERSION})\2(\s*,?\s*)$')
_EXACT_PIN = re.compile(rf'(["\'])([\w.-]+)(\[[\w,.-]*\])?==({_VERSION})\1')
_REVISION = re.compile(r'(["\'])([0-9a-f]{40})\1')
_PACKAGE_NAME = re.compile(r'^name\s*=\s*"([\w.-]+)"$')
_TABLE_HEADER = re.compile(r'^\s*\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$')
_HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')
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


class _Place(NamedTuple):
    """Where an added line sits: its table, and the package whose version it may carry."""

    table: str
    package: str


class _Project(NamedTuple):
    """A `pyproject.toml` at the head commit: its `project.name`, and the table of each line."""

    package: str
    tables: list[str]

    def place(self, line_number: int) -> _Place:
        table = self.tables[line_number - 1] if 0 < line_number <= len(self.tables) else ''
        return _Place(table, self.package)


def _read_project(text: str) -> _Project | None:
    """The package and the line tables of a `pyproject.toml`, or None when it names no package."""
    try:
        name = tomllib.loads(text).get(PROJECT_TABLE, {}).get('name')
    except (tomllib.TOMLDecodeError, AttributeError):
        return None
    if not isinstance(name, str):
        return None
    tables, table = [], ''
    for line in text.splitlines():
        if header := _TABLE_HEADER.match(line):
            table = re.sub(r'["\'\s]', '', header.group(1))
        tables.append(table)
    return _Project(package_key(name), tables)


def _add_pair(old: _Values, new: _Values, kind: FileKind, place: _Place, moves: Moves) -> bool:
    """Record the values one changed line moves. False when it changes more than a value in its place."""
    if old.shape != new.shape:
        return False
    if new.version != old.version:
        wanted = PROJECT_TABLE if kind is FileKind.PROJECT else LOCK_PACKAGE_HEADER
        if place.table != wanted or not place.package:
            return False
        (moves.lock_versions if kind is FileKind.LOCK else moves.versions).add((place.package, new.version))
    moved_pins = {n for o, n in zip(old.pins, new.pins, strict=True) if o != n}
    if moved_pins and place.table not in PIN_TABLES:
        return False
    moves.pins.update(moved_pins)
    moves.revisions.update(n for o, n in zip(old.revisions, new.revisions, strict=True) if o != n)
    return True


def _add_block(removed: list[str], added: list[str], kind: FileKind, places: list[_Place], moves: Moves) -> bool:
    """Record the values one block moves. False when the block changes more than a value."""
    if len(removed) != len(added) or any(mark in line for line in (*removed, *added) for mark in _MARKS):
        return False
    pairs = zip((_values(r, kind) for r in removed), (_values(a, kind) for a in added), places, strict=True)
    return all(_add_pair(old, new, kind, place, moves) for old, new, place in pairs)


def _lock_place(above: list[str], index: int) -> _Place:
    """The place of a `uv.lock` line: a version counts only directly under `[[package]]` and its `name`."""
    name = _PACKAGE_NAME.match(above[-1]) if index == 0 and len(above) >= 2 else None
    if name and above[-2] == LOCK_PACKAGE_HEADER:
        return _Place(LOCK_PACKAGE_HEADER, package_key(name.group(1)))
    return _Place('', '')


def _add_file(patch: str, kind: FileKind, project: _Project | None, moves: Moves) -> bool:
    """Record the values one file's patch moves. False when it changes more than a value."""
    removed: list[str] = []
    added: list[str] = []
    places: list[_Place] = []
    above: list[str] = []  # the unchanged lines of this hunk so far
    line_number = 0  # the line number in the head file of the next unchanged or added line
    for line in [*patch.splitlines(), '']:
        if line.startswith('\\'):  # "\ No newline at end of file" belongs to the line before it
            continue
        if line.startswith('-'):
            removed.append(line[1:])
            continue
        if line.startswith('+'):
            if project is not None:
                places.append(project.place(line_number))
            elif kind is FileKind.LOCK:
                places.append(_lock_place(above, len(added)))
            else:
                places.append(_Place('', ''))
            added.append(line[1:])
            line_number += 1
            continue
        if (removed or added) and not _add_block(removed, added, kind, places, moves):
            return False
        removed, added, places = [], [], []
        if hunk := _HUNK_HEADER.match(line):
            line_number, above = int(hunk.group(1)), []
        else:
            above.append(line[1:])
            line_number += 1
    return True


def moves_of(files: list[dict], read_file: Callable[[str], str]) -> Moves | None:
    """The values that the files move, or None when a change is more than a moved value.

    `read_file` returns the text of a changed file at the head commit.
    """
    moves = Moves(set(), set(), set(), set())
    for entry in files:
        patch = entry.get(FILE_PATCH)
        if entry.get(FILE_STATUS) != MODIFIED or not isinstance(patch, str):
            return None
        path = str(entry.get(FILE_NAME) or '')
        kind = next((k for k in FileKind if k.value == path.rpartition('/')[2]), FileKind.OTHER)
        project = _read_project(read_file(path)) if kind is FileKind.PROJECT else None
        if (kind is FileKind.PROJECT and project is None) or not _add_file(patch, kind, project, moves):
            return None
    return moves


def is_pin_move(files: list[dict], on_guarded_branch: Callable[[str], bool], read_file: Callable[[str], str]) -> bool:
    """Whether `files`, the changed files of one pull request, only move pins.

    `on_guarded_branch` answers whether a revision names a commit on the guarded branch, and
    `read_file` returns the text of a changed file at the head commit.
    """
    moves = moves_of(files, read_file)
    if moves is None or not (moves.pins or moves.revisions):
        return False
    if not (moves.pins | moves.lock_versions) <= moves.versions:
        return False
    return all(on_guarded_branch(revision) for revision in moves.revisions)
