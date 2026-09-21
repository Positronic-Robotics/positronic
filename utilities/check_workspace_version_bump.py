"""Keep each published workspace member, its version, and the root's pin on it in step.

A change under a member's directory must raise its `pyproject.toml` version, and the root's `==` pin on
the member must name that version: the release publishes each member with `skip-existing`, so an
unbumped member keeps the old wheel on the index and the root then ships depending on it. Judged
against `--base` (else `$RATCHET_BASE`, else `origin/main`) at the merge-base; an unresolvable base or
a base with no manifest for the member skips with a note, and a present but unreadable manifest fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_REV_ENV = 'RATCHET_BASE'

ROOT_MANIFEST = 'pyproject.toml'


class Member(NamedTuple):
    """A workspace member the root pins: the directory it ships from, and the name the root pins it by."""

    directory: str
    distribution: str
    # The package the root imports from it, named in a failure so the reader knows what a missing pin costs.
    package: str

    @property
    def manifest(self) -> str:
        return f'{self.directory}/pyproject.toml'


MEMBERS = (
    Member('client', 'positronic-platform-client', 'platform_client'),
    Member('wire', 'positronic-wire', 'positronic_wire'),
)

# Changes that cannot reach the installed wheel. A test is NOT here: it ships inside the package
# directory, and a reader comparing two revisions may expect the same code behind the same version.
EXEMPT_SUFFIXES = ('.md',)


def run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_base_ref(arg: str | None, env: str | None) -> str:
    """Pick the base ref: an explicit `--base` wins, then `$RATCHET_BASE`, else `origin/main`."""
    return arg or env or 'origin/main'


def resolve_merge_base(ref: str) -> str | None:
    """The commit this change is judged from, or None where the ref does not resolve at all."""
    merge_base = run_git('merge-base', 'HEAD', ref)
    if merge_base:
        return merge_base
    # No merge-base (shallow history, an unrelated ref): use the ref itself where it resolves.
    return run_git('rev-parse', ref) or None


def is_later_version(was: str, is_now: str, manifest: str = 'the manifest') -> bool:
    """Whether `is_now` is a LATER version than `was`, by the ordering the index will use.

    PEP 440 through `packaging`, not a hand-rolled tuple: the index resolves these versions by that
    ordering, so a gate that ranks them any other way guards something other than what ships. It is
    the pre-releases a hand-rolled comparison gets backwards — `1.0rc1` precedes `1.0`, and `1.0rc10`
    follows `1.0rc2` — and it is also what makes `1.0` and `1.0.0` compare EQUAL, so a padded zero is
    not read as a bump and cannot pass off a re-release as its successor.
    """
    try:
        return Version(is_now) > Version(was)
    except InvalidVersion as exc:
        # Fails closed, and says which value the index would have refused too.
        raise SystemExit(f'ERROR - {manifest} version is not PEP 440: {exc}') from exc


def declared_version(text: str, *, guarded: str | None = None) -> str | None:
    """The `version` a pyproject declares under `[project]`, or None where it declares none.

    Parsed out of TOML rather than scanned for as text, like the pin below: to a scan a `version`
    line under any other table — a tool's own section — reads as the project's own.

    `guarded` names the file where this repository's own copy is being read, whose corruption is a
    misconfiguration to fail on; a base-side manifest is one the gate merely cannot judge, so it
    abstains and the caller skips.
    """
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        if guarded is None:
            return None
        raise SystemExit(f'ERROR - {guarded} is not readable TOML: {exc}') from exc
    project = manifest.get('project')
    version = project.get('version') if isinstance(project, dict) else None
    return version if isinstance(version, str) else None


def pinned_version(text: str, distribution: str) -> str | None:
    """The version the root manifest pins `distribution` at, or None where it pins none.

    Read as a requirement out of parsed TOML rather than scanned for as text: to a scan a
    commented-out line reads as a live pin, so a dependency deleted the way one usually is — its
    line left behind under a `#` — passes the missing-dependency case this gate exists to refuse.
    """
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # A guarded file that is present and unparseable is corrupt, not absent.
        raise SystemExit(f'ERROR - {ROOT_MANIFEST} is not readable TOML: {exc}') from exc
    project = manifest.get('project')
    dependencies = project.get('dependencies') if isinstance(project, dict) else None
    if not isinstance(dependencies, list):
        return None
    for entry in dependencies:
        if not isinstance(entry, str):
            continue
        try:
            requirement = Requirement(entry)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != canonicalize_name(distribution):
            continue
        # Only `==<version>` alone, and unconditionally, is a pin: anything else resolves to whatever
        # the index offers, and a marker (`; python_version < '3'`) leaves the member uninstalled on
        # every interpreter this project supports, where the root imports it regardless.
        specifiers = list(requirement.specifier)
        if requirement.marker is None and len(specifiers) == 1 and specifiers[0].operator == '==':
            return specifiers[0].version
    return None


def changed_paths(base: str) -> list[str] | None:
    """Every path this change touches against `base` — working tree, index, and new files.

    A brand-new module is what a `diff` alone misses, and it is exactly what a wire change looks
    like: until it is staged, git reports it only as untracked.
    """
    tracked = run_git('diff', '--name-only', base, '--')
    if tracked is None:
        return None
    staged = run_git('diff', '--name-only', '--cached', base, '--') or ''
    untracked = run_git('ls-files', '--others', '--exclude-standard') or ''
    return sorted({p for p in (tracked + '\n' + staged + '\n' + untracked).splitlines() if p.strip()})


def shipped_changes(paths: list[str], member: Member) -> list[str]:
    """The changed paths under the member's directory that could alter what an install runs."""
    prefix = f'{member.directory}/'
    return [p for p in paths if p.startswith(prefix) and not p.endswith(EXEMPT_SUFFIXES)]


def check(base: str) -> list[str]:
    """Every way this change leaves a member, its version and the root's pin out of step."""
    paths = changed_paths(base)
    if paths is None:
        print(f'NOTE - could not diff against {base}; skipping the version-bump gate.', file=sys.stderr)
    root = (REPO_ROOT / ROOT_MANIFEST).read_text()
    return [failure for member in MEMBERS for failure in check_member(member, base, paths, root)]


def check_member(member: Member, base: str, paths: list[str] | None, root: str) -> list[str]:
    """Every way this change leaves `member`, its version and the root's pin on it out of step."""
    failures: list[str] = []
    now = declared_version((REPO_ROOT / member.manifest).read_text(), guarded=member.manifest)
    if now is None:
        raise SystemExit(f'ERROR - {member.manifest} declares no readable `version`, so the gate cannot judge it.')

    # 2. The pin travels with the version whether or not this change touched the member, so it is
    #    checked first and unconditionally: a bump that forgets the pin is the same stale install.
    pinned = pinned_version(root, member.distribution)
    if pinned is None:
        # Not an absence to skip past: the root imports the package, so a root that no longer names
        # an exact version resolves whatever the index offers — the same stale-or-incompatible
        # install this gate exists to refuse, reached by deleting the pin instead of by lagging it.
        failures.append(
            f'{ROOT_MANIFEST} declares no `{member.distribution}=={{version}}`, yet the root imports '
            f'`{member.package}`. A release would resolve whatever the index offers. Pin it at {now}, '
            f'or drop this gate along with the dependency.'
        )
    elif pinned != now:
        failures.append(
            f'{ROOT_MANIFEST} pins {member.distribution}=={pinned} while {member.manifest} declares {now}. '
            f'A release would publish {member.distribution} as {now} and then publish the root depending '
            f'on {pinned} — the previous wheel. Move the pin to {now}.'
        )

    if paths is None:
        return failures
    edited = shipped_changes(paths, member)
    if not edited:
        return failures

    # 1. Changed code needs a version the index has never seen, or `skip-existing` keeps the old one.
    before = run_git('show', f'{base}:{member.manifest}')
    if before is None:
        print(f'NOTE - {base} carries no {member.manifest}; skipping the version-bump gate.', file=sys.stderr)
        return failures
    was = declared_version(before)
    if was is None:
        print(f'NOTE - {base}:{member.manifest} declares no readable `version`; skipping.', file=sys.stderr)
        return failures
    if not is_later_version(was, now, member.manifest):
        moved = 'still' if was == now else f'moved BACKWARDS from {was} to'
        failures.append(
            f'{len(edited)} file(s) changed under {member.directory}/ with `version` {moved} {now}. '
            f'Bump it in {member.manifest} (and the root pin with it): the release publishes with '
            f'`skip-existing`, so republishing {now} is a silent no-op and the root would ship '
            f'depending on bytes this repository no longer contains. First changed file: {edited[0]}'
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=None, help=f'the ref this change is judged against (else ${BASE_REV_ENV})')
    parser.add_argument('filenames', nargs='*', help='ignored; the gate reads git, not the hook filter')
    args = parser.parse_args(argv)

    ref = resolve_base_ref(args.base, os.environ.get(BASE_REV_ENV))
    base = resolve_merge_base(ref)
    if base is None:
        print(f'NOTE - {ref} does not resolve; skipping the workspace version gate.', file=sys.stderr)
        return 0
    failures = check(base)
    for failure in failures:
        print(f'ERROR - {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
