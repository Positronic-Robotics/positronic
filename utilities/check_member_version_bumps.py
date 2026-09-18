"""Keep each published workspace member, its version, the root's pin on it, and the release that
publishes it in step.

The release workflow publishes every member before the root, with `skip-existing`. That flag is what
makes republishing an unchanged member a no-op rather than a failed release, and it is also the
hole: if a member changes and its version does not, PyPI keeps the old wheel, the publish step
reports success, and the root release then goes out against a version whose bytes are not the ones
in this repository. Nothing fails — a fresh install just gets the old member, which is why this is
caught here rather than at release time.

A change under a member must bump that member's `pyproject.toml` `version`, whichever member it is:
`skip-existing` does not care which package it skips. The root's `==` pin must then name exactly
that version, for the members in `EXACTLY_PINNED` — a set declared rather than read off the root's
current dependencies, since deriving it would mean a deleted pin stops being required.

The members come from the root's `[tool.uv.workspace] members`, so one added later is gated without
touching this file.

The version must INCREASE, not merely differ: a version already published under other code is worse
than no bump at all, since the index will keep whichever bytes got there first.

Judged against `--base` (else `$RATCHET_BASE`, else `origin/main`), at the merge-base, so a bump that
landed on the base side meanwhile is not this change's to claim.

Fails open (exit 0, note on stderr) where it cannot judge — an unresolvable base, or a base that
carries no manifest for a member (that member's own first commit) — so an offline commit is never
blocked and CI, which always has the base sha, is where the gate holds. A manifest that is present and
unreadable fails closed: that is a corrupt guarded file, not an absence.

It runs under `uv run` rather than a bare interpreter, because it reads the root's pin as a
requirement and compares versions by PEP 440 — both `packaging`'s to own, not this file's to
approximate.

The git plumbing below is this module's own rather than shared with the other `utilities/check_*`
scripts: they run as `python3 utilities/<script>.py`, where the repository root is not on `sys.path`
and a `utilities.` import cannot resolve.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_REV_ENV = 'RATCHET_BASE'

ROOT_MANIFEST = 'pyproject.toml'

RELEASE_WORKFLOW = '.github/workflows/release.yaml'

# The job that uploads the root distribution. It is the one that must WAIT for every member's
# job, since an install of the root resolves the members it requires.
ROOT_PUBLISH_JOB = 'publish-pypi'

# The distributions the root must pin exactly, canonicalized so a manifest spelling the same name
# another way is still held to it. A member the root depends on by a FLOOR is deliberate and is not
# listed: `positronic-eval-vocabulary` is append-only and read tolerantly.
EXACTLY_PINNED = frozenset(map(canonicalize_name, ('positronic-platform-client',)))

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


def is_later_version(was: str, is_now: str, manifest: str = ROOT_MANIFEST) -> bool:
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
        # the index offers, and a marker (`; python_version < '3'`) leaves the client uninstalled on
        # every interpreter this project supports, where the CLI imports `platform_client` regardless.
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


def workspace_members(text: str) -> list[str]:
    """The workspace members the root declares, each a directory that ships its own distribution.

    Read from the root rather than listed here, so a member added later is gated without an edit to
    this file — which is the whole failure this gate exists to catch, one level up.
    """
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f'ERROR - {ROOT_MANIFEST} is not readable TOML: {exc}') from exc
    workspace = manifest.get('tool', {}).get('uv', {}).get('workspace')
    members = workspace.get('members') if isinstance(workspace, dict) else None
    return [m for m in members if isinstance(m, str)] if isinstance(members, list) else []


def distribution_name(text: str, *, guarded: str) -> str:
    """The distribution a member's manifest declares. A member that names none cannot be published,
    and the release workflow would fail on it, so this is a misconfiguration rather than an absence."""
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f'ERROR - {guarded} is not readable TOML: {exc}') from exc
    project = manifest.get('project')
    name = project.get('name') if isinstance(project, dict) else None
    if not isinstance(name, str):
        raise SystemExit(f'ERROR - {guarded} declares no `name`, so the gate cannot judge what it publishes.')
    return name


def shipped_changes(paths: list[str], member: str) -> list[str]:
    """The changed paths under `member/` that could alter what an install runs."""
    prefix = f'{member}/'
    return [p for p in paths if p.startswith(prefix) and not p.endswith(EXEMPT_SUFFIXES)]


class Member(NamedTuple):
    """One published workspace member as this gate reads it: where it lives, the manifest that
    declares it, the distribution it publishes, and the version that manifest declares now."""

    dir: str
    manifest: str
    distribution: str
    version: str


def pin_failures(member: Member) -> list[str]:
    """Why the root's pin on `distribution` is out of step with the version the member declares.

    The pin travels with the version whether or not this change touched the member, so it is checked
    unconditionally: a bump that forgets the pin is the same stale install.
    """
    if canonicalize_name(member.distribution) not in EXACTLY_PINNED:
        return []
    pinned = pinned_version((REPO_ROOT / ROOT_MANIFEST).read_text(), member.distribution)
    if pinned is None:
        # Not an absence to skip past: a root that no longer names an exact version resolves whatever
        # the index offers — the same stale-or-incompatible install this gate exists to refuse,
        # reached by deleting the pin instead of by lagging it.
        return [
            f'{ROOT_MANIFEST} declares no `{member.distribution}=={{version}}`, yet it is depended '
            f'on exactly. A release would resolve whatever the index offers. Pin it at '
            f'{member.version}, or drop it from EXACTLY_PINNED along with the dependency.'
        ]
    if pinned != member.version:
        return [
            f'{ROOT_MANIFEST} pins {member.distribution}=={pinned} while {member.manifest} declares '
            f'{member.version}. A release would publish {member.dir} as {member.version} and then '
            f'publish the root depending on {pinned} — the previous wheel. Move the pin to '
            f'{member.version}.'
        ]
    return []


def bump_failures(base: str, member: Member, edited: list[str]) -> list[str]:
    """Why `edited` changes under `member` cannot reach an install at the version it declares.

    Abstains where the base carries no manifest for the member — its own first commit — so a member
    being added is never blocked by the gate that will guard it from the next change on.
    """
    before = run_git('show', f'{base}:{member.manifest}')
    if before is None:
        print(f'NOTE - {base} carries no {member.manifest}; skipping its version-bump gate.', file=sys.stderr)
        return []
    was = declared_version(before)
    if was is None:
        print(f'NOTE - {base}:{member.manifest} declares no readable `version`; skipping.', file=sys.stderr)
        return []
    if is_later_version(was, member.version, member.manifest):
        return []
    moved = 'still' if was == member.version else f'moved BACKWARDS from {was} to'
    pin = ' (and the root pin with it)' if canonicalize_name(member.distribution) in EXACTLY_PINNED else ''
    return [
        f'{len(edited)} file(s) changed under {member.dir}/ with `version` {moved} {member.version}. '
        f'Bump it in {member.manifest}{pin}: the release publishes with `skip-existing`, so '
        f'republishing {member.version} is a silent no-op and a fresh install would get bytes this '
        f'repository no longer contains. First changed file: {edited[0]}'
    ]


def read_member(name: str) -> Member:
    """The member `name` as its own manifest declares it. A member the root lists without one, or
    without a name or version, is a misconfiguration the release would fail on, so it raises."""
    manifest_path = f'{name}/pyproject.toml'
    manifest = REPO_ROOT / manifest_path
    if not manifest.exists():
        raise SystemExit(f'ERROR - {manifest_path} is missing, yet {ROOT_MANIFEST} lists {name} as a member.')
    text = manifest.read_text()
    version = declared_version(text, guarded=manifest_path)
    if version is None:
        raise SystemExit(f'ERROR - {manifest_path} declares no readable `version`, so the gate cannot judge it.')
    return Member(name, manifest_path, distribution_name(text, guarded=manifest_path), version)


def check_member(base: str, member: Member, paths: list[str] | None) -> list[str]:
    """Every way this change leaves one member, its version and the root's pin on it out of step."""
    failures = pin_failures(member)
    edited = shipped_changes(paths, member.dir) if paths is not None else []
    return failures + bump_failures(base, member, edited) if edited else failures


def release_jobs(text: str) -> dict[str, dict]:
    """The release workflow's jobs, by name. A workflow that declares none publishes nothing, which
    is a corrupt guarded file rather than an absence, so it raises."""
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemExit(f'ERROR - {RELEASE_WORKFLOW} is not readable YAML: {exc}') from exc
    jobs = workflow.get('jobs') if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        raise SystemExit(f'ERROR - {RELEASE_WORKFLOW} declares no `jobs`, so the gate cannot judge it.')
    return {name: job for name, job in jobs.items() if isinstance(job, dict)}


def publishing_jobs(jobs: dict[str, dict]) -> dict[str, str]:
    """Each member directory the release uploads, and the job that uploads it.

    A member's job names `packages-dir: <member>/dist`; the root's own upload names no
    `packages-dir` at all, so it is never read as a member's.
    """
    published: dict[str, str] = {}
    for name, job in jobs.items():
        for step in job.get('steps') or []:
            options = step.get('with') if isinstance(step, dict) else None
            directory = options.get('packages-dir') if isinstance(options, dict) else None
            if isinstance(directory, str) and directory.endswith('/dist'):
                published[directory.removesuffix('/dist')] = name
    return published


def job_needs(job: dict) -> list[str]:
    """The jobs this one waits for. GitHub takes one name or a list, so both read the same here."""
    needs = job.get('needs') or []
    return [needs] if isinstance(needs, str) else [n for n in needs if isinstance(n, str)]


def release_failures(members: list[str], text: str) -> list[str]:
    """Why the release would not put a member on the index before the root that requires it.

    The version half above demands a bump for every member the root declares. A member nothing
    uploads is then bumped forever and published never, and one the root does not wait for can
    reach the index after the release that requires it.
    """
    jobs = release_jobs(text)
    root = jobs.get(ROOT_PUBLISH_JOB)
    if root is None:
        raise SystemExit(f'ERROR - {RELEASE_WORKFLOW} declares no `{ROOT_PUBLISH_JOB}`, so nothing publishes the root.')
    published, needs = publishing_jobs(jobs), job_needs(root)
    failures = []
    for member in members:
        job = published.get(member)
        if job is None:
            failures.append(
                f'{RELEASE_WORKFLOW} uploads no {member}/dist, so a change under {member}/ is gated on a '
                f'version bump that no release publishes. Add a job that publishes it, and name that job '
                f'in `{ROOT_PUBLISH_JOB}`.'
            )
        elif job not in needs:
            failures.append(
                f'`{ROOT_PUBLISH_JOB}` does not need `{job}`, so the root can reach the index before the '
                f'{member} version it requires. Add `{job}` to its `needs`.'
            )
    return failures


def check(base: str) -> list[str]:
    """Every way this change leaves a published member out of step, over every member."""
    members = workspace_members((REPO_ROOT / ROOT_MANIFEST).read_text())
    if not members:
        raise SystemExit(f'ERROR - {ROOT_MANIFEST} declares no workspace members, so the gate guards nothing.')
    paths = changed_paths(base)
    if paths is None:
        print(f'NOTE - could not diff against {base}; skipping the version-bump half of the gate.', file=sys.stderr)
    failures = [failure for name in members for failure in check_member(base, read_member(name), paths)]
    return failures + release_failures(members, (REPO_ROOT / RELEASE_WORKFLOW).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=None, help=f'the ref this change is judged against (else ${BASE_REV_ENV})')
    parser.add_argument('filenames', nargs='*', help='ignored; the gate reads git, not the hook filter')
    args = parser.parse_args(argv)

    ref = resolve_base_ref(args.base, os.environ.get(BASE_REV_ENV))
    base = resolve_merge_base(ref)
    if base is None:
        print(f'NOTE - {ref} does not resolve; skipping the member version gate.', file=sys.stderr)
        return 0
    failures = check(base)
    for failure in failures:
        print(f'ERROR - {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
