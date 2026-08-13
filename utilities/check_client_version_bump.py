"""Keep the published client, its version, and the root's pin on it in step.

`positronic` requires `positronic-platform-client==<version>`, and the release workflow publishes
the client first with `skip-existing`. That flag is what makes republishing an unchanged client a
no-op rather than a failed release, and it is also the hole: if `client/` changes and its version
does not, PyPI keeps the old wheel, the publish step reports success, and the root release then goes
out depending on a version whose bytes are not the ones in this repository. Nothing fails — a fresh
install just gets the old client, which is why this is caught here rather than at release time.

Two things are checked, and it takes both to close it:

1. A change under `client/` bumps `client/pyproject.toml`'s `version`. Without this the new code
   never reaches the index, because `skip-existing` skips a version already published.
2. The root's `positronic-platform-client==` pin names exactly that version. Without this the
   client publishes fine and the root ships depending on the previous one.

The version must INCREASE, not merely differ: a version already published under other code is worse
than no bump at all, since the index will keep whichever bytes got there first.

Judged against `--base` (else `$RATCHET_BASE`, else `origin/main`), at the merge-base, so a bump that
landed on the base side meanwhile is not this change's to claim.

Fails open (exit 0, note on stderr) where it cannot judge — an unresolvable base, or a base with no
`client/pyproject.toml` (the client's own first commit) — so an offline commit is never blocked and
CI, which always has the base sha, is where the gate holds. A manifest that is present and
unreadable fails closed: that is a corrupt guarded file, not an absence.

The git plumbing below is this module's own rather than shared with the other `utilities/check_*`
scripts: they run as `python3 utilities/<script>.py`, where the repository root is not on `sys.path`
and a `utilities.` import cannot resolve.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_REV_ENV = 'RATCHET_BASE'

# The directory the distribution ships, its manifest, and the name the root pins it by.
CLIENT_DIR = 'client'
CLIENT_MANIFEST = 'client/pyproject.toml'
ROOT_MANIFEST = 'pyproject.toml'
DISTRIBUTION = 'positronic-platform-client'

# Changes that cannot reach the installed wheel. A test is NOT here: it ships inside the package
# directory, and a reader comparing two revisions may expect the same code behind the same version.
EXEMPT_SUFFIXES = ('.md',)

_VERSION_LINE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.M)
_PIN = re.compile(rf'{re.escape(DISTRIBUTION)}\s*==\s*([0-9][^"\',\s]*)')


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


def _ordered(version: str, parts: int) -> tuple:
    """`version` as something two of them compare by, padded to `parts` components.

    Numeric parts compare as numbers, so `0.10.0` does not read as older than `0.9.0`. Both sides
    pad to the same width, so `1.0` and `1.0.0` compare EQUAL — packaging treats them as one
    version, and reading the extra zero as a bump would accept a release that ships as its
    predecessor. A non-numeric part sorts after the numbers, so a pre-release suffix orders
    consistently instead of crashing the gate.
    """
    read = [(0, int(p)) if p.isdigit() else (1, p) for p in version.replace('-', '.').split('.')]
    return tuple(read + [(0, 0)] * (parts - len(read)))


def increases(was: str, is_now: str) -> bool:
    """Whether `is_now` is a LATER version than `was`, both read to the same number of parts."""
    width = max(len(was.replace('-', '.').split('.')), len(is_now.replace('-', '.').split('.')))
    return _ordered(is_now, width) > _ordered(was, width)


def declared_version(text: str) -> str | None:
    """The `version` a pyproject declares, or None where it declares none this can read."""
    match = _VERSION_LINE.search(text)
    return match.group(1) if match else None


def pinned_version(text: str) -> str | None:
    """The version the root manifest pins the client at, or None where it pins none."""
    match = _PIN.search(text)
    return match.group(1) if match else None


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


def shipped_changes(paths: list[str]) -> list[str]:
    """The changed paths under `client/` that could alter what an install runs."""
    prefix = f'{CLIENT_DIR}/'
    return [p for p in paths if p.startswith(prefix) and not p.endswith(EXEMPT_SUFFIXES)]


def check(base: str) -> list[str]:
    """Every way this change leaves the client, its version and the root's pin out of step."""
    failures: list[str] = []
    now = declared_version((REPO_ROOT / CLIENT_MANIFEST).read_text())
    if now is None:
        raise SystemExit(f'ERROR - {CLIENT_MANIFEST} declares no readable `version`, so the gate cannot judge it.')

    # 2. The pin travels with the version whether or not this change touched the client, so it is
    #    checked first and unconditionally: a bump that forgets the pin is the same stale install.
    pinned = pinned_version((REPO_ROOT / ROOT_MANIFEST).read_text())
    if pinned is None:
        print(f'NOTE - {ROOT_MANIFEST} pins no {DISTRIBUTION}=={{version}}; skipping the pin check.', file=sys.stderr)
    elif pinned != now:
        failures.append(
            f'{ROOT_MANIFEST} pins {DISTRIBUTION}=={pinned} while {CLIENT_MANIFEST} declares {now}. '
            f'A release would publish the client as {now} and then publish the root depending on '
            f'{pinned} — the previous wheel. Move the pin to {now}.'
        )

    paths = changed_paths(base)
    if paths is None:
        print(f'NOTE - could not diff against {base}; skipping the version-bump gate.', file=sys.stderr)
        return failures
    edited = shipped_changes(paths)
    if not edited:
        return failures

    # 1. Changed code needs a version the index has never seen, or `skip-existing` keeps the old one.
    before = run_git('show', f'{base}:{CLIENT_MANIFEST}')
    if before is None:
        print(f'NOTE - {base} carries no {CLIENT_MANIFEST}; skipping the version-bump gate.', file=sys.stderr)
        return failures
    was = declared_version(before)
    if was is None:
        print(f'NOTE - {base}:{CLIENT_MANIFEST} declares no readable `version`; skipping.', file=sys.stderr)
        return failures
    if not increases(was, now):
        moved = 'still' if was == now else f'moved BACKWARDS from {was} to'
        failures.append(
            f'{len(edited)} file(s) changed under {CLIENT_DIR}/ with `version` {moved} {now}. '
            f'Bump it in {CLIENT_MANIFEST} (and the root pin with it): the release publishes with '
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
        print(f'NOTE - {ref} does not resolve; skipping the client version gate.', file=sys.stderr)
        return 0
    failures = check(base)
    for failure in failures:
        print(f'ERROR - {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
