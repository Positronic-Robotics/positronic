"""Utilities for querying Git repository metadata.

This module is intentionally lightweight and safe to import in environments
without Git or outside of a repository. All functions return None when Git
information cannot be determined.
"""

import json
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


def get_git_state(workdir: Path | None = None) -> dict[str, str | bool] | None:
    """Return a dictionary with basic Git metadata or None if unavailable.

    The returned mapping includes:
      - commit: str  (current HEAD SHA)
      - branch: str  (current branch name)
      - dirty: bool  (True if there are uncommitted changes)

    Returns None if the current working directory is not inside a Git repo
    or if Git is not installed/accessible.
    """
    if workdir is None:
        workdir = Path.cwd()

    try:
        kwargs = {'capture_output': True, 'text': True, 'check': True, 'cwd': workdir}
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], **kwargs).stdout.strip()
        branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], **kwargs).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], **kwargs).stdout
        dirty = bool(status.strip())
        return {'commit': commit, 'branch': branch, 'dirty': dirty}
    except Exception:
        return None


def get_git_diff(workdir: Path | None = None, patterns: list[str] | None = None) -> str | None:
    """Return git diff for uncommitted changes matching patterns.

    Captures both staged and unstaged changes for files matching the specified
    patterns. If no patterns are provided, defaults to ['*.py'].

    Args:
        patterns: List of file patterns (e.g., ['*.py', '*.toml']).
                 Defaults to ['*.py'] if None.

    Returns:
        Git diff as string, or None if not in a git repo, git is unavailable,
        or there are no changes matching the patterns.
    """
    if workdir is None:
        workdir = Path.cwd()

    if patterns is None:
        patterns = ['*.py']

    try:
        # Use 'git diff HEAD' to capture both staged and unstaged changes
        kwargs = {'capture_output': True, 'text': True, 'check': True, 'cwd': workdir}
        result = subprocess.run(['git', 'diff', 'HEAD', '--'] + patterns, **kwargs)
        diff = result.stdout.strip()
        return diff if diff else None
    except Exception:
        return None


def get_package_checkout(distribution: str = 'positronic') -> Path | None:
    """Return the checkout an editable install of ``distribution`` imports from, or None.

    A wheel, a PyPI install and a missing distribution all answer None.
    """
    direct_url = _direct_url(distribution)
    if direct_url is None or not direct_url.get('dir_info', {}).get('editable'):
        return None
    return Path(url2pathname(urlparse(direct_url['url']).path))


def get_package_git_state(distribution: str = 'positronic') -> dict[str, str | bool] | None:
    """Return the git revision of the installed ``distribution``, or None if it has none.

    A wheel built from a VCS URL answers with the commit its ``direct_url.json`` names (PEP 610).
    An editable install answers with the state of the checkout it imports from. Any other install
    has no revision. The git repository around ``site-packages`` never answers: a venv inside a
    checkout would name that checkout, which is not the code in the process.
    """
    direct_url = _direct_url(distribution)
    if direct_url is None:
        return None
    vcs = direct_url.get('vcs_info')
    if vcs is not None:
        state: dict[str, str | bool] = {'commit': vcs['commit_id'], 'dirty': False, 'url': direct_url['url']}
        if 'requested_revision' in vcs:
            state['requested_revision'] = vcs['requested_revision']
        return state
    checkout = get_package_checkout(distribution)
    return get_git_state(workdir=checkout) if checkout is not None else None


def _direct_url(distribution: str) -> dict | None:
    try:
        text = importlib_metadata.distribution(distribution).read_text('direct_url.json')
    except importlib_metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


__all__ = ['get_git_state', 'get_git_diff', 'get_package_checkout', 'get_package_git_state']
