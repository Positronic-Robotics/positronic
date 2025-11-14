"""Utilities for querying Git repository metadata.

This module is intentionally lightweight and safe to import in environments
without Git or outside of a repository. All functions return None when Git
information cannot be determined.
"""

import subprocess
from pathlib import Path


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


__all__ = ['get_git_state', 'get_git_diff']
