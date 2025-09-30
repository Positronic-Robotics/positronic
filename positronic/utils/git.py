"""Utilities for querying Git repository metadata.

This module is intentionally lightweight and safe to import in environments
without Git or outside of a repository. All functions return None when Git
information cannot be determined.
"""

import subprocess


def get_git_state() -> dict[str, str | bool] | None:
    """Return a dictionary with basic Git metadata or None if unavailable.

    The returned mapping includes:
      - commit: str  (current HEAD SHA)
      - branch: str  (current branch name)
      - dirty: bool  (True if there are uncommitted changes)

    Returns None if the current working directory is not inside a Git repo
    or if Git is not installed/accessible.
    """
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
        dirty = bool(status.strip())
        return {'commit': commit, 'branch': branch, 'dirty': dirty}
    except Exception:
        return None


__all__ = ['get_git_state']
