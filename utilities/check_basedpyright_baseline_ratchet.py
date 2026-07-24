"""Enforce the basedpyright baseline as a one-way ratchet: a file's entry count may shrink, never grow.

Compares the working-tree `.basedpyright/baseline.json` against the merge-base with `origin/main`
(the merge-base, not the tip, so an unrelated main-side shrink cannot flag an untouched file here).
Any file whose entry count exceeds its merge-base count has a new error grandfathered into the
baseline; the fix is to resolve the error, not to widen the baseline.

Counts, never raw lines: re-anchoring an existing entry shifts its line numbers legitimately, so only
the number of entries per file is compared.

Fails open (exit 0, note on stderr) when the base ref or either baseline cannot be resolved, so an
offline commit is never blocked; CI always has origin/main, where the gate holds.
"""

import json
import subprocess
import sys
from typing import Any

BASELINE_PATH = '.basedpyright/baseline.json'


def _counts(baseline: Any) -> dict[str, int]:
    return {path: len(entries) for path, entries in baseline.get('files', {}).items()}


def grown_files(base: Any, current: Any) -> list[tuple[str, int, int]]:
    """Return (path, base_count, current_count) for every file whose baseline entry count grew.

    A file absent from `base` counts as 0, so a newly-appearing file with entries is a growth.
    """
    base_counts = _counts(base)
    grown: list[tuple[str, int, int]] = []
    for path, cur_n in sorted(_counts(current).items()):
        base_n = base_counts.get(path, 0)
        if cur_n > base_n:
            grown.append((path, base_n, cur_n))
    return grown


def _run_git(*args: str) -> str | None:
    try:
        r = subprocess.run(['git', *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _resolve_base() -> str | None:
    merge_base = _run_git('merge-base', 'HEAD', 'origin/main')
    if merge_base:
        return merge_base
    return _run_git('rev-parse', 'origin/main') or None


def _load_base_baseline(base: str) -> Any | None:
    raw = _run_git('show', f'{base}:{BASELINE_PATH}')
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _load_working_baseline() -> Any | None:
    try:
        with open(BASELINE_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    base = _resolve_base()
    if base is None:
        print('baseline-ratchet: could not resolve merge-base with origin/main; skipping check', file=sys.stderr)
        return 0

    base_baseline = _load_base_baseline(base)
    if base_baseline is None:
        print(f'baseline-ratchet: could not read {BASELINE_PATH} at {base}; skipping check', file=sys.stderr)
        return 0

    current_baseline = _load_working_baseline()
    if current_baseline is None:
        print(f'baseline-ratchet: could not read working-tree {BASELINE_PATH}; skipping check', file=sys.stderr)
        return 0

    grown = grown_files(base_baseline, current_baseline)
    if not grown:
        return 0

    print('basedpyright baseline is a one-way ratchet — it may shrink, never grow per file.', file=sys.stderr)
    print('These files gained baseline entries; fix the new issue instead of grandfathering it', file=sys.stderr)
    print('(see the no_grandfather_new_code discipline):', file=sys.stderr)
    for path, base_n, cur_n in grown:
        print(f'  {path}: {base_n} -> {cur_n} entries', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
