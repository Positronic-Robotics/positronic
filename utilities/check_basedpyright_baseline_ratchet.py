"""Enforce the basedpyright baseline as a one-way ratchet: a file's diagnostics may shrink, never grow.

Compares the working-tree `.basedpyright/baseline.json` against the merge-base with a base ref (the
merge-base, not the tip, so an unrelated base-side shrink cannot flag an untouched file here). A new
error grandfathered into the baseline is flagged; the fix is to resolve the error, not widen it.

The base ref is `--base <ref>`, else the `RATCHET_BASE` env var, else `origin/main` (the local
pre-commit default). CI runs a detached-HEAD checkout where `origin/main` may not resolve, so it
passes the PR base sha explicitly.

Per-code multiset, never raw lines: each baseline entry is `{"code": <rule>, "range": {startColumn,
endColumn, lineCount}}` — there is no absolute line, so re-anchoring on re-indentation shifts an
entry's columns but not its `code`. The re-anchor-safe identity is therefore the `code`: a file is
flagged if any diagnostic `code` occurs more times than at the base. That catches a brand-new
rule-kind, more entries of an existing kind, AND a swap of one code for a different one (which a bare
per-file entry-count comparison would miss, since the total stays flat).

Renames re-key the baseline (the entries move to the new path), so a rename map routes each moved
file's counts back to its base path; a pure rename/move of a file carrying existing entries is
therefore not flagged. The map unions committed renames (`base..HEAD`) with staged renames (`base`
vs the index) — under pre-commit the rename is staged but not yet in HEAD, so `--cached` is what
carries it. Rename detection fails open (empty map, every file compared under its own path) if the
git calls fail.

Residual: a swap WITHIN the same code — resolving one `reportReturnType` while introducing a different
`reportReturnType` in the same file — leaves the multiset identical and cannot be caught by a static
baseline diff (entries re-anchor by column, so there is no stable position identity). Distinguishing
it would require re-running basedpyright on both base and branch, out of scope for a fast guard.

Fails open (exit 0, note on stderr) when the base ref or either baseline cannot be resolved, so an
offline commit is never blocked; CI always has the base sha, where the gate holds.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any

BASELINE_PATH = '.basedpyright/baseline.json'


def _norm(path: str) -> str:
    """Drop the leading `./` that baseline keys carry, so paths match git's (`positronic/foo.py`)."""
    return path[2:] if path.startswith('./') else path


def _code_counts(baseline: Any) -> dict[str, Counter[str]]:
    files = baseline.get('files', {})
    return {_norm(path): Counter(str(e['code']) for e in entries) for path, entries in files.items()}


def grown_files(base: Any, current: Any, rename_map: dict[str, str] | None = None) -> list[tuple[str, str, int, int]]:
    """Return (path, code, base_count, current_count) for every diagnostic `code` a file gained.

    Compares the per-file multiset of diagnostic codes; a code absent from `base` counts as 0, so a
    new rule-kind (including a swap to a different code at flat total) or a newly-appearing file is
    flagged. `rename_map` (new path -> base path, git-style without `./`) routes a moved file's
    counts back to its base path, so a pure rename of a file with existing entries is not flagged.
    """
    renames = rename_map or {}
    base_counts = _code_counts(base)
    grown: list[tuple[str, str, int, int]] = []
    for path, cur_codes in sorted(_code_counts(current).items()):
        base_codes = base_counts.get(renames.get(path, path), Counter())
        for code, cur_n in sorted(cur_codes.items()):
            if cur_n > base_codes[code]:
                grown.append((path, code, base_codes[code], cur_n))
    return grown


def resolve_base_ref(arg: str | None, env: str | None) -> str:
    """Pick the base ref: an explicit `--base` arg wins, then `RATCHET_BASE` env, else `origin/main`."""
    return arg or env or 'origin/main'


def _parse_renames(out: str | None) -> dict[str, str]:
    """Parse `git diff --name-status -M` output into new path -> base path (git-style).

    Only `R` (rename) is mapped, never `C` (copy): a rename moves a file's entries to a new key, so
    the new path must inherit the base counts; a copy leaves the source in place and duplicates its
    diagnostics into a genuinely new file, which must face the new-file check, not reuse the source's
    grandfathered allowances.
    """
    renames: dict[str, str] = {}
    for line in (out or '').splitlines():
        parts = line.split('\t')
        if len(parts) == 3 and parts[0][:1] == 'R':
            _status, old, new = parts
            renames[new] = old
    return renames


def build_rename_map(base: str) -> dict[str, str]:
    """Map new path -> base path for files renamed from `base` to the commit under check.

    Unions committed renames (`base..HEAD`) with staged renames (`base` vs the index): under
    pre-commit the rename is staged but not yet in HEAD, so `--cached` is what carries it; in CI
    HEAD already holds it. Staged entries win on conflict, matching the tree being committed. Fails
    open (empty map) if the git calls fail, so rename-detection trouble never blocks a commit.
    """
    committed = _parse_renames(_run_git('diff', '--name-status', '-M', base, 'HEAD'))
    staged = _parse_renames(_run_git('diff', '--cached', '--name-status', '-M', base))
    return {**committed, **staged}


def _run_git(*args: str) -> str | None:
    try:
        r = subprocess.run(['git', *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _resolve_merge_base(ref: str) -> str | None:
    merge_base = _run_git('merge-base', 'HEAD', ref)
    if merge_base:
        return merge_base
    # merge-base unavailable (shallow history, unrelated ref): use the ref directly if it resolves.
    return _run_git('rev-parse', ref) or None


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Fail if any basedpyright baseline file grew vs the base ref.')
    parser.add_argument('--base', default=None, help='base ref to diff against (else $RATCHET_BASE, else origin/main)')
    args = parser.parse_args(argv)

    ref = resolve_base_ref(args.base, os.environ.get('RATCHET_BASE') or None)
    base = _resolve_merge_base(ref)
    if base is None:
        print(f'baseline-ratchet: could not resolve base {ref!r}; skipping check', file=sys.stderr)
        return 0

    base_baseline = _load_base_baseline(base)
    if base_baseline is None:
        print(f'baseline-ratchet: could not read {BASELINE_PATH} at {base}; skipping check', file=sys.stderr)
        return 0

    current_baseline = _load_working_baseline()
    if current_baseline is None:
        print(f'baseline-ratchet: could not read working-tree {BASELINE_PATH}; skipping check', file=sys.stderr)
        return 0

    grown = grown_files(base_baseline, current_baseline, build_rename_map(base))
    if not grown:
        return 0

    print('basedpyright baseline is a one-way ratchet — a file may shrink, never grow per code.', file=sys.stderr)
    print('These files gained baseline diagnostics; fix the new issue instead of grandfathering it', file=sys.stderr)
    print('(see the no_grandfather_new_code discipline):', file=sys.stderr)
    for path, code, base_n, cur_n in grown:
        print(f'  {path}: {code} {base_n} -> {cur_n}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
