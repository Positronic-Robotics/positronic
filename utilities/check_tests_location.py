"""Enforce the test-layout convention: every pytest file lives under a per-package `tests/` directory.

Takes file paths as arguments (pre-commit passes changed files; CI passes changed files), prints each
violation with the expected location, and exits non-zero if any are found.
"""

import sys
from pathlib import Path


def violations(paths: list[Path]) -> list[Path]:
    return [
        p
        for p in paths
        if p.suffix == '.py'
        and (p.name.startswith('test_') or p.name.endswith('_test.py'))
        and 'tests' not in p.parts[1:-1]
    ]


def main() -> int:
    bad = violations([Path(a) for a in sys.argv[1:]])
    for p in bad:
        # A root-level or root-tests/ file has no package to anchor the hint on.
        rootish = len(p.parts) == 1 or p.parts[0] == 'tests'
        hint = Path('<package>', 'tests', p.name) if rootish else p.parent / 'tests' / p.name
        print(f'{p}: test files live in a per-package tests/ directory, e.g. {hint}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
