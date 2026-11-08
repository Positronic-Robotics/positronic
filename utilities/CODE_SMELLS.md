# Code-smell checks

The `code-smells` pre-commit hook and code-style CI check scan tracked production Python across the
repository. Findings block the commit; the output lists every matching location. Tests and generated
stubs are excluded.

- **QSM001:** the same compound validation guard and exception type in at least three places.
- **QSM002:** the same two-statement operation pattern in separate paths within one function.

These are syntax checks, not proof that code should be shared. Before extracting a helper, check whether
the operations belong together and whether its consumers can depend on the same module. Fix the code
when sharing is appropriate; a false positive needs a detector correction and a regression test.
There are no suppressions or automatic rewrites.

```bash
pre-commit run code-smells --all-files
uv run --locked pytest utilities/tests/test_check_code_smells.py
```

The separate `code-complexity` hook is manual: branch, local-variable, statement and nesting counts are
review signals, and splitting code just to lower them can make it harder to read.

```bash
pre-commit run code-complexity --hook-stage manual --all-files
```

## Historical backtest

```bash
uv run --locked python -m utilities.backtest_code_smells --output /tmp/code-smells.json
```

The backtest reads historical source from Git without importing or executing it. It needs the commits
listed in the script; missing revisions or tool failures stop the run. The JSON includes commit IDs,
tool versions and individual findings, including misses. Calibration cases informed the detectors;
held-out cases measure their limits. These checks do not detect architectural coupling or replace review.
