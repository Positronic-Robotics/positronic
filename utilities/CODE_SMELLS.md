# Code-smell candidates

These checks are opt-in review tools. They name no application classes and do not run on ordinary commits.
Historical source comes directly from Git; the detectors never import or execute it.

## Proposals

| Check | Evidence and benefit | Cost and recommendation |
|---|---|---|
| **QSM001: repeated validation** | Finds compound conditions followed by a single exception. Catches the four finite/nonnegative checks consolidated in `a77112635`. Main at `ba15b748d` has five groups, including JPEG quality checks in five places. | Matching syntax does not prove shared ownership. A thin package must not acquire robotics dependencies merely to reuse validation. Recommend a review prompt. |
| **QSM002: repeated control paths** | Finds two-statement blocks making at least two calls, repeated in separate paths within one function. Catches the two submission paths removed in `126b77436`; intentional parallel calls pass. Main has seven groups, including common actuator initialization in two branches. | Initial emission versus loop emission, or cleanup on separate exits, can justify repetition. Values may differ; no automatic rewrite is safe. Recommend a review prompt. |
| **Ruff complexity profile** | Checks branches, locals, statements and nesting at Ruff's default thresholds. Identifies three historical loop simplifications. Main has 32 production diagnostics and 15 in tests. | Numerical routines and test setup can reasonably be large. Shrinking functions purely to meet counts can worsen the code. Recommend reviewing changes in these metrics before making them blocking. |

The manual hooks exit nonzero when they find candidates. This makes them usable as gates, but enabling them
on ordinary commits requires deciding how findings are reviewed and how justified cases are waived.
QSM checks currently report every match; they have no suppression mechanism or automatic fixes.

## What the backtest establishes

Using Ruff 0.15.2, six of eight calibration cases produce the relevant warning before the change and lose it
afterwards. Those include the two custom checks, three loop metrics, and an exploratory argument-count rule.
The cryptic overlapping-image expression and the public-method coupling between components are missed.
Calibration cases informed the implementation; six successes are not an independent accuracy estimate.

Four further changes were selected by commit subject before their diffs were inspected: consolidating RTC
context, combining mapping addition/removal, deleting image-size metadata, and moving a helper beside its
caller. None is correctly identified as the central problem by these checks. Whole-file warnings remain in
two cases; the JSON preserves them rather than counting them as successful detections.

The current-code scan covers all 639 tracked Python files at `ba15b748d`, excluding generated stubs.
Production and test counts are separate. Findings in accepted code still need judgment: acceptance alone
does not make a finding wrong.

## Candidates not recommended as blanket gates

- **Private access:** Ruff reports 163 production and 625 test accesses. Nested strategy objects and adapter
  implementations legitimately cooperate this way. It also misses problematic coupling through public methods.
- **Private forwarding helpers:** the prototype finds 12. Eleven are dataset adapters or codec hooks;
  removing those methods would remove required behavior. Body length alone does not tell whether a helper earns its place.
- **Argument count:** 102 production warnings. It catches the six-to-five-argument `_call_obs` simplification,
  but also flags explicit configuration interfaces. The RTC context consolidation even adds two argument-count
  warnings while improving locality. Keep it exploratory, especially for internal helpers.
- **Tests of internals:** private access is not enough to distinguish a private API being tested from an
  assertion about incidental implementation order. Keep behavior-specific regression tests and human review.

Foundational APIs can receive stronger dependency restrictions, but that requires declaring the allowed
dependency direction. A general syntax check cannot infer which public type belongs in which module.
No public-name allowlist or processor-specific dependency ban is included.

## Run

From the repository root, with development dependencies installed:

```bash
uv run --locked python -m utilities.check_code_smells
uv run --locked ruff check --config utilities/code_smells.toml positronic
uv run --locked python -m utilities.backtest_code_smells --output /tmp/code-smells.json
uv run --locked pytest utilities/tests/test_check_code_smells.py
```

Or use the manual hooks:

```bash
pre-commit run code-smells --hook-stage manual --all-files
pre-commit run code-complexity --hook-stage manual --all-files
```

The backtest needs the commits named in `backtest_code_smells.py` to exist locally; a shallow clone may not
contain them. It records full commit IDs, tool version, working-tree status and individual findings. A missing
revision, invalid Python source or linter failure stops the run. A completed run can report missed cases.

QSM001 preserves operators, bounds, attribute names and repeated-variable relationships, while ignoring local
variable spelling and diagnostic text. QSM002 ignores value spelling when comparing operation structure.
Neither resolves imports, dynamic calls or semantic equivalence. Different import aliases can hide matches;
identically spelled functions can have different meanings. A shared helper must still have a valid home.
Two-statement windows can overlap: three of the seven QSM002 groups describe one longer RoboLab setup block.

Ruff documents the [available complexity thresholds](https://docs.astral.sh/ruff/settings/#lint_pylint).
The locals and nesting rules require preview mode; the opt-in profile enables it without changing ordinary lint.
