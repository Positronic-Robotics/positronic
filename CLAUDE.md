# Contributor guide

## Contributor behavior
- Don't restore code that you wrote and I deleted
- Don't make commits or git changes unprompted; an invoked PR workflow skill (e.g. `/polish`, `/push-pr`)
  authorizes its commits and pushes
- Don't hide errors with try-catch blocks; let failures surface until asked otherwise
- Stay scoped: no features beyond the request. Structure of the code you touch is governed by Design
  discipline below — bringing it to the end state is in scope, refactoring unrelated code is not
- Don't add comments, docstrings, or type annotations to code you didn't change
- Ask clarifying questions only when requirements are genuinely ambiguous and investigation can't resolve them

## Commands
- Every Python execution goes through `uv run --locked` — bare `python`/`pytest` bypasses the locked venv
- Run tests: `uv run --locked pytest --no-cov`
- Run single test file: `uv run --locked pytest path/to/test_file.py --no-cov`
- Lint: `uv run --locked ruff check --fix .`
- Format: `uv run --locked ruff format .`
- Run any Python: `uv run --locked python script.py`
- Syntax check: `uv run --locked python -m py_compile file.py`

## Dependency management
- `uv.lock` is committed; CI and Docker run `uv sync --locked` to install exactly what's locked
- To change deps: edit `pyproject.toml`, then run `uv lock`, then commit `pyproject.toml` and `uv.lock` together in one reviewed change — never let `uv.lock` drift implicitly

## Design discipline
- When a change has a design thesis ("World owns time", "Harness is name-free"), enumerate its consequences for
  every touched interface before coding, and implement the end state — old pathways (constructor args, public
  mutators, parallel flags) must not survive the refactor
- Every value has one owner. When a value gains a new home, re-route consumers to it; don't plumb the new source
  into the old parameter
- When current code conflicts with the target design, resolve it now (rename + migrate) or bridge loudly with a
  TODO/HACK comment. Never bridge silently with an extra field, class, or indirection
- Internal code breaks cleanly — no speculative compat shims. Before a migrate-everywhere change, grep for
  consumers that need the old form; alias or migrate only the real ones
- A new class, file, or field must earn its place: if existing structure already encodes the distinction (the dict
  an entry lives in, an enum, the calling context), don't reify the concept into a type. Extend an existing module
  rather than adding a file
- No `X | None` for logically required values — an Optional that is never None in practice lies about the contract
- After every fix or refactor, re-read the resulting code as a whole, not the diff: fixes create new smells (e.g.
  two classes become structurally identical only after their serializers are unified — merge them)

## Code style
- No imports inside functions/methods; always place imports at the top of the file
- Exception: circular dependencies or truly unavoidable cases
- No `hasattr`/`getattr` hacks for type dispatch; use `isinstance` with proper base classes or protocols
- Judge names at the call site (`ds`, `teleop` — not `dataset`, `teleoperation`) and against the roadmap
  (`mujoco_franka`, not `sim` — more sims will exist)
- Before inventing a name or pattern, mirror adjacent code (config placeholders are named `placeholder`; new
  constants join the existing constants block)
- Configuronic: always the `@cfn.config` decorator, never `NAME = cfn.Config(func)`; define variants with `.override`
- No suppressions or guards for problems never observed: a `noqa` must suppress an error that actually fires; no
  defensive idioms with explanatory comments

## Comments & docstrings
- Write for a fresh reader: no references to past or future state ("no longer", "previously", "step N", "today").
  Future work is a TODO, nothing else
- Never write a comment justifying awkward code — the urge to justify is the signal to fix the design or mark it
  HACK/TODO
- Don't restate the code; if there is nothing to say beyond it, stay silent. Docstrings are plain statements of
  what the thing is
- Comments wrap at 120 columns, same as code

## Testing
- Don't add new test files unless explicitly asked

## Commit messages
- Short, imperative sentences (e.g., "Fix wrong type", not "Fixed wrong type")
- Use backticks for code references (e.g., "Fix `RemoteDataset` connection leak")
- No trailing period for short messages
- No Claude/AI attribution
- Never amend commits; always create new commits
- Never use `--no-gpg-sign` or `--no-verify` — commits must be signed

## Infrastructure
- Machines, Docker contexts and images: `docker/CONTEXTS.md`
- Model-specific workflows: `positronic/vendors/{lerobot,gr00t,openpi}/README.md`
- Reconstructing previous runs: read `run_metadata_*.yaml` and episode `static.json` from output directory

## Related repositories
- `../gr00t` — GR00T model configs and training (Positronic-Robotics/gr00t)
- `../openpi` — OpenPI model integration
- `../internal` — Internal scripts and infrastructure
