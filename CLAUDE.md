# Contributor behavior
- Read [`pimm/README.md`](pimm/README.md) before writing or reviewing a control system, a world or
  anything wired into one — it is the runtime everything here runs on, and `ARCHITECTURE.md` states
  what this repository builds on top of it
- Read `CODE_RULES.md` before writing code here — the named rules it carries govern authoring, not only
  review, and `check-rules` is how you check a change against them
- Don't restore code that you wrote and I deleted
- Fix a defect when you find it, whether or not a current path reaches it. An unexercised wrong path
  produces no signal until something depends on it, so the failure surfaces later and further from its
  cause. Scope can justify deferring; being latent cannot
- Stay scoped: no features beyond the request. Structure of the code you touch is governed by Design
  discipline below — bringing it to the end state is in scope, refactoring unrelated code is not
- Don't add comments, docstrings, or type annotations to code you didn't change
- Ask clarifying questions only when requirements are ambiguous and investigation can't resolve them

# Commands
- Every Python execution goes through `uv run --locked` — bare `python`/`pytest` bypasses the locked venv
- Run tests: `uv run --locked pytest`
- Run single test file: `uv run --locked pytest path/to/test_file.py`
- Lint: `uv run --locked ruff check --fix .`
- Format: `uv run --locked ruff format .`
- Run any Python: `uv run --locked python script.py`
- Syntax check: `uv run --locked python -m py_compile file.py`

# Dependency management
- `uv.lock` is committed; CI and Docker run `uv sync --locked` to install exactly what's locked
- To change deps: edit `pyproject.toml`, then run `uv lock`, then commit `pyproject.toml` and `uv.lock` together in one reviewed change — never let `uv.lock` drift implicitly
- The dev tools are a dependency group, so every `uv run` and `uv sync` installs them without being asked. An
  image that must stay lean excludes them with `--no-dev`
- `uv run` and `uv sync` make the venv match what the command asks for and uninstall everything else, so name
  every extra the task needs in one command (`uv sync --locked --extra yam --extra hardware`); a second command
  replaces the first rather than adding to it
- `pre-commit` is a uv tool (`uv tool install pre-commit`), not a project dependency: the git hook execs it by
  path, so it has to live somewhere a sync cannot remove

# Design discipline
- When a change has a design thesis ("World owns time", "Harness is name-free"), enumerate its consequences for
  every touched interface before coding, and implement the end state — old pathways (constructor args, public
  mutators, parallel flags) must not survive the refactor
- Every value has one owner. When a value gains a new home, re-route consumers to it; don't plumb the new source
  into the old parameter
- When current code conflicts with the target design, resolve it now (rename + migrate) or bridge loudly with a
  TODO/HACK comment. Never bridge silently with an extra field, class, or indirection
- Internal code breaks cleanly — no speculative compat shims. Before a migrate-everywhere change, grep for
  consumers that need the old form; alias or migrate only the real ones
- After every fix or refactor, re-read the resulting code as a whole, not the diff: fixes create new smells (e.g.
  two classes become structurally identical only after their serializers are unified — merge them)

# Code style
- No imports inside functions/methods; always place imports at the top of the file
- Exception: circular dependencies or cases with no other resolution
- No `hasattr`/`getattr` hacks for type dispatch; use `isinstance` with proper base classes or protocols
- Judge names at the call site (`ds`, `teleop` — not `dataset`, `teleoperation`) and against the roadmap
  (`mujoco_franka`, not `sim` — more sims will exist)
- Before inventing a name or pattern, mirror adjacent code (config placeholders are named `placeholder`; new
  constants join the existing constants block)
- Configuronic: never define a function whose only purpose is to build a config — decorate it with `@cfn.config`
  directly. When wrapping a class or function usable as-is (without configuronic), assign `NAME = cfn.Config(Thing)`
  rather than writing a wrapper. Define variants with `.override`
- Linter suppressions (`noqa`, `type: ignore`, `pyright: ignore`) must be narrowly scoped to a specific rule and
  suppress an error that actually fires — no blanket or speculative suppressions

# Writing
- Write every text that a person reads in **Simplified Technical English** (ASD-STE100). This covers
  comments, docstrings, commit messages, pull requests, review replies, issues and documents
- Give each word one meaning and one part of speech. Use the same word for the same thing every time. Do not
  use a synonym for variety
- Choose the shortest common word that carries the meaning. Write the articles. Put not more than three nouns
  together
- Write not more than 20 words in an instruction. Write not more than 25 words in a description. Write one
  instruction in one sentence
- Use the active voice. Use the present tense for a fact or a state, and the past tense for an event that
  happened. Start an instruction with its verb
- Write no idiom, no metaphor, no joke and no rhetorical question
- Code, identifiers, configuration values and log lines are out of scope. Quote text from a source as written
- The standard is at asd-ste100.org, free for non-commercial use. Part 2 is a dictionary of approved words.
  Apply the shortest-common-word rule where you cannot check it
- A caveat, a correction and a mark on an unverified claim all stay. Write each one in short sentences. Do not
  drop one to meet a word count

# Comments & docstrings
- Write for a colleague who knows this codebase but not the thing you are documenting: don't explain the domain,
  the neighbouring modules, or vocabulary the repo already uses
- No references to past or future state ("no longer", "previously", "step N", "today"). Future work is a TODO,
  nothing else
- Never write a comment justifying awkward code — the urge to justify is the signal to fix the design or mark it
  HACK/TODO
- A docstring states what the thing is, and nothing else. Behaviour worth pinning down belongs in a test name,
  a design decision in a TODO or a commit message, a usage example once in the module docstring
- Leave a comment only for what the code cannot say: an invariant, a constraint from outside the file, a reason
  the obvious alternative fails. Dry and factual. If it restates the code, delete it
- Be economical: a class docstring runs a few lines, a method's one, and most methods need none. Past that it is
  teaching rather than stating — cut it, or move the content to the home named above
- Comments wrap at 120 columns, same as code
- No empty intensifiers, in comments or in docs — "honestly", "load-bearing", "genuinely", "truly", "precisely".
  Delete the word; if the sentence still means the same thing, it was never doing any work

# Testing
- A test goes in the file that already covers its subject. A test written for a bug is an ordinary test of the
  thing that broke — it belongs beside the others and needs neither a file of its own nor "regression" framing
- Look for that home by subject rather than by filename, and conclude there is none only after looking. A new
  test file claims the subject has no home yet, and that claim is usually wrong
- When a test has nowhere to go, the file organization is what needs adjusting — that is a finding,
  not a licence to invent a home. Don't restructure on your own initiative: put the test in the least-wrong
  existing home and say plainly that nothing fit, so the gap reaches a human instead of being absorbed

# Commit messages
- Short, imperative sentences (e.g., "Fix wrong type", not "Fixed wrong type")
- Use backticks for code references (e.g., "Fix `RemoteDataset` connection leak")
- No trailing period for short messages
- No Claude/AI attribution
- Never amend commits; always create new commits
- Never use `--no-gpg-sign` or `--no-verify` — commits must be signed

# Infrastructure
- Machines, Docker contexts and images: `docker/CONTEXTS.md`
- Model-specific workflows: `positronic/vendors/{lerobot,gr00t,openpi}/README.md`
- Inference serving, and the adapter/codec/wire-client separation of responsibilities (read BEFORE
  writing a sim/rig adapter): `positronic/offboard/README.md`
- Reconstructing previous runs: read `run_metadata_*.yaml` and episode `static.json` from output directory
