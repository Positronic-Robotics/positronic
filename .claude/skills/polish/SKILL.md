---
name: polish
description: Autonomous design + style pass over recently changed code — rewrites files and commits the result. Invoke only on an explicit polish request (/polish) or from the push-pr flow; NOT for read-only review or analysis asks, which must not modify the worktree. Reviews touched files as a whole (not diff hunks), fixes everything it finds including refactors, iterates until clean, and reports — design changes first.
---

# Polish: Autonomous Design Review + Style Rewriter

Make the changed code PR-ready without consuming the user's attention. You detect, you fix,
you re-check, you commit, you report. The user reviews the resulting diff — not a list of
questions.

**Operating rules:**

- **Autonomous by default.** Implement every finding, including multi-file refactors (merging
  classes, moving ownership, renaming + migrating). Do NOT present findings and wait.
- **Ask only when** (a) an action is irreversible beyond the working tree — dataset/data
  migrations, deleting files you didn't create whose purpose is unclear, anything touching
  remote state; or (b) two designs are genuinely defensible and externally visible — then ask
  with a recommendation. Everything else: decide, fix, report.
- **Review code as a whole, never as a diff.** The diff only tells you *which* files to look
  at. Read each touched file (and the immediate neighbors it interacts with) in its entirety
  and judge the resulting state. A patch that looks fine as a hunk can leave the file
  incoherent.
- **Iterate to a fixed point.** Fixes create new smells: unify two serializers and two classes
  become structural twins; delete a field and a helper goes dead; move ownership and a
  parameter loses its reason to exist. After each round of fixes, re-run detection on the
  resulting state of the touched code. Stop only when a round produces no findings.
- **Sweep for siblings.** Every time you fix an instance of a smell, search the rest of the
  changed code (and its neighborhood) for the same pattern before moving on.
- **Commit the result.** When the pass finishes clean, commit its changes following the
  repo's commit conventions (no AI attribution). The user reviews the result on GitHub and
  comments on anything they dislike.

## Step 1: Scope

```bash
git diff HEAD --stat            # uncommitted changes (staged + unstaged)
```

If empty, use the branch diff against the merge base with the default branch
(`git diff $(git merge-base HEAD main)... --stat`), or `git diff HEAD~1 --stat` outside a
feature branch. If the user named files, restrict to those.

The output is a list of **touched files**. From here on, work with whole files: read each
touched file fully, plus the modules it tightly interacts with (callers/callees of changed
interfaces).

For large scopes (roughly >400 changed lines or >6 files), fan the detection out to parallel
subagents — one each for design integrity, duplication, and comments/naming — each returning
a findings list with file:line and a one-line rationale. Apply all fixes yourself.

## Step 2: Design integrity

The expensive smells. Judge the *resulting state* of each touched module against these:

**Invariants & ownership**

- **Half-committed invariant.** The change has a design thesis ("X owns Y", "Z is
  name-free") but old pathways survive: a constructor still accepts what is now owned
  internally, a base class exposes a mutator only one orchestrator should call, a flag
  coexists with the parameter it replaced, a component still drives state that the thesis
  moved elsewhere. Enumerate the thesis's consequences for every touched interface and kill
  each leftover pathway.
- **Two owners for one value.** The same value threads through two layers (a config plumbs
  `task.timeout` into a driver while the consumer also holds the `Task`). Move ownership to
  the right layer and delete the redundant path — don't feed the new source into the old
  parameter.
- **Silent bridge.** A field, alias, or indirection added to avoid confronting a rename or
  migration (a `record_name` field instead of renaming the signal and migrating data).
  Resolve it now, or keep the bridge with a loud HACK/TODO stating what resolves it. Never
  bridge silently with structure.

**Structures that don't earn their place**

- **Concept reification.** A class/file/field minted for a design-doc concept that existing
  structure already encodes (which dict an entry lives in, an enum, the calling context).
  Delete it; let the structure carry the distinction.
- **Structural twins** *(run this check explicitly every round)*: compare the field lists of
  the dataclasses/classes in touched modules. Two that are identical — or one a strict
  subset of the other — get merged, even if they started the session with different
  signatures. This is the canonical emergent smell: it often appears only *after* another
  fix.
- **Always-paired classes.** Two classes always instantiated together and composed — one class.
- **Parallel hierarchies.** Two class trees mirroring each other 1:1, always composed — collapse.
- **Dead composition points.** An ABC method or config parameter with exactly one
  implementation/value codebase-wide is not an extension point.
- **Post-construction mutation.** A factory creates an object then pokes its attributes
  (`result.meta['key'] = value`) — the object should compute this from constructor params.
- **Config function doing too much.** A config/factory computing derived values and injecting
  them — the object owns that logic.

**Placement & boundaries**

- **Leaky abstraction.** Private dict keys, wire formats, vendor structures referenced
  outside their owning module.
- **Code in the wrong layer.** Generic logic (Signal, ndarray, Episode transforms) buried in
  a domain module → move to the shared library. Domain thresholds/heuristics in library code
  → move out. A composition of existing primitives that others would reuse → name it in the
  library.
- **Scattered data construction.** The same dict/tuple literal structure built in 2+ places
  → shared helper.

**Convention mirroring**

- Before accepting any new name, idiom, or pattern in the diff, look for how adjacent code
  does it (the sibling config, the neighboring module, the same file). New code that invents
  where it could mirror is wrong even when it works.
- **Optional that lies.** `X | None` for a value that is logically required weakens the
  contract — make it required, or restructure so the None path doesn't exist.

## Step 3: Style rules

Apply the rulebook below (R1–R9) to the changed code. Mechanical style fixes stay within
diff-touched lines; design fixes from Step 2 go wherever the design requires.

## Step 4: Comments & docstrings sweep

For **every** comment and docstring in the touched files (not just new ones in the diff):

1. **Archeology** — references to past or future state: "no longer", "previously", "used
   to", "today", "for now", "step N", "in a later PR/follow-up", "unifies them". Delete, or
   rewrite as a present-state fact. Future work is a TODO, nothing else. A quick grep helps
   seed the pass, but judge every comment, not just grep hits:
   ```bash
   grep -nE 'no longer|previously|used to|for now|today|step [0-9]|later|follow-?up|will be' <files>
   ```
2. **Justification** — a comment defending awkward code ("bool is an int subclass — check
   identity first", "the sim paces virtual time because..."). The comment is the smell
   marker: fix the design it apologizes for, or convert to an honest HACK/TODO. Never keep
   the apology.
3. **Restatement** — says what the code already says. Delete. If there is nothing to say
   beyond the code, silence is the correct comment.
4. **Width** — re-wrap anything beyond 120 columns.
5. **Dead suppressions** — for each `noqa` in touched code, remove it and check whether
   `ruff check` flags the line; keep only the ones that fire. Apply the same test to
   `# type: ignore` only if the project runs a type checker (mypy/pyright configured) —
   ruff cannot prove a type suppression dead, so without one, leave them alone.

## Step 5: Iterate to a fixed point

Re-run Steps 2–4 on the **resulting state** of the touched code. Repeat until a full round
produces no findings. In practice 2–3 rounds; the second round exists precisely to catch
what the first round's fixes created (structural twins, dead parameters, orphaned helpers,
stale comments describing the pre-fix code).

## Step 6: Verify

```bash
uv run --locked ruff check --fix <files> && uv run --locked ruff format <files>
uv run --locked pytest --no-cov -q
```

Run the full test suite whenever Step 2 changed structure or behavior-adjacent code. Fix
failures you introduced; report pre-existing failures without hiding them.

## Step 7: Report

Lead with what deserves the user's eyes, then compress the rest:

```
## Design changes (review these)
1. Merged `Privileged` into `Observation` — after the serializer unification they were
   structural twins; the observations/privileged dicts already carry the distinction.
2. Moved `timeout` ownership to Harness (reads it from Task); deleted the driver parameter.

## Mechanical fixes
- Comments: removed 4 archeology comments, re-wrapped 3 to 120 cols, dropped 1 dead noqa
- R1: merged second `match` on directive into the existing dispatch
- R3: `partial` instead of lambda in 2 callbacks

## Verification
ruff clean; pytest: 214 passed
```

If something was deliberately left (a defensible bridge kept with a TODO, a pre-existing
failure), say so explicitly.

---

## Style Rules (ranked by priority)

### R1. Control Flow — Minimize branches, preserve reader flow

Code should read top-to-bottom with minimal interruptions. Reduce nesting, avoid
`break`/`continue` where possible, and keep the happy path at the shallowest indentation.

- **`match/case` for multi-way dispatch** — not `if/elif` chains. A `match` is heavyweight —
  **one per value**. Two `match` blocks dispatching on the same variable is a red flag; merge
  them using guards (`case _ if condition:`) for state-dependent behavior. Destructure inline:
  ```python
  # BAD
  if isinstance(command, Reset):
      return {'type': command.TYPE}
  elif isinstance(command, CartesianPosition):
      return {'type': command.TYPE, 'pose': command.pose.as_vector()}
  # GOOD
  match command:
      case Reset():
          return {'type': command.TYPE}
      case CartesianPosition(pose):
          return {'type': command.TYPE, 'pose': pose.as_vector()}
  ```

- **Avoid `continue`** — prefer a positive condition wrapping the body:
  ```python
  # BAD
  for k in keys:
      if k in seen:
          continue
      seen.add(k)
      yield k
  # GOOD
  for k in keys:
      if k not in seen:
          seen.add(k)
          yield k
  ```

- **Avoid `break`** — structure loops to terminate via their condition or `return`.

- **Guard clauses at the top**, then linear happy path:
  ```python
  # BAD
  def process(signal):
      if len(signal) > 0:
          if 'key' in signal:
              return compute(signal)
      return None
  # GOOD
  def process(signal):
      if len(signal) == 0:
          return None
      if 'key' not in signal:
          return None
      return compute(signal)
  ```

- **Keep nesting shallow** (max 3 levels typical, 4 rare). If deeper, extract a helper with a
  distinct responsibility — not just to reduce line count. A multi-branch computation feeding
  one value (`delay = ...` over 3 branches) is a helper whose name replaces the comment that
  would otherwise explain it.

- **Integrate, don't layer.** New behavior should slot into the existing control flow, not
  wrap around it or duplicate it. If adding a feature requires scaffolding — stop and
  restructure the surrounding code so the new behavior lands cleanly. Don't hesitate to
  simplify or reshape the function the diff touches. Red flags to watch for:
  - **Second `match`/`if-elif` on the same value** — merge into one dispatch, use guards for
    state-dependent cases
  - **`try/except` wrapping an existing `try/except`** — slot the new error handling into the
    existing handler
  - **Duplicate `yield`/`return` to support a new `continue`** — restructure with `if/else`
    so the loop has one exit point
  - **New branch that wraps existing code** — find where in the existing `if/else` the new
    case naturally belongs, add it as a sibling rather than a parent
  - **A loop that grew wordy** after gaining a responsibility — restructure the loop body
    (extract the new responsibility into a named helper) instead of accreting branches and
    comments

### R2. Brevity — Fewer lines, same clarity

Always ask: can we reduce the number of lines without sacrificing readability? The audience
is strong engineers — write code that reads like a comment. Comments that explain *why* are
valuable; comments that restate *what* the code does are noise. If code needs a "what"
comment, rewrite the code instead.

- **Inline single-use variables** when the expression is clear:
  ```python
  # BAD
  data = response.json()
  return data['models']
  # GOOD
  return response.json()['models']
  ```

- **`or` for defaulting collections** (but never for values where `0`/`False` is valid):
  ```python
  # BAD
  if meta is None:
      meta = {}
  # GOOD
  meta = meta or {}
  ```

- **Ternary for simple conditional assignments**:
  ```python
  ts = ts if ts >= 0 else self._clock.now_ns()
  ```

- **Comprehensions over accumulator loops** when logic is straightforward:
  ```python
  # BAD
  row = []
  for key in keys:
      row.append(mapping.get(key))
  # GOOD
  row = [mapping.get(key) for key in keys]
  ```

- **Merge duplicate calls** with non-overlapping args:
  ```python
  # BAD
  Derive(started=lambda ep: ...), Derive(uph=uph)
  # GOOD
  Derive(started=lambda ep: ..., uph=uph)
  ```

- **Use `assert` for internal invariants** — contracts between our own modules where a
  failure is a programmer error — not multi-line `if/raise`. Validation of external input
  (user CLI args, network payloads, on-disk data) keeps explicit raises: `python -O` strips
  asserts, so they must never carry production validation:
  ```python
  # BAD
  if a is None and b is None:
      raise ValueError(...)
  if a is not None and b is not None:
      raise ValueError(...)
  # GOOD
  assert (a is None) ^ (b is None), 'Exactly one of a or b must be provided'
  ```

- **Remove dead code**: unused imports, unreachable branches, empty files. Don't leave
  commented-out code.

- **Delete abstractions that add no value**: if a class just wraps a dict, emits two
  commands, or delegates everything to one field — remove it and use the underlying thing
  directly. Two classes that are always instantiated together and always composed — merge
  them into one.

- **Repeated data literals are duplication too.** Dict/tuple literals with the same structure
  appearing in 2+ places (e.g., `{'shape': (dim,), 'dtype': 'float32'}` constructed in every
  codec class) should be extracted into a shared helper. This is the most commonly missed
  form of duplication — look for it explicitly.

- **Deduplicate across the diff.** Look at the changed code as a whole — within each file and
  across all changed files. If new code introduces a class, helper, constant, or pattern that
  already exists (or closely mirrors something nearby), consolidate onto the existing one.
  This applies everywhere: repeated class definitions, near-identical test setups,
  copy-pasted helpers, overlapping test cases where one subsumes another, same multi-line
  pattern appearing in 2+ files. Consolidate autonomously and list it in the report; before a
  consolidation that removes a public form, grep for consumers that need it.

### R3. Modern Python — Use the latest idioms

- **`T | None`** not `Optional[T]`. **`dict[str, Any]`** not `Dict[str, Any]`.
- **`is`/`is not`** for enum and sentinel comparisons, **`isinstance`** for type checks —
  never `hasattr`/`getattr` hacks.
- **`functools.partial`** for callbacks — not lambdas:
  ```python
  # BAD
  transforms = {k: lambda ep, k=k: self._encode(k, ep) for k in keys}
  # GOOD
  transforms = {k: partial(self._encode, k) for k in keys}
  ```
- **`zip(..., strict=True)`** when lengths must match, `strict=False` when mismatch is
  acceptable.
- **`@final`** on public API methods of base classes; subclasses implement `_`-prefixed
  private methods.
- **Keyword-only args** with `*` separator for public APIs:
  ```python
  def __init__(self, base_url: str, *, timeout: float = 30.0):
  ```
- **f-strings** everywhere — no `.format()`, no `%`-style.
- **`tuple()` wrapping** for storing config/init sequences (defensive immutability):
  ```python
  self._transforms = tuple(transforms)
  ```

### R4. Naming — Domain-accurate, no implementation leakage

- Names describe **what it is**, not how it works: `Receiver` not `Reader`,
  `CartesianPosition` not `CartesianMove`.
- **Judge names at the call site**, not the definition site: `ds` and `teleop` read better in
  configs than `dataset` and `teleoperation`. Simulate how the name appears where it is used
  (CLI flags, config overrides, call expressions) and optimize for that.
- **Names must survive the roadmap**: `mujoco_franka`, not `sim` (more sims will exist);
  avoid names that become ambiguous the moment a second variant appears.
- **Mirror adjacent conventions** before inventing: if sibling code names its sentinel
  `placeholder`, the new one is `placeholder` too — not `SENTINEL`.
- Don't shadow built-ins: `message_bytes` not `bytes`.
- Singular module/directory names: `roboarm/` not `roboarms/`.
- Strict `snake_case` for variables and functions.
- Use short loop variable names (`k, v`, `ep`, `ts`) when it keeps expressions on one line.
- **Consistent names across the diff** — the same concept should carry the same variable name
  in every file touched by the diff.

### R5. Trust & Explicitness — No defensive ceremony

- **No defensive `isinstance` checks** at internal boundaries — trust the caller.
- **No guards against problems never observed** — a `noqa` must suppress an error that
  actually fires; an identity check or workaround for a hypothetical needs to go, along with
  the comment explaining it.
- **No `**kwargs` propagation** — name every parameter explicitly.
- **The constructor defines the initialization contract** — if a value is part of what the
  object *is* (configuration, identity), it belongs in the constructor, not set afterwards.
  Reserve direct attribute mutation for truly dynamic state. Knowing the concrete type
  doesn't justify bypassing the constructor:
  ```python
  # BAD — known at construction, but set after
  result = MyDecoder(key='foo')
  result.horizon = horizon
  # GOOD — passed through constructor
  result = MyDecoder(key='foo', horizon=horizon)
  ```
- **No logging in domain logic** — if something is wrong, raise. Logging belongs only at I/O
  boundaries (process lifecycle, network, hardware).
- **No comment banners** (`# === Section ===`) — code structure communicates organization.
- **No speculative abstractions** — every ABC must have 2+ live implementations that differ
  along a real axis.
- **Sentinel objects** (`NODEFAULT = object()`) when `None` is a valid value — don't overload
  `None` to mean "not provided".
- **`None` is valid data** — never silently filter it out or treat it as "missing".

### R6. Code Organization

- **Top-level imports only** — no imports inside functions (except circular deps).
- **Imports ordered**: stdlib > third-party > local, separated by blank lines.
- **Module-level private functions** over nested functions/closures:
  ```python
  # BAD
  def ckpt(ep):
      def _split(path): ...
      parts = _split(raw_path)

  # GOOD
  def _split(path): ...
  def ckpt(ep):
      parts = _split(raw_path)
  ```
- **Standalone functions over methods** when using only public API of the argument.
- **Keep long sequential methods intact** (`# noqa: C901`) when they are state machines or
  control loops — don't split just for line count. Extract only when there is a distinct,
  reusable responsibility.
- **Fewer files**: a new module must earn its existence. Two small concept-modules that are
  always imported together belong in one broader module.
- **Constants live together** — a new constant joins the module's existing constants block,
  not wherever it was first needed.
- **Shallow hierarchies** (1-2 levels max). Composition handles variation — don't build
  `Base → Abstract → Partial → Concrete` chains.
- **Minimal ABCs** (1-4 abstract methods). If an ABC grows, split it into separate concerns.
- **Template method pattern**: `@final` on public API, `_private` abstract methods for
  subclasses. The algorithm is sealed; only the primitive operations are overridden.
- **Dataclasses only for pure data transfer** (messages, commands, metadata). If it has
  methods that operate on state, use a regular class.
- **Class vs function**: use a class when state persists across calls or type dispatch is
  needed. Pure transforms stay as functions.

### R7. Error Handling — Surface, don't hide

- **No `try/except: pass`** — let failures surface.
- **`raise ... from e`** to enrich context, but keep error messages concise:
  ```python
  # BAD
  f'Checkpoint not found: {checkpoint_id}. Available: {available}'
  # GOOD
  f'Checkpoint not found: {checkpoint_id}'
  ```
- **Narrow `try/except` scope** — wrap only the specific operation that might fail, not the
  whole function.
- **No defensive try/except** around code that "might fail" — only catch specific expected
  exceptions at system boundaries.
- **Fail loud at wiring time** — when a misuse can be detected at construction/wiring
  (a stateful serializer shared between two consumers), raise there instead of guarding at
  every use site.

### R8. Formatting — Dense but readable

- **No unnecessary blank lines** inside short methods or between tightly related statements.
- **Single-line dicts/lists** when they fit within line length:
  ```python
  # BAD
  camera_dict={
      'left': 'left_ph',
      'right': 'right_ph',
  },
  # GOOD
  camera_dict={'left': 'left_ph', 'right': 'right_ph'},
  ```
- **Single-quote strings** as the norm. Double quotes only when the string contains single
  quotes.
- **Dict literals** (`{}`), never `dict()` constructor.
- **Iterate directly** over collections — never `for i in range(len(...))`. Use `enumerate`
  when index is needed.

### R9. What NOT to change

- **Don't add** docstrings, comments, or type annotations to code that wasn't part of the
  diff (the comments *sweep* edits existing comments in touched files; it never adds new ones).
- **Don't add** speculative features, "just in case" classes, or defensive abstractions.
- **Don't create** new files (test files, documentation, helpers) unless the diff already
  created them.
- **Don't refactor** unrelated code. But do restructure code affected by the diff when it
  helps new behavior integrate cleanly — see R1 "Integrate, don't layer" and the Step 2
  design fixes, which go wherever the design requires.
- In the **style phase**, don't change logic or behavior. Behavior-affecting restructuring
  belongs to Step 2, where it is deliberate, tested, and reported under "Design changes".
