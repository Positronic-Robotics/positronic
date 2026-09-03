# Code Rules

Judgment rules for Positronic code — the checks a linter cannot make. They hold in every Positronic
repository, not only this one. This file is the only copy: a repository that carries no
`CODE_RULES.md` of its own is checked against this one, read from `main`.

So every rule here is general. One that turns on a single repository's architecture, layout or
tooling is not a code rule and belongs in that repository's own docs, beside its subject. An
*example* drawn from this repository is not that, and stays — a rule is easier to apply from real
code than from an abstraction.

Every rule has an **id** — its heading. Cite it when reporting, fixing, or waiving a violation. Add and
recalibrate rules through the `add-rule` skill.

## Waiving a rule

A violation is waived when the offending line or its enclosing block carries:

```python
# rules-allow: <rule-id> — <reason>
```

The reason must say why this instance is correct. One rule id, at the site — as narrow as a `noqa`.

## Rules

### misleading-name

Name anything — function, class, module, variable — for what it is and does. A name describing where it
is used misleads, and information about the callers must not leak into it. So does one describing the
activity performed when what comes back is a `bool`: there the name carries the question that `bool`
answers, so `if f(...):` reads as the claim being tested. A function that also acts is the exception —
it keeps its action name and returns an enum whose members carry the answer, because renaming it for its
verdict hides what it did.

A helper that recomposes a pose through a fixed transform is `change_frame`, not `to_policy_frame`: it
has never heard of a policy.

An activity name over a `bool` leaves the verdict invisible at the call site, so a reader takes it on
trust until they open the definition, and takes it backwards wherever the sense inverts. The tell is a
docstring sentence of the form "True means …"; once the name carries it, that sentence has nothing left
to say.

```python
# Bad — a query named for the activity, and a command renamed until its action disappeared
if evaluate(path, run_id): ...
if idle_ends_the_run(msg, clock): ...

# Good
if asks_this_run_to_finish(path, run_id): ...
if take_idle_round(msg, clock) is IdleRoundOutcome.ENDS_THE_RUN: ...
```

The same enum answers the case where the verdict was never binary — a status, one of several endings.
There the `bool` was the mistake rather than the name (`primitive-type`).

### hardcoded-keys

Don't write the names of shared data as bare literals — dict keys, signal names, field paths. The name
is a contract between whoever writes the data and whoever reads it, and a literal leaves that contract
invisible and repeated.

Two remedies, by who owns the name. When callers may legitimately use different names, take it as a
parameter with a default for the usual one. When everyone must agree on the same name, define it once
as a named constant in a shared module.

The same holds for any literal two pieces of code must spell identically — an environment-variable
name, a filename two processes agree on, a wire field. Hoist it at the first duplication, into the module every
consumer can import, usually the most constrained of them. Not a third copy neither side imports —
and where there is no shared module at all, as across languages, name it once on each side, beside
the code that parses it.

Exception: a name the component itself owns and defines, rather than one it reads from its input.

```python
# Bad
return {**data, 'ee_pose': change_frame(data['ee_pose'])}

# Good — pose_key is a constructor parameter defaulting to 'ee_pose'
return {**data, self._pose_key: change_frame(data[self._pose_key])}
```

### overspecific

Don't bake assumptions about how a component will be used into its interface. Before adding a
parameter, a branch, or a category to a signature, ask what else the component could legitimately be
asked to do — many keys instead of one, several value types instead of one, the same operation in
reverse.

Handling those uniformly usually comes out simpler and easier for a human to read: fewer branches,
often fewer lines, and every use falling into place without deliberation. When it does, the specific
version was wrong.

Generalise until it simplifies, and stop there. Past that point the signature stops having an opinion
and the caller must read the implementation to know what to pass — the same mistake pointed the other
way. This is a matter of taste and does not reduce to a measurement: judge the result as a reader, not
by counting.

Widening what a component accepts is not the same as weakening how its values are typed. Sharpening a
type is `primitive-type` and is never overspecification: it constrains what a value may be, not what
the component may be asked to do.

```python
# Bad — assumes every key belongs to exactly one category, and that the categories need different handling
def __init__(self, pose_keys, command_keys): ...

# Good — one list, handled the same way in both directions
def __init__(self, keys): ...
```

### diff-comments

Write a comment for someone reading the file, not for someone reading the change. Read it back knowing
nothing about why the line was touched: if it answers "why did you change this?" rather than "what must
be true here?", delete it.

Two ways it goes wrong. The comment narrates the change — what the code did before, what replaced it,
the reasoning that got there. Or it describes what other code happens to do — what the caller does
next, what a consumer waits for. Neither is anchored: the first only parses for someone who saw the
edit, and nothing keeps the second true.

Each has a near neighbour that stays. A constraint the code is written against is worth one dry line,
including one on why the obvious alternative fails. The test is whether this code would be wrong if the
claim stopped holding: "a `readOnce()` here would take the connection the control loop holds" earns its
place, where "the caller's `finally` halts the control thread" does not.

Write it flat — no intensifiers, no rhetorical build, no clause there only to make the point land.
Usually one line, and often none.

```python
# Bad — narrates what it replaced, at paragraph length, and builds to a flourish
def _reset(self, robot, robot_state, rate_limiter, should_stop):
    """Home the arm, yielding until it arrives. Drive with ``yield from``.

    A generator rather than a blocking call because the waiting has to keep clearing robot
    errors. `set_target_joints` returns as soon as the target is published, and the driver's
    own loop is what notices and clears a reflex — so a wait that parks inside the library
    stops the only thing that can end the move it is waiting for.
    """

# Good
def _reset(self, robot, robot_state, rate_limiter, should_stop):
    """Home the arm, yielding until it arrives. Drive with ``yield from``."""
```

```python
# Bad — describes what the caller happens to do; this code does not depend on it
if should_stop.value:
    return  # shutting down — the caller's `finally` halts the control thread

# Good
if should_stop.value:
    return
```

### hidden-dependency

Don't let one component depend on another without saying so. The dependency is real either way; leaving
it unstated only means a reader cannot see it, and a consumer that does not know about it is silently
wrong rather than broken.

Common forms, not a closed list — judge a change against the sentence above, not against these:

- **Handoff through shared data** — one stage writes a value so that a later one can find it, and
  neither signature mentions the other.
- **Split meaning** — a value cannot be interpreted on its own. A frame *name* needs the model it names
  a site in; a transform between frames needs nothing. Prefer the value that carries the whole fact.
- **Order dependence** — a step is correct only if another ran first, and nothing states or enforces it.

State the dependency where the thing is declared: an argument, a constructor parameter, or a single
value that is complete on its own.

Changing data is not this. Converting poses between frames, or letting an absent transform mean
identity, transforms the data without hiding anything — the smell is a dependency a reader cannot see,
not a change of representation.

```python
# Bad — b() is correct only if a() ran, and only if b remembers to look
def a(data): data['scale'] = 2.0; return data
def b(data): return data['values'] * data.get('scale', 1.0)

# Good — the dependency is in the signature
def b(values, scale: float): return values * scale
```

### stranded-definition

Don't leave a definition far from the code that uses it. Put it directly above its first user, or
inside the single entity that uses it. Distance costs the reader a search in both directions: from the
use, to find out what it does; from the definition, to find out why it exists.

A private name has every user in the file, so its place is determined. With a single user, first ask
whether the name is worth keeping at all (`earn-its-place`); this rule only decides where a name that
is worth keeping goes. Not touching `self` is no reason to stay at module level — that is what
`@staticmethod` is for.

A public name is looser: most of its users are elsewhere, and a module may order its surface
deliberately. Group it with its in-file callers where that ordering does not say otherwise. Module
scope is right either way once several entities in the file use it.

```python
# Bad — defined at the top of the file, its one caller 400 lines below
def _as_transform(value):
    if isinstance(value, Transform3D):
        return value
    return Transform3D.from_vector(np.asarray(value), QUAT)

# Good — in the constructor that wanted it
if not isinstance(transform, Transform3D):
    transform = Transform3D.from_vector(np.asarray(transform), QUAT)
```
### primitive-type

Don't leave a value typed by its representation when its domain has a type of its own: a filesystem
path is a `Path`, a bounded set of strings is an enum, a return a caller can only read by position is
a named struct. The primitive carries none of the domain's meaning or operations, so every consumer
re-derives what the value holds.

Some domains own a primitive, and then the primitive is the specific type. This is about a
representation standing in for a domain, not about preferring the richer-looking type. A pair
unpacked where it is called, each element typed, re-derives nothing either: a struct over
`(emitter, receiver)` would restate the call site (`earn-its-place`).

An enum is the case that can go either way. If this code dispatches on the set — a `match`, an `if`
ladder against literals — the set is closed and the type only says so; a value this code passes
through untouched is not yours to close. Once it is an enum, iterate it rather than restating its
members.

Convert once, where the value enters: a CLI token, an environment variable, a wire field. A union of
a type with its own string form is that conversion left undone, pushed onto every consumer. Don't
let the annotation lie — where a framework hands the value through uncoerced, keep the honest `str`
and convert immediately inside, or fix the framework.

```python
# Bad
def record(recording_dir: str | Path): ...

if state == 'OPEN': ...
elif state == 'CLOSED': ...

# Good
def record(recording_dir: Path): ...

if state is GripperState.OPEN: ...
```

### optional-lie

Don't type a value `X | None` when it is never `None`. The annotation says `None` is a case this code
handles, so every consumer writes the guard — and the guard that matters is then indistinguishable
from the ones that are dead.

Where the value really is sometimes absent, say when it is: a default in the signature, or a
distinct state. `None` meaning both "not supplied" and "not applicable" is the same lie one level
down.

### swallowed-error

Don't catch an exception so the code can carry on, and don't fall back to a second path when the first
comes back empty. Let the failure surface: a pipeline that silently yields nothing is harder to
diagnose than one that raises, and a fallback becomes the path everything quietly runs through.

This is about hiding a failure, and neither an exception nor an absence is always one. Where a
contract says the value may not be there — `Empty` from a non-blocking `get`, a cache the format
declares optional — reading that and going on is how the contract is used, not a fallback. A retry
that re-raises when it gives up hides nothing either: the failure still reaches the caller.

Where a failure genuinely must not stop the caller — one malformed file in a scan, an optional
feature, a cleanup — catch that specific exception, log it at ERROR with what failed, and say in one
line why continuing is correct. `except Exception: pass` says none of it.

```python
# Bad
try:
    meta = _read_meta(path)
except Exception:
    continue

# Good
try:
    meta = _read_meta(path)
except json.JSONDecodeError:
    logger.error('Skipping %s: malformed episode meta', path)
    continue
```

### earn-its-place

Don't add a class, file, function, field or parameter for a distinction the code already encodes. Check
each of the places one can already live — the dict an entry sits in, an enum member, the calling
context, a module that owns the subject — and use that instead.

The cost is not the lines. Every new type is another thing a reader holds, another place a value can
live, and another edge to keep in sync; a field duplicating what its caller already knows goes stale
the first time only one of the two is set.

With a single user, ask what the name adds over what already stands at that one use. A function whose
body is its name spelled out, a class whose fields the caller holds as locals anyway, a parameter only
ever passed one value — in each the use site already carries the distinction, and the name is a second
place to look. A name earns its place by holding what the use site cannot say: a diagnostic it raises,
an invariant it keeps, a domain predicate that takes more than one clause.

A new distinction lands beside its siblings, in the module that owns the subject. Put it there
without asking. Raise it only when nothing suggests a home: that is a finding about the layout, and
a human decides it. This rule does not apply to a repository still laying out its structure.

### large-indent

Don't let a block grow into a wall of indented code. A long loop or branch body is a function
waiting to be named and extracted, a long `try` is the same with the boundary in the wrong place,
and a whole function body inside a `with` is a decorator. What is left then reads as what it does,
with no enclosing condition to hold in mind.

Keep the `with` where no decorator can be reached: a decorator is evaluated where the function is
defined, so it never sees `self._lock`, and only a `ContextDecorator` works as one. Keep it too
where the `as` value is used, or where the context must wrap the work rather than the call — a
`ContextDecorator` on an `async def` opens and closes while the coroutine is merely being created,
and on a generator it spans the call rather than the `yield`s.

```python
# Bad
def step(self, action):
    with telemetry.span('env.step'):
        ...

# Good
@telemetry.traced('env.step')
def step(self, action):
    ...
```

### stale-doc

Don't write a document against its own past — no "previously", no "this PR", no "as of this writing",
no resolved-while-writing note, no struck-through section kept for context. State what holds now. The
reader has no access to the past state, so prose written against it is noise that ages into a lie.

Don't keep worklogs — status, progress or implementation-summary files. Move what is still true
into the document that owns the subject, and delete the rest. A changelog is the exception: history
is its subject.

Where a change must be referenced, cite something durable — a ticket, a tag, a version. Two things
that look like history and stay: a footgun the reader can still walk into, and a current limitation
with its workaround.

### grandfathered-violation

Don't silence a violation in code you write or touch — fix it. An exception list — a type-check
baseline, a lint allowlist, a suppression file — is for one case: a gate landing on a codebase that
already breaks it, where fixing everything first is impractical. It records the debt standing that
day, and shrinks from there.

So a change may only remove entries. A file you add lands clean, and a file you touch loses the
entries your change covers. The exception is a large refactor, which moves code it did not write:
re-typing all of it would bury the change, so it may leave a touched file's existing entries alone.

A limitation outside the code takes the narrowest suppression the tool offers, at the site, with a
reason: an import that cannot resolve gets `# pyright: ignore[reportMissingImports]`, leaving every
other diagnostic in that file live. Where the gate compares per-file counts rather than entries —
`.basedpyright/baseline.json` is one such baseline — re-anchoring line numbers can absorb a new
finding without it noticing, so check that yourself.

### unchecked-field-name

Don't address a typed object's field by a string the type checker never sees. `obj.field` is
checked, and `getattr(obj, 'field')` raises on a rename. The dict side of a model is neither:
`model_dump(exclude=…)`, `model_copy(update=…)`, `getattr(obj, 'field', default)` and a filter
over `.items()` take names pydantic never compares against the model, so a rename passes every
gate and changes what the code does at run time.

Keep a typed value typed to the end: copy the model and assign the field, which the type checker
reads. Where a set of names is needed, derive it from the type (`model_fields`,
`dataclasses.fields`) or pin it with a test that fails when the model renames or gains a field.

A model that refuses or validates the assignment — frozen, a frozen field, `validate_assignment` —
keeps `model_copy(update=…)`, and a test pins each name to `model_fields`, so a rename fails the
suite instead of the run.

Exception: a name that is the wire — a JSON key two deploys exchange, an environment variable, a
third-party object's attribute. There the string is the contract (`hardcoded-keys`).

```python
# Bad — pydantic checks none of these names, so a rename is silent
data.update({k: v for k, v in ended.model_dump().items() if k not in ('active', 'phase')})
attempt = spec.model_copy(update={'artifact_location': moved})

# Good — the type checker reads the assignment
attempt = spec.model_copy()
attempt.artifact_location = moved
```
