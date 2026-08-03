# Code Rules

Judgment rules for this repository — the checks a linter cannot make.

Every rule has an **id** — its heading. Cite it when reporting, fixing, or waiving a violation. Add and
recalibrate rules through the `add-rule` skill.

## Waiving a rule

A violation is waived when the offending line or its enclosing block carries:

```python
# rules-allow: <rule-id> — <reason>
```

The reason must say why this instance is correct. One rule id, at the site — as narrow as a `noqa`.

## Rules

### caller-in-name

Don't name anything — function, class, module, variable — after where it is used. Name it after what it
does. Information about the callers must not leak into the name.

A helper that recomposes a pose through a fixed transform is `change_frame`, not `to_policy_frame`: it
has never heard of a policy.

### hardcoded-keys

Don't write the names of shared data as bare literals — dict keys, signal names, field paths. The name
is a contract between whoever writes the data and whoever reads it, and a literal leaves that contract
invisible and repeated.

Two remedies, by who owns the name. When callers may legitimately use different names, take it as a
parameter with a default for the usual one. When everyone must agree on the same name, define it once
as a named constant in a shared module.

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
