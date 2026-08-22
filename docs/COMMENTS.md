# Writing comments and docstrings

Each section is one way a comment fails, with the rejected and accepted text taken from real code in
this repository. Read a draft back as someone who has never seen the change: every rule here is a
way that reading goes wrong.

## First ask whether it is needed at all

Most comments should not be written. Take these in order, and only write one when all three fail:

1. **Delete it.** The code already says it. A ternary reading `pinned.theirs if ... else
   logger.level` needs no sentence describing itself.
2. **Change the code instead.** A comment explaining what a value holds is a name that has not been
   chosen yet: `name` meaning a level name in a file where `name` also means a logger name became
   `level_name`, and the comment was no longer needed. A comment labelling a stretch of a function
   is a function waiting to be extracted.
3. **Cut it to one line.** A docstring's first line is usually the whole docstring. A second
   paragraph has to carry something the signature and the body cannot.

✗

```python
"""The number `level_name` stands for in this process, raising when it names no level.

The error names `source`, the variable that carried the name: a threshold that quietly became
something else reads as working configuration.
"""
```

✓

```python
"""The number `level_name` stands for in this process, raising when it names no level."""
```

The cut paragraph argued for a design choice. The signature and the `raise` line already show what
`source` does.

## Say what it is before why it is shaped that way

A reader landing on a definition needs its job first. The constraint on it is the second sentence,
never the first.

✗ `# It is read-only: it takes precedence over init_logging's level, so a value written back would beat the argument of every later call.`

✓ `# The environment variable that sets the threshold for the whole program. It is read-only: it takes precedence over ...`

The module docstring does not cover this. A reader arrives at line 27, not at line 1.

## Use words the repository defines

An undefined term sends the reader looking for a definition that does not exist. Check the term
appears in `ARCHITECTURE.md`, `pimm/README.md`, or the code — otherwise use words that need no
definition.

| ✗ | ✓ | why |
|---|---|---|
| the entry point | the main process | nothing defines "entry point"; "main process" pairs with "subprocess" |
| the operator's own `LOG_LEVEL` | `LOG_LEVEL` | "operator" means the person at the robot console elsewhere in this repo |
| the reading process's registry | `logging.getLevelNamesMapping()` | name the actual call |
| `configure_process_logging` is pimm's own | `configure_process_logging` is called for you | "pimm's own" states a feeling, not a fact |
| `LOG_LEVEL` outranks it | `LOG_LEVEL` takes precedence over it | in a logging module "outranks" reads as severity |

## Use the name a property already has

A described property costs the reader a reconstruction. A named one costs nothing.

✗ `# ... Read, never written: it takes precedence over ...`

✓ `# ... It is read-only: it takes precedence over ...`

"Read, never written" also runs through a negation, and a negation is built before it is discarded:
the reader forms "written", then strikes it out. Say what holds.

## Name the subject

A pronoun or bare noun next to a definition attaches to the wrong thing.

✗ `# A number, not a name: ...` — sitting above `RESOLVED_LOG_LEVEL_ENV = 'PIMM_RESOLVED_LOG_LEVEL'`, which is a name.

✓ `# It carries a number, not a level name: ...`

A word that counts (`both`, `these`, `the two`) or points ahead (`below`, `later`) does the same. The
reader holds a placeholder until they find what it stood for. Name the things.

✗ ``# Both reads test `is not None`, so an empty variable raises rather than falling back to INFO.``

✓ ``# An empty `LOG_LEVEL` or `PIMM_RESOLVED_LOG_LEVEL` raises rather than falling back to INFO.``

## Statement, then consequence

`X, not Y` makes the reader hold a wrong idea for a beat before correcting it.

✗ `# basicConfig sets the root logger's level, not its handler's.`

✓ `# A component logger admits a record on its own level. The root logger's level is not consulted after that, so only its handler's level can still filter the record, and basicConfig leaves that at NOTSET.`

The accepted form is longer. It is also the only one that explains why the loop underneath exists.

## Open with the plain subject

Simple English order: noun, verb, details. Details build on what came before them. A sentence that
opens with a qualified subject or an aside only parses on the second read.

✗ `A process nothing else configures — a spawned control system — would otherwise sit at the stdlib default and drop every line it emits.`

✓ `` `component_levels` holds the per-logger levels the main process had set. A fresh interpreter has none of them, so a subprocess is given them here. ``

## Run forward, not backward

A comment above a block describes that block. Pointing at later code and doubling back makes the
reader hold two positions at once.

✗ `# The pin loop below reads each library logger's current level to work out what the application asked for, so these are set first.`

✓ `# The application's own levels go on first. The pin loop below reads each library logger's current level to work out what it asked for.`

## Put it on the line it explains

A one-clause fact about a single line is a trailing comment on that line.

✗

```python
# NOTSET is 0, so a library nobody set takes the floor
pin = max(level, logging.WARNING, theirs)
```

✓

```python
pin = max(level, logging.WARNING, theirs)  # NOTSET is 0, so a library nobody set takes the floor
```

Rationale for a type's shape goes on the type, not on the variable that happens to hold one.

## One fact, one home

A fact stated in two places goes stale in one of them. Before writing, check the homes the reader
will already have passed: the constant's own comment, the class the value belongs to, the docstring
above the function.

A test is not one of those homes. A fact pinned by a test name stays invisible to someone reading
the code, so state it where it applies and keep it to a line.

## Comment what would otherwise be broken

Keep a comment when deleting or reordering the code would break something silently:

- a loop that looks like a redundant re-set, but is what makes the threshold hold
- two loops whose order matters, because the second reads what the first wrote
- a stdlib constant whose value makes an expression correct (`NOTSET` is `0`, so `max` picks the floor)

Delete it when it narrates. A ternary that reads `pinned.theirs if ... else logger.level` does not
need a sentence saying so.

## A docstring says what comes back, not how it is decided

The body holds the decision. A docstring that walks the same branches ages with them and tells the
caller nothing the signature did not.

✗ ``"""The threshold: the level the main process resolved, else `LOG_LEVEL`, else INFO."""``

✓ `"""The level this process logs at."""`

## Trimming has a floor

Cutting the "what" leaves a reader with a justification for something they cannot identify. When a
comment is down to its constraint alone, it has gone one step too far — restore the sentence naming
the thing.

## Checklist

Before keeping a comment, confirm each:

1. Deleting it, or renaming something, would not serve the reader better.
2. Its first sentence says what the thing does.
3. Every term in it is defined somewhere a reader can reach.
4. Its subject is named, not implied by position.
5. It reads forward: this code, then the reason.
6. The fact is not already in a neighbouring comment or the docstring above.
7. Deleting or reordering the code it guards would break something silently.
