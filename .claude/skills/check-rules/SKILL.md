---
name: check-rules
description: Check the current changes against CODE_RULES.md and report violations. Use before pushing, or when asked whether a change breaks the repo's rules. Each rule is checked by an isolated agent with no session context.
---

# Check Rules

Checks a diff against `CODE_RULES.md`. Nothing else — not a code review, not a style pass, not a bug
hunt. A finding that does not name a rule is out of scope, however good it is.

Each rule is checked by a subagent that starts empty — no conversation history, no files this session
has read, no other rule. That much the platform guarantees; the one thing that would undo it is a fork,
which inherits the parent conversation, so never use one here.

What the platform cannot guarantee is the prompt you write, and that is the only channel into the agent.
You are the worst possible author for it: you know why every line looks the way it does, and that
knowledge is what stops you seeing the violation. Whatever of it you write into the prompt,
the agent inherits.

## Step 1: Scope the diff

The caller owns the scope. Check whatever they name — a branch, a commit range, a set of files — and
nothing else.

With nothing named, check the whole change as it would land if merged as it stands. Work out what that
means here; it is a judgement about this checkout, not a fixed recipe.

Say which scope you used. A clean report over the wrong scope is worse than no report.

## Step 2: One agent per rule

```bash
grep -n '^### ' CODE_RULES.md
```

Spawn one agent per rule, in parallel. Its prompt contains exactly two things:

1. the full text of that one rule, verbatim from `CODE_RULES.md`;
2. the Step 1 change, as patch text.

Nothing else goes in: not the other rules, not what the change was for, not what you expect it to find,
not why the code looks the way it does. The agent reads the repository itself for anything more.

One agent per rule, not one agent holding all of them — an agent given nine rules skims for nine and
checks none. A narrow agent also fails visibly: reporting nothing is one rule cleanly checked, not
nine rules half-checked.

Each agent returns two sections, either of which may be empty:

```
FINDINGS
Rule <rule-id> violated:
<file>:<line>
<what the code does, and the safe path>

WAIVERS
<file>:<line> — <the reason the waiver gives>
```

Instruct every agent to:

- give every violation a file and a line, best effort — a finding with nowhere to point is not a finding;
- honour `# rules-allow: <its-rule-id> — <reason>` on the line or its enclosing block, and list every
  waiver it honoured. A rule whose every match is waived returns an empty `FINDINGS` and a populated
  `WAIVERS`, which is not the same as passing;
- leave `FINDINGS` empty rather than reach for a marginal one. A checker that always finds something
  gets ignored, which costs more than the violation it caught.

## Step 3: Report

Aggregate every agent's `FINDINGS` into **one list**, ordered by file and line, each entry keeping its
own rule and the agents' format, so local output matches what Codex posts on the PR. State
the scope and the rules checked, then list the collected `WAIVERS` separately — a rule silenced by a
waiver must never look the same as a rule that passed.

Report only. Fixing is a separate decision and the user makes it.

## When a finding is wrong

A false positive is information about the rule, not just about the code. If a rule produces findings
you disagree with, take it to the `add-rule` skill and recalibrate or reword it there. Arguing with the
same rule twice means the rule is broken.
