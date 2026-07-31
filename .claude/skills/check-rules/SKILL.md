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

Collect the patches for everything not yet on the mainline: HEAD, the index, the worktree, and the
contents of untracked files. A violation can sit in any one of them while the others look clean —
staged and then reverted, or committed and then fixed only in the index — and that set is closed, so
there is no fifth place for code to hide.

Take the base from a merge base against a remote-tracking ref, never local `main`, which goes stale the
moment someone merges upstream. Stop if there is no merge base rather than falling back to a comparison
against HEAD, which reports clean while checking nothing.

Narrow to the files the user named if they named any, and say which scope you used. A clean report over
the wrong scope is worse than no report.

## Step 2: One agent per rule

```bash
grep -n '^### ' CODE_RULES.md
```

Spawn one agent per rule, in parallel. Its prompt contains exactly two things:

1. the full text of that one rule, verbatim from `CODE_RULES.md`;
2. the Step 1 patches, plus the contents of any untracked files.

Nothing else goes in: not the other rules, not what the change was for, not what you expect it to find,
not why the code looks the way it does. The agent reads the repository itself for anything more.

One agent per rule, not one agent holding all of them — an agent given nine rules skims for nine and
checks none. A narrow agent also fails visibly: reporting nothing is one rule cleanly checked, not
nine rules half-checked.

Each agent returns two sections, either of which may be empty:

```
FINDINGS
Rule <rule-id> violated:
<snapshot> <file>:<line>
<what the code does, and the safe path>

WAIVERS
<snapshot> <file>:<line> — <the reason the waiver gives>
```

`<snapshot>` is `HEAD`, `index`, `worktree`, or `untracked` — which of the four the violating text lives
in. Without it a finding against HEAD points at a line the reader opens and finds already clean, with
nothing to say where the fix is owed.

Instruct every agent to:

- report only what it can point at — a file, a line, and the text that breaks the rule;
- honour `# rules-allow: <its-rule-id> — <reason>` on the line or its enclosing block, and list every
  waiver it honoured. A rule whose every match is waived returns an empty `FINDINGS` and a populated
  `WAIVERS`, which is not the same as passing;
- report the violation anyway when the waiver's reason does not say why that instance is correct.
  `# rules-allow: hardcoded-keys — temporary` explains nothing and would silence a real finding;
- leave `FINDINGS` empty rather than reach for a marginal one. A checker that always finds something
  gets ignored, which costs more than the violation it caught.

## Step 3: Report

Aggregate every agent's `FINDINGS` into **one list**, ordered by file and line, each entry keeping its
own rule and snapshot and the agents' format, so local output matches what Codex posts on the PR. State
the scope and the rules checked, then list the collected `WAIVERS` separately — a rule silenced by a
waiver must never look the same as a rule that passed.

Report only. Fixing is a separate decision and the user makes it.

## When a finding is wrong

A false positive is information about the rule, not just about the code. If a rule produces findings
you disagree with, take it to the `add-rule` skill and recalibrate or reword it there. Arguing with the
same rule twice means the rule is broken.
