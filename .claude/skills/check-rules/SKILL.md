---
name: check-rules
description: Check the current changes against CODE_RULES.md and report violations. Use before pushing, or when asked whether a change breaks the repo's rules. Each rule is checked by an isolated agent with no session context.
---

# Check Rules

Checks a diff against `CODE_RULES.md`. Nothing else — not a code review, not a style pass, not a bug
hunt. A finding that does not name a rule is out of scope, however good it is.

The check runs in fresh subagents with no access to this conversation. That isolation is the point: the
session that wrote the code knows why every line looks the way it does, and that knowledge is exactly
what stops it from seeing a violation.

## Step 1: Scope the diff

```bash
BASE=$(git merge-base HEAD main)
git diff $BASE --stat                                      # committed, staged and unstaged
git status --porcelain --untracked-files=all | grep '^??'  # untracked files, which no diff shows
```

Everything not yet on `main` is in scope: committed, staged, unstaged, and untracked. Pass the untracked
files' contents alongside the diff — a brand-new file is where rules bite hardest and is the one thing
`git diff` cannot show.

Narrow to the files the user named if they named any, and say which scope you used. A clean report over
the wrong scope is worse than no report.

## Step 2: One agent per rule

```bash
grep -n '^### ' CODE_RULES.md
```

Spawn one agent per rule, in parallel. Each gets exactly: the full text of **its own rule and no
other**, the diff, and the repository to read around the change. Don't tell it what the other rules
say, what the change was for, or what you expect it to find.

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

- report only what it can point at — a file, a line, and the text that breaks the rule;
- honour `# rules-allow: <its-rule-id> — <reason>` on the line or its enclosing block, and list every
  waiver it honoured. A rule whose every match is waived returns an empty `FINDINGS` and a populated
  `WAIVERS`, which is not the same as passing;
- report the violation anyway when the waiver's reason does not say why that instance is correct.
  `# rules-allow: hardcoded-keys — temporary` explains nothing and would silence a real finding;
- leave `FINDINGS` empty rather than reach for a marginal one. A checker that always finds something
  gets ignored, which costs more than the violation it caught.

## Step 3: Report

Aggregate every agent's `FINDINGS` into **one list**, ordered by file and line, each entry naming its own
rule and keeping the agents' format so local output matches what Codex posts on the pull request. State
the scope and the rules checked, then list the collected `WAIVERS` separately — a rule silenced by a
waiver must never look the same as a rule that passed.

Report only. Fixing is a separate decision and the user makes it.

## When a finding is wrong

A false positive is information about the rule, not just about the code. If a rule produces findings
you disagree with, take it to the `add-rule` skill and recalibrate or reword it there. Arguing with the
same rule twice means the rule is broken.
