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
git diff --stat                                   # uncommitted work
git diff $(git merge-base HEAD main)... --stat    # the whole branch
```

Use the uncommitted changes if there are any, the branch diff otherwise, or the files the user named.
Say which scope you used — a clean report over the wrong scope is worse than no report.

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

Each agent returns either the single word `none`, or findings in this form:

```
Rule <rule-id> violated:
<file>:<line>
<what the code does, and the safe path>
```

Instruct every agent to:

- report only what it can point at — a file, a line, and the text that breaks the rule;
- skip code carrying `# rules-allow: <its-rule-id> — <reason>` on the line or its enclosing block, and
  list what it skipped;
- answer `none` plainly rather than reach for a marginal finding. A checker that always finds something
  gets ignored, which costs more than the violation it caught.

## Step 3: Report

Aggregate every agent's findings into **one list**, ordered by file and line, each entry naming its own
rule and keeping the agents' format so local output matches what Codex posts on the pull request. State
the scope, the rules checked, and every waiver that was honoured — a rule silenced by a waiver must
never look the same as a rule that passed.

Report only. Fixing is a separate decision and the user makes it.

## When a finding is wrong

A false positive is information about the rule, not just about the code. If a rule produces findings
you disagree with, take it to the `add-rule` skill and recalibrate or reword it there. Arguing with the
same rule twice means the rule is broken.
