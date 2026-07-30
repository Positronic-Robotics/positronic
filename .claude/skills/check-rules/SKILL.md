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
BASE=$(git merge-base HEAD upstream/main 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null)
[ -n "$BASE" ] || { echo 'no merge base — fetch, deepen a shallow clone, or name the base commit'; exit 1; }

git diff "$BASE" HEAD                                      # committed — what a push sends
git diff "$BASE" --cached                                  # index — what a plain commit records
git diff "$BASE"                                           # worktree — what you see in the files
git status --porcelain --untracked-files=all | grep '^??'  # untracked — what no diff shows
```

Git keeps three snapshots — HEAD, the index, the worktree — and a violation can sit in any one of them
while the others look clean: staged and then reverted, or committed and then fixed only in the index.
Check all three, plus untracked files. That set is closed; there is no fourth place for code to hide.

These commands produce patches, not summaries. The agents need the patch text: when the snapshots
disagree, the violating lines exist in one of them and nowhere in the files on disk, so an agent that
only reads the repository cannot see them. Add `--stat` separately if you want a file list for your own
report.

In the ordinary case the three agree and it is one patch. Pass the untracked files' contents alongside
it — a brand-new file is where rules bite hardest and is the one thing no diff shows.

The base is a merge base against a remote-tracking ref, never local `main` — that one goes stale the
moment someone merges upstream and would quietly widen the scope to changes already on the mainline.
One emptiness check covers every way this can fail: ref missing, ancestry not fetched, shallow clone.
`BASE` comes out empty and the run stops, rather than the expansion vanishing and each command
degrading into a diff against HEAD that reports clean.

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
