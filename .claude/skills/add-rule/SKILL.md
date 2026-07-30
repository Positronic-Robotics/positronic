---
name: add-rule
description: Author, calibrate, or retire a rule in CODE_RULES.md. Use when a review comment or a recurring code smell should become a durable repo rule, or when an existing rule is noisy, never fires, or keeps getting waived.
---

# Add Rule

`CODE_RULES.md` is read by the Codex reviewer on every pull request, by Claude while implementing, and
by `/check-rules`. Every rule in it spends attention on every review, so each one earns its place or
comes out. This skill writes one, calibrates it against the repo, or retires one.

Nothing is written to `CODE_RULES.md` without the user's explicit approval of the final wording.

## What a rule is

A judgment call no linter can make, stated as an instruction.

A good rule is:

- **Imperative** — "Don't X. Do Y instead." Not a description of the symptom; the reader should not have
  to work out what to do.
- **Explicit about the safe path**, including the exception where one exists.
- **Checkable from a diff alone**, by someone who was not in the conversation that produced it.
- **Durable** — describes behaviour, not the names of functions, files, or classes that exist today.
- **Consequential** — worth someone stopping their work to fix.

Not a rule:

- Anything ruff or basedpyright can enforce. That belongs in `pyproject.toml`, where it runs
  deterministically on every commit.
- An aspiration — "keep interfaces clean", "name things well". Unfalsifiable, so it fires at random.
- A symptom with no instruction attached.
- Something so broad that most files violate it.

## Step 1: Get the incident

Ask what real code triggered this, and read it — the review comment, the commit, the function. A rule
with no incident behind it is a preference; say so and stop.

The incident is also the source of the bad/good example pair in Step 3, so capture it before it's lost.

## Step 2: Rule out the cheaper homes

Two checks, both of which kill the rule if they hit:

```bash
sed -n '/\[tool.ruff/,/^\[tool\.setuptools/p' pyproject.toml   # is it already a lint rule?
grep -n '^### ' CODE_RULES.md                                  # does an existing rule cover it?
```

If ruff or basedpyright could catch it, enable the check there instead and stop. If an existing rule
covers it, sharpen that rule rather than adding a second one — overlapping rules make findings
ambiguous about which one was violated.

## Step 3: Draft it

```markdown
### <rule-id>

Don't <the thing>. <Do this instead>.

Exception: <where it is legitimately fine>.
```

Then a bad/good code pair whenever the rule is about code shape rather than a naming or comment habit.
Keep both halves to a few lines and take them from the Step 1 incident, stripped of anything
repo-specific that will rot.

`<rule-id>` is kebab-case and names the violation, not the remedy — it appears in review comments as
`Rule <rule-id> violated:` and in waivers as `# rules-allow: <rule-id> — <reason>`, so it must read as
the thing that went wrong.

## Step 4: Cold-read test

The rule will be applied by readers with no access to this conversation. Test that directly: hand a
fresh subagent **only** the draft rule and two unlabelled code snippets — one violating, one clean —
and ask which violates it and why.

If it picks wrong, or its reasoning is not the reasoning you intended, the rule is unclear. Reword and
repeat. Do not proceed on a rule that needed the conversation to be understood.

## Step 5: Calibrate against the repo

A rule that has never fired is a guess. Scan the existing tree — not just the recent diff — for code
the rule would flag, then **read a sample of the hits** (five, or all of them if fewer) and judge each
one yourself.

The count sizes the sample; the sample decides. Report both, and the fork they imply:

- **Hits you agree with, few.** Calibrated. Land it.
- **Hits you agree with, many.** The rule is right and the repo has a backlog. Land it, and raise
  separately whether to clean up now or let it bite on future changes — a rule that flags a lot of
  existing code will make the next few reviews noisy, and the user should know that going in.
- **Hits you disagree with.** The rule is too broad. Narrow it, or drop it.

Report the sample honestly, including hits that make the rule look bad.

## Step 6: Approve and append

Show the final wording, the id, the hit count, and the sample. On explicit approval, append the rule to
`CODE_RULES.md` in its section, then commit following the repo's conventions.

## Retiring a rule

Rules only accumulate unless something removes them, and a stale rule dilutes attention across the
whole file. Retire one when:

- it has not fired in review for a long stretch, and a scan of the tree finds nothing it would catch;
- it is waived more often than it is obeyed — count the `# rules-allow:` markers carrying its id:

  ```bash
  grep -rn 'rules-allow: <rule-id>' --include='*.py' .
  ```

  A rule with several live waivers is describing an exception as if it were the norm. Either the rule
  is wrong or the exception is the real rule; say which.
- a linter can now enforce it. Move it to `pyproject.toml` and delete it here.

Deleting a rule needs the same explicit approval as adding one. Say what it was, why it is going, and
what replaces it if anything does.
