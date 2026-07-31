---
name: add-rule
description: Author or recalibrate a rule in CODE_RULES.md. Use when a review comment or a recurring code smell should become a durable repo rule, or when an existing rule produces findings people argue with.
---

# Add Rule

Every rule in `CODE_RULES.md` spends attention on every review, so each one earns its place. This skill
writes one, or recalibrates one that has gone noisy.

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

Two checks, both of which kill the rule if they hit: whether ruff or basedpyright already covers it
(`pyproject.toml`), and whether an existing rule does (`CODE_RULES.md`).

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

The rule will be applied by readers with no access to this conversation. Test that directly with a
subagent, which starts empty unless you fill it in. Give it the draft rule and two unlabelled snippets,
one violating and one clean, and ask which violates it and why. Keep out which is which, the incident
behind the rule, and what you hope it will say.

If it picks wrong, or its reasoning is not the reasoning you intended, the rule is unclear. Reword and
repeat. Do not proceed on a rule that needed the conversation to be understood.

## Step 5: Approve and write

Show the final wording and the id. On explicit approval, write it to
`CODE_RULES.md` — a new rule is appended, a revision of an existing id replaces that rule in place, so
no id ever appears twice. Then commit following the repo's conventions.

## Calibrating a noisy rule

Run this when a rule starts producing findings people argue with — not on every rule that lands.
Search the tree for what it would flag, read a sample of the hits, and judge each one yourself. Hits
you agree with mean the rule is right and the repo has a backlog; hits you disagree with mean it is too
broad and needs narrowing or dropping. Say how large the sample was, so a thin one cannot read as a
clean sweep.
