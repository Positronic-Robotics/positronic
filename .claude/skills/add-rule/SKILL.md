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

The incident is also where the rule's example comes from, so capture it before it's lost.

## Step 2: Rule out the cheaper homes

Three checks, each of which kills the rule if it hits: whether ruff or basedpyright already covers it
(`pyproject.toml`), whether an existing rule does (`CODE_RULES.md`), and whether it is general at all.

If ruff or basedpyright could catch it, enable the check there instead and stop. If an existing rule
covers it, sharpen that rule rather than adding a second one — overlapping rules make findings
ambiguous about which one was violated.

**And if it only holds in one repository, it does not go here.** `CODE_RULES.md` is read by every
Positronic repository, so a rule turning on one repository's architecture, layout or tooling spends
every other repository's review attention on something that cannot apply to it — and fires there as
a false positive, which is how a rule set stops being read. Write it in that repository's own
architecture doc or `AGENTS.md`, where its subject lives, and say that is where it went.

The test: strip the names of the modules, directories and tools it mentions. If nothing is left, the
rule was about this repository's shape. An *example* from real code is not that — a rule stated
generally and illustrated concretely is the shape to aim for.

## Step 3: Draft it

Read the rules already in `CODE_RULES.md` and write one like them — same shape, same length, same
plainness. Be concise and stay on the essence of the issue.

`<rule-id>` is kebab-case and names the violation, not the remedy — it appears in review comments as
`Rule <rule-id> violated:` and in waivers as `# rules-allow: <rule-id> — <reason>`, so it must read as
the thing that went wrong.

The rule will be applied by people who were not in the conversation that produced it. If it only makes
sense to someone who was, reword it.

## Step 4: Approve and write

Show the final wording and the id. On explicit approval, write it to
`CODE_RULES.md` — a new rule is appended, a revision of an existing id replaces that rule in place, so
no id ever appears twice. Then commit following the repo's conventions.

## Calibrating a noisy rule

Run this when a rule starts producing findings people argue with — not on every rule that lands.
Search the tree for what it would flag, read a sample of the hits, and judge each one yourself. Hits
you agree with mean the rule is right and the repo has a backlog; hits you disagree with mean it is too
broad and needs narrowing or dropping. Say how large the sample was, so a thin one cannot read as a
clean sweep.
