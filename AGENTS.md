# Positronic

Python-native stack for real-life ML robotics.

## Code Review Rules

Apart from your normal review, read `CODE_RULES.md` in the repository root and check the diff against
every rule in it. The rules are not repeated here — that file is the whole set.

They govern authoring as much as review: follow them when writing code here, not only when reviewing
someone else's. The heading is named for the review case because that is the section name Codex looks
for.

Report each violation as a comment in exactly this form:

```
Rule {name} violated:
{details}
```

`{name}` is the rule's id — its heading in `CODE_RULES.md`. `{details}` names the offending code and
the safe path out of it.

Do not raise a violation when the offending line or its enclosing block carries a
`# rules-allow: <rule-id> — <reason>` comment for that rule.
