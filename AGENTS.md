# Positronic

Python-native stack for real-life ML robotics: recording, datasets, policy training and on-robot inference.

## Code Review Rules

Apart from your normal review, read `CODE_RULES.md` in the repository root and check the diff against
every rule in it. The rules are not repeated here — that file is the whole set.

Report each violation as a comment in exactly this form:

```
Rule {name} violated:
{details}
```

`{name}` is the rule's id — its heading in `CODE_RULES.md`. `{details}` names the offending code and
the safe path out of it.

Do not raise a violation when the offending line or its enclosing block carries a
`# rules-allow: <rule-id> — <reason>` comment for that rule.
