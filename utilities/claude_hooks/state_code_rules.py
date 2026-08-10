#!/usr/bin/env python3
"""Claude Code SessionStart hook: state the opening paragraph of every rule in `CODE_RULES.md`.

The rules govern writing code as much as reviewing it, so they have to reach the model before any
code is written. Only each rule's opening paragraph goes out — its instruction — leaving the
exceptions and examples to be read from the file when a rule is in question. The digest is derived
on every call, so it cannot drift from the rules it states.

Wired in `.claude/settings.json` (SessionStart, every source): prints JSON to stdout, exits 0.
Stdlib-only so it runs without the project venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hook_payload

RULES_FILE = Path(__file__).resolve().parents[2] / 'CODE_RULES.md'
RULE_HEADING = '### '
PARAGRAPH_BREAK = '\n\n'
PREAMBLE = (
    'Rules from CODE_RULES.md, which governs writing code here as much as reviewing it. Each rule below is its '
    'own opening paragraph; the exceptions and examples that complete it are under that heading in the file, '
    'and a rule in question is read there rather than from this summary. Cite the id when you follow, waive or '
    'report one. A rule is waived at the offending line or its enclosing block, and nowhere else:\n\n'
    '    # rules-allow: <rule-id> — <reason this instance is correct>'
)


def digest(rules: str) -> str:
    """Each `### rule-id` heading paired with the paragraph that opens the rule."""
    entries = []
    for section in rules.split(f'\n{RULE_HEADING}')[1:]:
        rule_id, _, body = section.partition('\n')
        opening = body.strip().split(PARAGRAPH_BREAK)[0]
        entries.append(f'{rule_id.strip()}:\n{opening}')
    return PARAGRAPH_BREAK.join(entries)


def main() -> int:
    if not RULES_FILE.is_file():
        return 0
    context = PARAGRAPH_BREAK.join([PREAMBLE, digest(RULES_FILE.read_text())])
    print(json.dumps(hook_payload.additional_context(context, hook_payload.SESSION_START)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
