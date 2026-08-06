#!/usr/bin/env python3
"""Claude Code PreToolUse hook: an edit to `CODE_RULES.md` goes through the `add-rule` skill.

The rules file is read on every review in every Positronic repository, so what it says is worth a
process: the skill asks what real code triggered the change, rules out the cheaper homes (a Ruff
check, an existing rule), and shows the wording for approval before writing. Editing the file
directly skips all three, and nothing about the edit itself reveals that it did.

Advisory, never a refusal — the skill's own final step is an edit to this file, so a hook that
blocked would block the remedy it names. Wired in `.claude/settings.json` (PreToolUse, matcher
Edit|Write|MultiEdit): reads the hook payload on stdin, prints to stdout, exits 0. Stdlib-only so
it runs without the project venv.
"""

from __future__ import annotations

import json
import os
import sys

RULES_FILE = 'CODE_RULES.md'
ADVICE = (
    f'{RULES_FILE} is edited through the `add-rule` skill, not by hand: it asks what code triggered'
    ' the change, rules out a Ruff check or an existing rule covering it, and shows the wording for'
    ' approval before writing. Run the skill unless you are already inside it.'
)


def advises(payload: dict) -> bool:
    """Whether this tool call writes the rules file."""
    tool_input = payload.get('tool_input') or {}
    path = tool_input.get('file_path') or tool_input.get('notebook_path') or ''
    return os.path.basename(str(path)) == RULES_FILE


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if advises(payload):
        print(ADVICE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
