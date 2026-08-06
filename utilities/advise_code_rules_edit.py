#!/usr/bin/env python3
"""Claude Code PreToolUse hook: advise the `add-rule` skill on a write to `CODE_RULES.md`.

Advisory, never a refusal: the skill's own final step is a write to that file. The advice goes out
as `additionalContext` because a PreToolUse hook's plain stdout reaches the debug log and nothing
else.

Wired in `.claude/settings.json` (PreToolUse, matcher Edit|Write|MultiEdit): reads the hook payload
on stdin, prints JSON to stdout, exits 0. Stdlib-only so it runs without the project venv.
"""

from __future__ import annotations

import json
import os
import sys

import hook_payload

RULES_FILE = 'CODE_RULES.md'
ADVICE = (
    f'{RULES_FILE} is edited through the `add-rule` skill, not by hand: it asks what code triggered'
    ' the change, rules out a Ruff check or an existing rule covering it, and shows the wording for'
    ' approval before writing. Run the skill unless you are already inside it.'
)


def advises(payload: dict) -> bool:
    """Whether this tool call writes the rules file."""
    return os.path.basename(hook_payload.target_path(payload)) == RULES_FILE


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if advises(payload):
        print(json.dumps(hook_payload.additional_context(ADVICE)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
