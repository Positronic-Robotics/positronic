#!/usr/bin/env python3
"""Claude Code PreToolUse hook: advise the `add-rule` skill on a write to `CODE_RULES.md`.

Advisory, never a refusal: the skill's own final step is a write to that file. The advice goes out
as `additionalContext` because a PreToolUse hook's plain stdout reaches the debug log and nothing
else.

Wired in `.claude/settings.json` (PreToolUse, matcher Bash|Edit|Write|MultiEdit): reads the hook
payload on stdin, prints JSON to stdout, exits 0. Stdlib-only so it runs without the project venv.
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
    """Whether this tool call could write the rules file.

    A shell command is matched on the file's name appearing anywhere in it, since a write reaches
    the file through `cat >`, `tee`, `sed -i`, a heredoc or a python one-liner, and reading the
    shell well enough to tell those apart is a parser this does not need. Advice on a command that
    merely mentions the file costs a line; missing the way an agent most often writes one does not.
    """
    if os.path.basename(hook_payload.target_path(payload)) == RULES_FILE:
        return True
    return RULES_FILE in hook_payload.command(payload)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if advises(payload):
        print(json.dumps(hook_payload.additional_context(ADVICE, hook_payload.PRE_TOOL_USE)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
