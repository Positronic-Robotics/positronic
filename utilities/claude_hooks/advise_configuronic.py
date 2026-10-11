#!/usr/bin/env python3
"""Advise once per Claude Code session when a Python edit adds ``cfn.Config(``.

The trigger compares literal occurrences in Edit/MultiEdit replacements or a Write's before/after
contents. Reads and shell commands are outside its scope. The session marker uses exclusive creation
so concurrent hook calls emit one reminder. Errors are logged without blocking the tool.
"""

import hashlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import hook_payload

logger = logging.getLogger(__name__)

CALL = 'cfn.Config('
ADVICE = (
    "Before adding cfn.Config(...), read Config's implementation/docstring and check whether deferred "
    'construction is needed. Pass plain settings as values; use @cfn.config for factories you define '
    "and .override() for variants. Reuse Configuronic's override and copying behavior. "
    'Guide: https://github.com/Positronic-Robotics/configuronic#readme'
)


def introduces_config(payload: dict) -> bool:
    target = hook_payload.target_path(payload)
    if not target or Path(target).suffix != '.py':
        return False
    arguments = hook_payload.tool_input(payload)
    match hook_payload.tool_name(payload):
        case 'Edit' | 'MultiEdit':
            edits = [arguments] if hook_payload.tool_name(payload) == 'Edit' else arguments.get(hook_payload.EDITS, [])
            return any(
                edit.get(hook_payload.NEW_STRING, '').count(CALL) > edit.get(hook_payload.OLD_STRING, '').count(CALL)
                for edit in edits
            )
        case 'Write':
            path = Path(payload.get(hook_payload.CWD) or Path.cwd()) / target
            try:
                before = path.read_text()
            except FileNotFoundError:
                before = ''
            return arguments.get(hook_payload.CONTENT, '').count(CALL) > before.count(CALL)
        case _:
            return False


def advise(payload: dict) -> None:
    if not isinstance(payload, dict):
        return
    session = payload.get(hook_payload.SESSION_ID)
    if not isinstance(session, str) or not session or not introduces_config(payload):
        return
    session_hash = hashlib.sha256(session.encode()).hexdigest()
    marker = Path(tempfile.gettempdir()) / f'positronic-configuronic-advised-{session_hash}'
    try:
        marker.touch(mode=0o600, exist_ok=False)
    except FileExistsError:
        return
    print(json.dumps(hook_payload.additional_context(ADVICE, hook_payload.PRE_TOOL_USE)))


def main() -> int:
    try:
        advise(json.load(sys.stdin))
    except (json.JSONDecodeError, UnicodeError, OSError) as error:
        logger.error('Configuronic advisory hook failed: %s', error)
    return 0


if __name__ == '__main__':
    sys.exit(main())
