#!/usr/bin/env python3
"""The field names of the JSON a Claude Code hook is handed on stdin, and readers for them.

The names belong to the harness. Stdlib-only, so a hook that imports this still runs without the
project venv; the harness runs a hook by path, which puts this directory on `sys.path`.
"""

from __future__ import annotations

TOOL_INPUT = 'tool_input'
COMMAND = 'command'
FILE_PATH = 'file_path'
NOTEBOOK_PATH = 'notebook_path'
CWD = 'cwd'
HOOK_SPECIFIC_OUTPUT = 'hookSpecificOutput'
HOOK_EVENT_NAME = 'hookEventName'
ADDITIONAL_CONTEXT = 'additionalContext'
PRE_TOOL_USE = 'PreToolUse'


def command(payload: dict) -> str:
    """The shell command a Bash tool call runs, or '' when the payload carries none."""
    return str((payload.get(TOOL_INPUT) or {}).get(COMMAND) or '')


def target_path(payload: dict) -> str:
    """The file a write-shaped tool call targets, or '' when it names none."""
    tool_input = payload.get(TOOL_INPUT) or {}
    return str(tool_input.get(FILE_PATH) or tool_input.get(NOTEBOOK_PATH) or '')


def additional_context(text: str) -> dict:
    """A PreToolUse reply putting `text` in front of the model and deciding nothing else.

    A hook's plain stdout reaches the debug log, so anything a model must read goes here.
    """
    return {HOOK_SPECIFIC_OUTPUT: {HOOK_EVENT_NAME: PRE_TOOL_USE, ADDITIONAL_CONTEXT: text}}
