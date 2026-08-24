#!/usr/bin/env python3
"""The field names of the JSON a Claude Code hook is handed on stdin, and readers for them.

The names belong to the harness. Stdlib-only, so a hook that imports this still runs without the
project venv; the harness runs a hook by path, which puts this directory on `sys.path`.
"""

from __future__ import annotations

TOOL_NAME = 'tool_name'
TOOL_INPUT = 'tool_input'
COMMAND = 'command'
FILE_PATH = 'file_path'
NOTEBOOK_PATH = 'notebook_path'
CWD = 'cwd'
HOOK_SPECIFIC_OUTPUT = 'hookSpecificOutput'
HOOK_EVENT_NAME = 'hookEventName'
ADDITIONAL_CONTEXT = 'additionalContext'
PRE_TOOL_USE = 'PreToolUse'
SESSION_START = 'SessionStart'


def tool_name(payload: dict) -> str:
    """The tool a call names, or '' when the payload names none."""
    return str(payload.get(TOOL_NAME) or '')


def tool_input(payload: dict) -> dict:
    """The arguments a tool call carries, or {} when it carries none readable as arguments."""
    arguments = payload.get(TOOL_INPUT)
    return arguments if isinstance(arguments, dict) else {}


def command(payload: dict) -> str:
    """The shell command a Bash tool call runs, or '' when the payload carries none."""
    return str((payload.get(TOOL_INPUT) or {}).get(COMMAND) or '')


def target_path(payload: dict) -> str:
    """The file a write-shaped tool call targets, or '' when it names none."""
    arguments = payload.get(TOOL_INPUT) or {}
    return str(arguments.get(FILE_PATH) or arguments.get(NOTEBOOK_PATH) or '')


def additional_context(text: str, event: str) -> dict:
    """A reply to `event` putting `text` in front of the model and deciding nothing else."""
    return {HOOK_SPECIFIC_OUTPUT: {HOOK_EVENT_NAME: event, ADDITIONAL_CONTEXT: text}}
