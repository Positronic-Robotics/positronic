import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hook_payload  # noqa: E402


def test_it_reads_the_command_a_bash_call_runs():
    assert hook_payload.command({'tool_input': {'command': 'git push'}}) == 'git push'


def test_it_reads_the_file_a_write_targets_under_either_name():
    assert hook_payload.target_path({'tool_input': {'file_path': '/w/x.py'}}) == '/w/x.py'
    assert hook_payload.target_path({'tool_input': {'notebook_path': '/w/x.ipynb'}}) == '/w/x.ipynb'


def test_a_payload_carrying_neither_reads_empty():
    for payload in ({}, {'tool_input': {}}, {'tool_input': None}):
        assert hook_payload.command(payload) == ''
        assert hook_payload.target_path(payload) == ''


def test_the_reply_carries_the_text_where_the_harness_reads_it():
    reply = hook_payload.additional_context('read the rules first')
    assert reply == {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'additionalContext': 'read the rules first'}}
