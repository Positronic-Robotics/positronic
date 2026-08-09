import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hook_payload  # noqa: E402

# rules-allow: hardcoded-keys — these tests spell the harness's field names rather than importing
# the constants under test. Asserting a constant against itself passes however far it has drifted
# from what the harness actually sends; the literal is the independent statement of the wire.


def test_it_reads_the_command_a_bash_call_runs():
    assert hook_payload.command({'tool_input': {'command': 'git push'}}) == 'git push'


def test_it_reads_the_file_a_write_targets_under_either_name():
    assert hook_payload.target_path({'tool_input': {'file_path': '/w/x.py'}}) == '/w/x.py'
    assert hook_payload.target_path({'tool_input': {'notebook_path': '/w/x.ipynb'}}) == '/w/x.ipynb'


def test_a_payload_carrying_neither_reads_empty():
    for payload in ({}, {'tool_input': {}}, {'tool_input': None}):
        assert hook_payload.command(payload) == ''
        assert hook_payload.target_path(payload) == ''


def test_the_reply_carries_the_text_where_the_harness_reads_it_for_the_event_it_answers():
    assert hook_payload.additional_context('read the rules first', hook_payload.PRE_TOOL_USE) == {
        'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'additionalContext': 'read the rules first'}
    }
    assert hook_payload.additional_context('the rules', hook_payload.SESSION_START) == {
        'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': 'the rules'}
    }
