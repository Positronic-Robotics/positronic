import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import advise_code_rules_edit as advise  # noqa: E402
import hook_payload  # noqa: E402

HOOK = Path(__file__).resolve().parents[1] / 'advise_code_rules_edit.py'


def payload(path):
    return {'tool_name': 'Edit', 'tool_input': {'file_path': path}}


def test_it_advises_on_the_rules_file_wherever_the_checkout_sits():
    for path in ('/w/positronic/CODE_RULES.md', 'CODE_RULES.md', '/somewhere/else/CODE_RULES.md'):
        assert advise.advises(payload(path)), path


def test_it_says_nothing_about_any_other_file():
    for path in ('/w/positronic/CLAUDE.md', '/w/positronic/utilities/x.py', '/w/CODE_RULES.md.bak', ''):
        assert not advise.advises(payload(path)), path


def test_it_advises_on_a_shell_command_that_could_write_the_file():
    """A heredoc, a redirect or an in-place edit reaches the file without a file_path."""
    for command in ('cat > CODE_RULES.md <<EOF', "sed -i s/x/y/ CODE_RULES.md", 'tee ../CODE_RULES.md'):
        assert advise.advises({'tool_input': {'command': command}}), command


def test_it_says_nothing_about_a_shell_command_that_does_not_name_it():
    for command in ('git status', 'cat CLAUDE.md', 'pytest utilities/tests/'):
        assert not advise.advises({'tool_input': {'command': command}}), command


def test_a_payload_it_cannot_read_is_not_a_rules_edit():
    assert not advise.advises({})
    assert not advise.advises({'tool_input': {}})


def test_it_allows_the_edit_and_reaches_the_model_with_the_skill_s_name():
    """Advisory, so exit 0 — and as `additionalContext`, since a PreToolUse hook's plain stdout
    reaches the debug log and nothing else."""
    run = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload('/w/positronic/CODE_RULES.md')),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0
    emitted = json.loads(run.stdout)[hook_payload.HOOK_SPECIFIC_OUTPUT]
    assert emitted[hook_payload.HOOK_EVENT_NAME] == hook_payload.PRE_TOOL_USE
    assert 'add-rule' in emitted[hook_payload.ADDITIONAL_CONTEXT]


def test_an_unreadable_payload_is_silent():
    run = subprocess.run([sys.executable, str(HOOK)], input='not json', capture_output=True, text=True)
    assert run.returncode == 0
    assert run.stdout == ''
