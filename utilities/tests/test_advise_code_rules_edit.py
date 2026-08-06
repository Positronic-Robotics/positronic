import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import advise_code_rules_edit as advise  # noqa: E402

HOOK = Path(__file__).resolve().parents[1] / 'advise_code_rules_edit.py'


def payload(path):
    return {'tool_name': 'Edit', 'tool_input': {'file_path': path}}


def test_it_advises_on_the_rules_file_wherever_the_checkout_sits():
    for path in ('/w/positronic/CODE_RULES.md', 'CODE_RULES.md', '/somewhere/else/CODE_RULES.md'):
        assert advise.advises(payload(path)), path


def test_it_says_nothing_about_any_other_file():
    for path in ('/w/positronic/CLAUDE.md', '/w/positronic/utilities/x.py', '/w/CODE_RULES.md.bak', ''):
        assert not advise.advises(payload(path)), path


def test_a_payload_it_cannot_read_is_not_a_rules_edit():
    assert not advise.advises({})
    assert not advise.advises({'tool_input': {}})


def test_it_allows_the_edit_and_names_the_skill():
    """Advisory: the skill's own last step is an edit to this file, so a refusal would block it."""
    run = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload('/w/positronic/CODE_RULES.md')),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0
    assert 'add-rule' in run.stdout


def test_an_unreadable_payload_is_silent():
    run = subprocess.run([sys.executable, str(HOOK)], input='not json', capture_output=True, text=True)
    assert run.returncode == 0
    assert run.stdout == ''
