import json
import re
import subprocess
import sys
from pathlib import Path

# The hooks are scripts the harness runs by path, not an importable package, so the directory holding
# them has to reach `sys.path` before the import rather than at the top of the file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hook_payload  # noqa: E402
import state_code_rules as state  # noqa: E402

HOOK = Path(__file__).resolve().parents[1] / 'state_code_rules.py'

SAMPLE = """# Code Rules

Prose about the file, which states no rule.

## Rules

### first-rule

Don't do the first thing. Do the safe thing instead.

The exception, which the digest leaves behind.

### second-rule

Don't do the second thing.

```python
# Bad
x = 1
```
"""


def test_it_states_every_rule_the_file_declares():
    rules = state.RULES_FILE.read_text()
    declared = re.findall(r'^### (.+)$', rules, re.MULTILINE)
    assert declared, 'CODE_RULES.md declares no rules; the digest would be empty'
    for rule_id in declared:
        assert f'{rule_id}:' in state.digest(rules), rule_id


def test_a_rule_arrives_as_its_opening_paragraph_and_nothing_after_it():
    assert state.digest(SAMPLE) == (
        "first-rule:\nDon't do the first thing. Do the safe thing instead.\n\nsecond-rule:\nDon't do the second thing."
    )


def test_it_reaches_the_model_where_the_harness_reads_a_session_start_reply():
    # rules-allow: hardcoded-keys — the waiver syntax and one rule id are spelled here rather than read
    # back from the module under test. Building the expectation from `state.PREAMBLE` and `state.digest`
    # would assert those functions against themselves and pass whatever they emitted.
    run = subprocess.run([sys.executable, str(HOOK)], capture_output=True, text=True)
    assert run.returncode == 0
    emitted = json.loads(run.stdout)[hook_payload.HOOK_SPECIFIC_OUTPUT]
    assert emitted[hook_payload.HOOK_EVENT_NAME] == hook_payload.SESSION_START
    assert 'rules-allow' in emitted[hook_payload.ADDITIONAL_CONTEXT]
    assert 'optional-lie:' in emitted[hook_payload.ADDITIONAL_CONTEXT]


def test_a_checkout_carrying_no_rules_file_is_silent(monkeypatch, tmp_path, capsys):
    """A repository with no `CODE_RULES.md` of its own is checked against the one on `main`."""
    monkeypatch.setattr(state, 'RULES_FILE', tmp_path / 'CODE_RULES.md')
    assert state.main() == 0
    assert capsys.readouterr().out == ''
