import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import advise_configuronic as advise  # noqa: E402
import hook_payload  # noqa: E402


@pytest.mark.parametrize(
    'before, after, expected',
    [
        ('settings = defaults()', 'settings = cfn.Config(defaults)', True),
        ('', 'a = cfn.Config(A)\nb = cfn.Config(B)', True),
        ('a = cfn.Config(A)', 'a = cfn.Config(A)\nb = cfn.Config(B)', True),
        ('a = cfn.Config(A)', 'a = cfn.Config(A)\nvalue = 1', False),
        ('a = cfn.Config(A)', 'a = cfn.Config(A, value=1)', False),
        ('a = cfn.Config(A)', 'a = A()', False),
        ('', '@cfn.config()\ndef value():\n    return {}', False),
    ],
)
@pytest.mark.parametrize('tool', ['Edit', 'MultiEdit', 'Write'])
def test_only_an_added_constructor_triggers_advice(tmp_path, tool, before, after, expected):
    path = tmp_path / 'config.py'
    path.write_text(before)
    edit = {hook_payload.OLD_STRING: before, hook_payload.NEW_STRING: after}
    arguments = {hook_payload.FILE_PATH: str(path)}
    if tool == 'Edit':
        arguments.update(edit)
    elif tool == 'MultiEdit':
        arguments[hook_payload.EDITS] = [
            {hook_payload.OLD_STRING: 'unrelated', hook_payload.NEW_STRING: 'change'},
            edit,
        ]
    else:
        arguments[hook_payload.CONTENT] = after
    assert advise.introduces_config({hook_payload.TOOL_NAME: tool, hook_payload.TOOL_INPUT: arguments}) is expected


def test_a_write_to_a_new_python_file_triggers_advice(tmp_path):
    assert advise.introduces_config({
        hook_payload.TOOL_NAME: 'Write',
        hook_payload.CWD: str(tmp_path),
        hook_payload.TOOL_INPUT: {hook_payload.FILE_PATH: 'new.py', hook_payload.CONTENT: 'value = cfn.Config(Value)'},
    })


@pytest.mark.parametrize(
    'tool, path', [('Read', 'config.py'), ('Bash', 'config.py'), ('Write', 'README.md'), ('Edit', '')]
)
def test_reads_shell_commands_and_non_python_files_are_silent(tool, path):
    assert not advise.introduces_config({
        hook_payload.TOOL_NAME: tool,
        hook_payload.TOOL_INPUT: {
            hook_payload.FILE_PATH: path,
            hook_payload.CONTENT: 'value = cfn.Config(Value)',
            hook_payload.NEW_STRING: 'value = cfn.Config(Value)',
            hook_payload.COMMAND: "rg 'cfn.Config(' config.py",
        },
    })


@pytest.fixture
def run_hook(tmp_path):
    def run(session: str, replacement: str = 'value = cfn.Config(Value)'):
        payload = {
            hook_payload.SESSION_ID: session,
            hook_payload.TOOL_NAME: 'Edit',
            hook_payload.TOOL_INPUT: {
                hook_payload.FILE_PATH: str(tmp_path / 'config.py'),
                hook_payload.OLD_STRING: '',
                hook_payload.NEW_STRING: replacement,
            },
        }
        return subprocess.run(
            [sys.executable, str(Path(advise.__file__))],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env={**os.environ, 'TMPDIR': str(tmp_path)},
        )

    return run


def test_nonmatching_edits_do_not_spend_the_session_reminder(run_hook):
    assert run_hook('session', 'value = 1').stdout == ''
    first = run_hook('session')
    assert first.returncode == 0
    assert first.stderr == ''
    assert json.loads(first.stdout) == hook_payload.additional_context(advise.ADVICE, hook_payload.PRE_TOOL_USE)
    assert run_hook('session').stdout == ''
    assert run_hook('different-session').stdout == first.stdout


def test_concurrent_hook_processes_emit_one_reminder(run_hook):
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(run_hook, ['session'] * 4))
    assert all(result.returncode == 0 and result.stderr == '' for result in results)
    assert sum(bool(result.stdout) for result in results) == 1


@pytest.mark.parametrize('payload', ['invalid json', 'null', '{}'])
def test_bad_input_does_not_block_the_tool(payload):
    result = subprocess.run([sys.executable, str(Path(advise.__file__))], input=payload, text=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout == ''


def test_state_write_failure_is_logged_without_blocking(tmp_path, monkeypatch, capsys, caplog):
    state_file = tmp_path / 'not-a-directory'
    state_file.touch()
    monkeypatch.setattr(advise.tempfile, 'gettempdir', lambda: str(state_file))
    payload = {
        hook_payload.SESSION_ID: 'session',
        hook_payload.TOOL_NAME: 'Edit',
        hook_payload.TOOL_INPUT: {
            hook_payload.FILE_PATH: str(tmp_path / 'config.py'),
            hook_payload.NEW_STRING: 'value = cfn.Config(Value)',
        },
    }
    monkeypatch.setattr(advise.sys, 'stdin', io.StringIO(json.dumps(payload)))
    assert advise.main() == 0
    assert capsys.readouterr().out == ''
    assert 'Configuronic advisory hook failed' in caplog.text


def test_project_hook_registration_matches_only_structured_edits():
    root = Path(__file__).resolve().parents[3]
    settings = json.loads((root / '.claude/settings.json').read_text())
    entries = [
        entry
        for entry in settings['hooks'][hook_payload.PRE_TOOL_USE]
        if any('advise_configuronic.py' in handler[hook_payload.COMMAND] for handler in entry['hooks'])
    ]
    assert len(entries) == 1
    assert entries[0]['matcher'] == 'Edit|Write|MultiEdit'
    assert os.access(Path(advise.__file__), os.X_OK)
