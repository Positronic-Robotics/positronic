import re
import shlex
from pathlib import Path

import pytest

from positronic.utils.credentials import CredentialInRunMetadata, check_command, check_text

TOKEN = 'q7RfX2mKp9WzL4vNc8YbT3hJd'
LETTERS_ONLY_TOKEN = 'qRfXmKpWzLvNcYbThJdGsAeUo'
REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    'argv',
    [
        ['pos', f'--policy.headers={{"Modal-Key": "{TOKEN}", "Modal-Secret": "{LETTERS_ONLY_TOKEN}"}}'],
        ['pos', f'--policy.headers={ {"Modal-Key": TOKEN, "Modal-Secret": LETTERS_ONLY_TOKEN}!r}'],
        ['pos', repr({'Modal-Key': TOKEN})],
        ['pos', '--policy.headers={"X-Session": "short"}'],
        ['pos', '--policy.headers', '{}'],
        ['pos', '--policy.headers.Modal-Key=short'],
        ['pos', f'--x.api_key={TOKEN}'],
        ['pos', f'--x.api_key={LETTERS_ONLY_TOKEN}'],
        ['pos', '--token', TOKEN],
        ['pos', f'Authorization: Bearer {TOKEN}'],
        ['pos', f'--token=act_{TOKEN}'],
    ],
    ids=[
        'inline header set',
        'header set as a Python repr',
        'Python repr with no headers flag',
        'inline header set with no key word',
        'inline header set as the next argument',
        'one inline header',
        'key word and token',
        'key word and letters-only token',
        'flag and token as two arguments',
        'authorization value',
        'token with a gitleaks stopword',
    ],
)
def test_a_command_with_a_credential_is_refused(argv):
    with pytest.raises(CredentialInRunMetadata, match=re.escape('argv[1]')):
        check_command(argv)


@pytest.mark.parametrize(
    'arg',
    [
        '--policy.headers.path=~/.config/endpoint/headers.json',
        '--policy.headers=.bearer_headers',
        '--output_dir=s3://recordings/run/',
        '--policy.checkpoints_dir=s3://checkpoints/act/run_1/checkpoint-30000',
        '--policy-image=ghcr.io/positronic/policy@sha256:' + 'a3f1' * 16,
        '--model.revision=9f12de04ab9f12de04ab9f12de04ab9f12de04ab',
        '--wrist_key=image.wrist_2',
        '--state_key=robot_state.q_7dof',
    ],
)
def test_a_command_with_no_credential_passes(arg):
    check_command(['pos', arg])


def _documented_commands() -> list[str]:
    files = sorted([*REPO.glob('docs/**/*.md'), *REPO.glob('positronic/vendors/*/README.md')])
    lines = []
    for file in files:
        for block in re.findall(r'```[^\n]*\n(.*?)```', file.read_text(encoding='utf-8'), re.S):
            lines.extend(line.strip() for line in block.replace('\\\n', ' ').splitlines())
    return [line for line in lines if line and not line.startswith('#')]


def test_every_documented_command_passes():
    commands = _documented_commands()
    assert len(commands) > 100
    for command in commands:
        try:
            argv = shlex.split(command, comments=True)
        except ValueError:  # a code block line that is not a shell command, such as Python with an open quote
            continue
        check_command(argv)


def test_the_refusal_names_the_argument_and_not_the_value():
    with pytest.raises(CredentialInRunMetadata) as exc:
        check_command(['pos', '--output_dir=s3://recordings/run/', '--token', TOKEN])

    assert 'argv[2]' in str(exc.value)
    assert TOKEN not in str(exc.value)


def test_a_diff_line_with_a_credential_is_refused_by_field_and_line():
    diff = f'--- a/cfg.py\n+++ b/cfg.py\n@@ -1 +1 @@\n-API_TOKEN = None\n+API_TOKEN = "{TOKEN}"'

    with pytest.raises(CredentialInRunMetadata, match=r'^git\.current\.diff: line 5 ') as exc:
        check_text(diff, 'git.current.diff')

    assert TOKEN not in str(exc.value)


def test_a_diff_with_no_credential_passes():
    check_text(
        '+    image_key = "image.wrist_2"\n+    api_version = "v1"\n+    token = os.environ[TOKEN_ENV]\n',
        'git.current.diff',
    )
