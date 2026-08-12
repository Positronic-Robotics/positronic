"""Every documented command runs from a bare `uv`, against a platform that answers everything.

The commands and the example scripts are what a new user types first, so what is checked here is
the whole of that first experience: a checkout, no installation step, no packages on the path, and
the exact invocation the READMEs cite. A subprocess with `VIRTUAL_ENV` and `PYTHONPATH` scrubbed is
what makes that claim honest — `uv run` has to build the environment itself or the command fails.

The platform is a fake that answers 200 to every route, so a failure here is the invocation or the
environment, never the platform.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from platform_client import routes
from platform_client.client import API_KEY_ENV, API_URL_ENV, CREDENTIAL_ENV
from platform_client.responses import QUOTA_SUBMISSIONS_DAY

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION = '5f3a91c2b7d40e18'
AT = '2026-03-04T05:06:07Z'

# One canned answer per route, each the shape its response model requires. A submission is finished
# the first time it is read, so a polling example terminates on its first pass.
ANSWERS: dict[str, object] = {
    routes.USERS_REGISTER: {
        'user_id': 'a0',
        'artifact_location': 's3://bucket/users/a0/',
        'api_key': 'pk_live_fake',
        'key_status': 'created',
    },
    routes.USERS_ME: {
        'user_id': 'a0',
        'alias': 'demo',
        'tenant': 'demo-tenant',
        'plan': 'demo-plan',
        'quota': [
            {
                'key': QUOTA_SUBMISSIONS_DAY,
                'meter': 'submissions',
                'unit': 'submission',
                'scale': 1,
                'window': 'day',
                'subject': 'user',
                'scope': [],
                'limit': 2,
                'used': 1,
                'resets_at': AT,
                'on_exhausted': 'block',
            }
        ],
    },
    routes.SUBMISSIONS_CREATE: {
        'submission_id': SUBMISSION,
        'status': 'pending',
        'policy_image_digest': 'sha256:abc',
        'eval_version': 'fake.smoke@0123456789ab',
    },
    routes.SUBMISSIONS_GET: {
        'id': SUBMISSION,
        'status': 'finished',
        'scores': {'primary': 0.5, 'episodes': 4},
        'artifacts': {'result': 's3://bucket/result.json'},
    },
    routes.SUBMISSIONS_LIST: {
        'submissions': [
            {'id': SUBMISSION, 'user_id': 'a0', 'status': 'finished', 'eval': 'fake.smoke', 'received_at': AT}
        ]
    },
    routes.SUBMISSIONS_CANCEL: {'status': 'cancelled', 'refunded': True},
    routes.RANKINGS_GET: {
        'board': 'fake.smoke',
        'eval': 'fake.smoke',
        'eval_version': 'fake.smoke@0123456789ab',
        'primary_metric': 'success_rate',
        'rankings': [
            {
                'rank': 1,
                'display_name': 'demo',
                'tag': '0ddba7',
                'scores': {'primary': 0.5, 'episodes': 4},
                'submission_id': SUBMISSION,
                'submitted_at': AT,
            }
        ],
    },
    routes.RANKINGS_LIST: {
        'boards': [
            {
                'board': 'fake.smoke',
                'title': 'Fake smoke',
                'eval': 'fake.smoke',
                'eval_version': 'fake.smoke@0123456789ab',
                'primary_metric': 'success_rate',
                'visibility': 'public',
            }
        ]
    },
}


class _Handler(BaseHTTPRequestHandler):
    def _answer(self) -> None:
        path = self.path.split('?', 1)[0]
        payload = ANSWERS.get(path)
        if payload is None:
            self.send_error(404, f'no canned answer for {path}')
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, format: str, *args: object) -> None:  # the test reports failures itself
        pass


@pytest.fixture(scope='module')
def uv() -> str:
    """The `uv` every documented command runs through — a dependency, never a skip.

    Skipping on a missing one would report a first-run experience as checked on a box where nothing
    ran, and say so in the same words whether the commands passed or never executed. This repo is a
    uv project, so an absent `uv` is a broken environment and fails like one.
    """
    found = shutil.which('uv')
    assert found is not None, 'uv is not on PATH, and these are its commands — install it to run this suite'
    return found


@pytest.fixture(scope='module')
def platform_url() -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()


def run_from_a_clean_environment(command: list[str], *, platform_url: str) -> subprocess.CompletedProcess:
    """Run one documented command as a user with nothing installed would."""
    # An inherited interpreter or import path would let the command succeed on packages `uv run`
    # never had to provide, which is exactly the claim under test.
    env = {
        'PATH': os.environ['PATH'],
        'HOME': os.environ['HOME'],
        API_URL_ENV: platform_url,
        API_KEY_ENV: 'pk_live_fake',
        CREDENTIAL_ENV: 'token',
    }
    return subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800)


DOCUMENTED_COMMANDS = {
    'account-register': ['uv', 'run', 'positronic', 'account', 'register', '--alias=demo'],
    'eval-run-on-platform': [
        'uv',
        'run',
        'positronic',
        'eval',
        'run',
        '--eval=fake.smoke',
        '--policy-image=org/policy:v1',
    ],
    'eval-status': ['uv', 'run', 'positronic', 'eval', 'status', f'--submission-id={SUBMISSION}'],
    'eval-list': ['uv', 'run', 'positronic', 'eval', 'list'],
    'eval-cancel': ['uv', 'run', 'positronic', 'eval', 'cancel', f'--submission-id={SUBMISSION}'],
    'walkthrough': ['uv', 'run', 'positronic/cli/examples/walkthrough.py'],
    'submit-sample': [
        'uv',
        'run',
        'positronic/cli/examples/nebius_competition/submit_sample.py',
        '--policy-image=org/policy@sha256:abc',
    ],
}


@pytest.mark.parametrize('command', DOCUMENTED_COMMANDS.values(), ids=DOCUMENTED_COMMANDS.keys())
def test_a_documented_command_runs_with_nothing_installed(command: list[str], platform_url: str, uv: str):
    # Through the resolved `uv`, so what ran is the one the fixture found rather than whatever a
    # subprocess PATH would have picked.
    result = run_from_a_clean_environment([uv, *command[1:]], platform_url=platform_url)
    assert result.returncode == 0, f'{" ".join(command)} failed:\n{result.stdout}\n{result.stderr}'


README_FILES = (REPO_ROOT / 'client' / 'README.md', REPO_ROOT / 'positronic' / 'cli' / 'examples' / 'README.md')


def cited_commands() -> set[tuple[str, ...]]:
    """Every `uv run …` line in the READMEs, as its command path and its option NAMES."""
    cited = set()
    for path in README_FILES:
        for line in path.read_text().splitlines():
            stripped = line.strip().removeprefix('$ ')
            if stripped.startswith('uv run '):
                # A placeholder may hold a space (`--alias=<display name>`), so it goes before the
                # split rather than being filtered out of the words afterwards.
                cited.add(_shape(re.sub(r'<[^>]*>', '', stripped).split()))
    return cited


def _shape(command: list[str]) -> tuple[str, ...]:
    """A command's path and the NAMES of its options, with every value dropped.

    The values differ by construction — a README carries `--submission-id=<hex id>` where this test
    runs a real one — but the names must not, or a README could rename an option to something the
    CLI does not accept and this check would go on passing.
    """
    return tuple(token.split('=', 1)[0] for token in command)


def test_every_uv_command_the_readmes_cite_is_one_this_test_runs():
    # What is documented and what is checked drift apart silently otherwise: a command renamed in a
    # README goes on passing here under its old name, and the first person to hit it is a new user.
    run_here = {_shape(command) for command in DOCUMENTED_COMMANDS.values()}
    assert cited_commands() <= run_here, f'cited but never run: {sorted(cited_commands() - run_here)}'
