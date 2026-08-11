import json
import subprocess
from unittest.mock import patch

import pytest

from positronic.utils import nebius

_SECRET = {'metadata': {'id': 'mbsec-abc'}}
_PAYLOAD = {'data': {'string_value': 'a-token'}}


def _cli(*responses):
    """Stands in for ``subprocess.run``, answering each `nebius` call with the next response."""
    replies = iter(responses)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(next(replies)), stderr='')

    return run


def test_auth_token_reads_the_payload_of_the_secret_named():
    with patch('subprocess.run', side_effect=_cli(_SECRET, _PAYLOAD)) as run:
        assert nebius.auth_token(secret='a-secret', parent_id='project-1') == 'a-token'
    lookup, fetch = (call.args[0] for call in run.call_args_list)
    assert lookup[:4] == ['nebius', 'mysterybox', 'secret', 'get-by-name']
    assert lookup[lookup.index('--name') + 1] == 'a-secret'
    assert lookup[lookup.index('--parent-id') + 1] == 'project-1'
    # The payload is addressed by the id the lookup returned, the only form that call accepts.
    assert fetch[fetch.index('--secret-id') + 1] == 'mbsec-abc'


def test_an_empty_payload_is_not_passed_off_as_a_token():
    with patch('subprocess.run', side_effect=_cli(_SECRET, {'data': {'string_value': ''}})):
        with pytest.raises(RuntimeError, match='no AUTH_TOKEN'):
            nebius.auth_token()


def test_a_missing_cli_names_the_alternative():
    with patch('subprocess.run', side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match='authed_remote'):
            nebius.auth_token()


def test_a_login_prompt_fails_instead_of_waiting_for_a_human():
    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('nebius', 30.0)):
        with pytest.raises(RuntimeError, match='login'):
            nebius.auth_token()


def test_a_failing_call_surfaces_what_the_cli_said():
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'nebius', stderr='denied')):
        with pytest.raises(RuntimeError, match='denied'):
            nebius.auth_token()
