import json
import subprocess
from unittest.mock import patch

import pytest

from positronic.utils import nebius

# These spell out Nebius' response shape rather than reading it from the parser. A fixture built from
# the parser's own field names could only assert that the code agrees with itself; written independently,
# it fails when the code stops reading what Nebius sends.
# rules-allow: hardcoded-keys — an independent restatement of an external wire format is what a fixture is
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


def test_a_missing_resource_raises_not_found():
    stderr = 'Error: rpc error: code = NotFound desc = not found'
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'nebius', stderr=stderr)):
        with pytest.raises(nebius.NebiusNotFound):
            nebius.call('ai', 'endpoint', 'get-by-name', '--name', 'x')


def test_a_failure_other_than_not_found_is_not_read_as_absence():
    stderr = 'Error: rpc error: code = Unavailable desc = connection reset'
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'nebius', stderr=stderr)):
        with pytest.raises(nebius.NebiusError) as error:
            nebius.call('ai', 'endpoint', 'get-by-name', '--name', 'x')
    assert not isinstance(error.value, nebius.NebiusNotFound)


def test_output_that_is_not_json_raises_a_nebius_error():
    printed = subprocess.CompletedProcess(['nebius'], 0, stdout='A new version is available\n{}', stderr='')
    with patch('subprocess.run', return_value=printed):
        with pytest.raises(nebius.NebiusError, match='printed no JSON'):
            nebius.call('ai', 'job', 'get', '--id', 'x')
