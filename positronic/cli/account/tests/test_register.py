"""`positronic account register`, over a stub platform transport."""

import shlex

import pytest
from platform_client import routes

from positronic.cli.account import gateway as gateway_module
from positronic.cli.account.register import register


def test_register_prints_the_export_line_for_a_minted_key(platform, run_command, capsys):
    platform.answer({
        'user_id': 'a0',
        'artifact_location': 's3://b/users/a0/',
        'api_key': 'pk_new',
        'key_status': 'created',
    })

    run_command(register, alias='demo')

    assert platform.request.url.path == routes.USERS_REGISTER
    assert 'authorization' not in platform.request.headers
    assert platform.body == {'credential': 'token', 'alias': 'demo', 'rotate': False}
    out = capsys.readouterr().out
    assert 'user a0 (created)' in out
    assert f'export {gateway_module.API_KEY_ENV}=pk_new' in out


def test_an_export_line_survives_a_key_holding_shell_characters(platform, run_command, capsys):
    # The key is opaque; pasting an unquoted export line would run the `;` and drop the rest.
    platform.answer({
        'user_id': 'a0',
        'artifact_location': 's3://b/users/a0/',
        'api_key': 'pk a$b;rm -rf /',
        'key_status': 'created',
    })

    run_command(register)

    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith('export '))
    assert shlex.split(line)[1] == f'{gateway_module.API_KEY_ENV}=pk a$b;rm -rf /'


def test_register_refuses_when_no_credential_is_in_the_environment(platform, run_command, monkeypatch):
    monkeypatch.delenv(gateway_module.CREDENTIAL_ENV)
    with pytest.raises(SystemExit) as raised:
        run_command(register)
    assert gateway_module.CREDENTIAL_ENV in str(raised.value)


def test_an_alias_the_request_model_refuses_is_a_refusal_naming_the_field(platform, run_command):
    # configuronic literal-evaluates a CLI value, so `--alias=5` arrives as an int, which the model
    # refuses — a refusal to read rather than a traceback out of the constructor.
    with pytest.raises(SystemExit) as raised:
        run_command(register, alias=5)
    assert 'alias' in str(raised.value)
    assert platform.seen is None


def test_register_says_no_key_came_back_for_an_existing_registration(platform, run_command, capsys):
    platform.answer({'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})

    run_command(register)

    assert 'no key issued' in capsys.readouterr().out
