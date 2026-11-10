"""`positronic account terms`, `positronic account accept-terms` and the terms step of `register`."""

import pytest
from platform_client import routes
from platform_client.enums import ErrorCode, KeyStatus
from platform_client.ids import ApiKey
from platform_client.responses import RegisterResponse

from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.account.register import register
from positronic.cli.account.terms import accept_terms, terms
from positronic.cli.conftest import AT, KEY, USER, refusal

TERMS = {'program': 'platform', 'terms': {'version': '1.3', 'url': 'https://example.com/terms'}}
ACCEPT_COMMAND = 'positronic account accept-terms --version=\'"1.3"\''
CREATED = RegisterResponse(
    user_id=USER, artifact_location='s3://b/users/a0/', api_key=ApiKey('pk_new'), key_status=KeyStatus.created
)
EXISTING = RegisterResponse(user_id=USER, artifact_location='s3://b/users/a0/', key_status=KeyStatus.existing)


def entry(version: str | None) -> dict:
    return {'program': 'platform', 'entered_at': AT, 'terms_version': version}


def test_terms_prints_the_page_and_the_command_that_accepts_them(platform, run_command, capsys):
    platform.answer_by_route({
        routes.TERMS_GET: (TERMS, 200),
        routes.PROGRAMS_ENTRY: ({'program': 'platform', 'entry': None}, 200),
    })

    run_command(terms)

    assert platform.request_to(routes.PROGRAMS_ENTRY).url.params['program'] == 'platform'
    assert platform.request_to(routes.PROGRAMS_ENTRY).headers['authorization'] == f'Bearer {KEY}'
    assert 'authorization' not in platform.request_to(routes.TERMS_GET).headers
    out = capsys.readouterr().out
    assert 'version 1.3: https://example.com/terms' in out
    assert ACCEPT_COMMAND in out
    assert routes.PLATFORM_ACCEPT_TERMS not in platform.paths


def test_terms_says_so_when_the_account_accepted_the_current_version(platform, run_command, capsys):
    platform.answer_by_route({
        routes.TERMS_GET: (TERMS, 200),
        routes.PROGRAMS_ENTRY: ({'program': 'platform', 'entry': entry('1.3')}, 200),
    })

    run_command(terms)

    out = capsys.readouterr().out
    assert 'You accepted version 1.3.' in out
    assert ACCEPT_COMMAND not in out


def test_terms_names_an_earlier_acceptance_as_not_current(platform, run_command, capsys):
    platform.answer_by_route({
        routes.TERMS_GET: (TERMS, 200),
        routes.PROGRAMS_ENTRY: ({'program': 'platform', 'entry': entry('1.2')}, 200),
    })

    run_command(terms)

    out = capsys.readouterr().out
    assert ACCEPT_COMMAND in out
    assert 'You accepted version 1.2, which is not current.' in out


def test_accept_terms_sends_the_version_that_the_person_read_with_the_key(platform, run_command, capsys):
    platform.answer_by_route({routes.PLATFORM_ACCEPT_TERMS: (entry('1.3'), 200)})

    run_command(accept_terms, version='1.3')

    assert platform.body_to(routes.PLATFORM_ACCEPT_TERMS) == {'terms_version': '1.3'}
    assert platform.request_to(routes.PLATFORM_ACCEPT_TERMS).headers['authorization'] == f'Bearer {KEY}'
    assert 'You accepted the platform terms, version 1.3.' in capsys.readouterr().out


@pytest.mark.parametrize('version', [1.3, 13, None, ''])
def test_accept_terms_refuses_a_version_that_is_not_text_before_any_call(platform, run_command, version):
    # The CLI reads `--version=1.3` as a number, so `1.10` would reach the platform as `1.1`.
    with pytest.raises(CommandFailed) as raised:
        run_command(accept_terms, version=version)

    assert raised.value.fault is Fault.USAGE
    assert platform.seen is None


def test_accept_terms_reports_a_version_that_is_not_current_as_the_callers_fault(platform, run_command):
    message = 'the platform terms are now version 1.4: read them and accept them again'
    platform.answer_by_route({routes.PLATFORM_ACCEPT_TERMS: (refusal(ErrorCode.bad_request, message), 400)})

    with pytest.raises(CommandFailed) as raised:
        run_command(accept_terms, version='1.3')

    assert raised.value.fault is Fault.CALLER
    assert message in str(raised.value)


def test_register_prints_the_terms_and_accepts_nothing_without_the_flag(platform, run_command, capsys):
    platform.answer_by_route({routes.USERS_REGISTER: (CREATED, 200), routes.TERMS_GET: (TERMS, 200)})

    run_command(register)

    assert routes.PLATFORM_ACCEPT_TERMS not in platform.paths
    assert ACCEPT_COMMAND in capsys.readouterr().out


def test_register_accepts_the_named_version_with_the_key_it_minted(platform, run_command, capsys):
    platform.answer_by_route({routes.USERS_REGISTER: (CREATED, 200), routes.PLATFORM_ACCEPT_TERMS: (entry('1.3'), 200)})

    run_command(register, accept_terms='1.3')

    assert platform.body_to(routes.PLATFORM_ACCEPT_TERMS) == {'terms_version': '1.3'}
    assert platform.request_to(routes.PLATFORM_ACCEPT_TERMS).headers['authorization'] == 'Bearer pk_new'
    assert 'You accepted the platform terms, version 1.3.' in capsys.readouterr().out


def test_register_that_mints_no_key_accepts_nothing_and_says_so(platform, run_command, capsys):
    platform.answer_by_route({routes.USERS_REGISTER: (EXISTING, 200), routes.TERMS_GET: (TERMS, 200)})

    run_command(register, accept_terms='1.3')

    assert routes.PLATFORM_ACCEPT_TERMS not in platform.paths
    out = capsys.readouterr().out
    assert 'did not accept the terms' in out
    assert ACCEPT_COMMAND in out


def test_register_refuses_a_terms_version_that_is_not_text_before_the_sign_in(platform, run_command):
    with pytest.raises(CommandFailed) as raised:
        run_command(register, accept_terms=1.3)

    assert raised.value.fault is Fault.USAGE
    assert platform.seen is None
