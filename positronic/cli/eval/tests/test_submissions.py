"""`positronic eval status|list|cancel`: reading back what a platform run did, over a stub transport.

The gateway plumbing every command shares — the key, the URL, a refusal — is checked here, on the
commands that require a key.
"""

import pytest
from platform_client import routes

from positronic.cli.account import gateway as gateway_module
from positronic.cli.conftest import AT, ID
from positronic.cli.eval.submissions import cancel, list_submissions, status


def test_status_prints_the_fields_of_the_variant_it_got(platform, run_command, capsys):
    platform.answer({'id': ID, 'status': 'pending', 'received_at': AT, 'queued_at': AT, 'queue_position': 3})

    run_command(status, submission_id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_GET
    assert platform.request.url.params['id'] == ID
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'{ID} pending'
    assert '  queue_position: 3' in out


def test_list_prints_a_line_per_submission(platform, run_command, capsys):
    platform.answer({
        'submissions': [
            {'id': ID, 'user_id': 'a0', 'alias': 'demo', 'status': 'running', 'eval': 'fake.smoke', 'received_at': AT}
        ]
    })

    run_command(list_submissions)

    assert platform.request.url.path == routes.SUBMISSIONS_LIST
    assert capsys.readouterr().out.strip() == f'{ID} 2026-03-04 05:06 running fake.smoke demo'


def test_cancel_reports_whether_the_quota_came_back(platform, run_command, capsys):
    platform.answer({'status': 'cancelled', 'refunded': True})

    run_command(cancel, submission_id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_CANCEL
    assert platform.body == {'id': ID}
    assert 'cancelled, quota refunded' in capsys.readouterr().out


def test_a_submission_id_the_parser_read_as_a_number_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='hexadecimal'):
        run_command(status, submission_id=1234567890123456)


def test_a_submission_id_that_is_not_hexadecimal_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='not a submission id'):
        run_command(cancel, submission_id='zz')


def test_a_refusal_by_the_platform_exits_with_its_message(platform, run_command):
    platform.answer({'error': {'code': 'quota_exceeded', 'message': 'no submissions left today'}}, status=429)

    with pytest.raises(SystemExit) as exit_info:
        run_command(list_submissions)

    assert str(exit_info.value) == 'quota_exceeded: no submissions left today'


def test_a_command_needing_a_key_names_the_variable_that_holds_it(platform, run_command, monkeypatch):
    monkeypatch.delenv(gateway_module.API_KEY_ENV)

    with pytest.raises(SystemExit, match=gateway_module.API_KEY_ENV):
        run_command(list_submissions)


def test_an_unconfigured_url_leaves_the_client_on_its_default_platform(platform, run_command, monkeypatch):
    # A user should never have to know a URL: with nothing set, the client reaches the platform.
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer({'submissions': []})

    run_command(list_submissions)

    assert platform.base_url is None


def test_the_platform_url_argument_overrides_the_environment(platform, run_command):
    platform.answer({'submissions': []})

    run_command(list_submissions, platform_url='http://other.test')

    assert platform.base_url == 'http://other.test'
