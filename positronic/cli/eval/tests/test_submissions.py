"""`positronic eval status|list|cancel|catalog`: reading back what the platform is doing, over a stub transport.

The gateway plumbing every command shares — the key, the URL, a refusal — is checked here, on the
commands that require a key.
"""

import pytest
from platform_client import routes

from positronic.cli.account import gateway as gateway_module
from positronic.cli.conftest import AT, ID
from positronic.cli.eval.submissions import cancel, catalog, list_runs, status


def test_status_prints_the_fields_of_the_variant_it_got(platform, run_command, capsys):
    platform.answer({'id': ID, 'status': 'pending', 'received_at': AT, 'queued_at': AT, 'queue_position': 3})

    run_command(status, id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_GET
    assert platform.request.url.params['id'] == ID
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'submission {ID} pending'
    assert '  queue_position: 3' in out
    # The header carries the id and the status, so the body below it repeats neither.
    assert [line for line in out[1:] if line.startswith(('  id:', '  status:'))] == []


PLAN_PAYLOAD = {'plan_id': ID, 'status': 'running', 'episodes': {'total': 20, 'done': 3, 'outstanding': 17}}
NO_SUBMISSION = ({'error': {'code': 'not_found', 'message': 'no such submission'}}, 404)


def test_an_id_the_platform_knows_no_submission_for_is_read_as_a_plan(platform, run_command, capsys):
    # A submission id and a plan id are both bare hex, so the id says nothing about which it names.
    platform.answer_by_route({routes.SUBMISSIONS_GET: NO_SUBMISSION, routes.EVALS_GET: (PLAN_PAYLOAD, 200)})

    run_command(status, id=ID)

    assert platform.paths == [routes.SUBMISSIONS_GET, routes.EVALS_GET]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'plan {ID} running'
    assert "  episodes: {'total': 20, 'done': 3, 'outstanding': 17}" in out


def test_list_prints_a_labelled_line_per_submission_and_per_plan(platform, run_command, capsys):
    platform.answer({
        'submissions': [
            {'id': ID, 'user_id': 'a0', 'alias': 'demo', 'status': 'running', 'eval': 'fake.smoke', 'received_at': AT}
        ],
        'plans': [{'plan_id': ID, 'status': 'filed', 'episodes': {'total': 20, 'done': 0, 'outstanding': 20}}],
    })

    run_command(list_runs)

    assert platform.request.url.path == routes.EVALS_LIST
    out = capsys.readouterr().out.splitlines()
    assert out == [f'submission {ID} 2026-03-04 05:06 running fake.smoke demo', f'plan {ID} filed 0/20 episodes']


def test_catalog_prints_what_the_key_may_name(platform, run_command, capsys):
    platform.answer({
        'evals': [{'id': 'fake.smoke', 'embodiment': 'franka', 'tasks': ['a'], 'composable': True}],
        'tasks': [{'id': 'eight-spoons-into-grey-tote', 'embodiment': 'franka', 'task': 'spoons'}],
    })

    run_command(catalog)

    assert platform.request.url.path == routes.CATALOG_TASKS
    out = capsys.readouterr().out
    assert '"id": "fake.smoke"' in out
    assert '"id": "eight-spoons-into-grey-tote"' in out


def test_cancel_reports_whether_the_quota_came_back(platform, run_command, capsys):
    platform.answer({'status': 'cancelled', 'refunded': True})

    run_command(cancel, id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_CANCEL
    assert platform.body == {'id': ID}
    assert 'cancelled, quota refunded' in capsys.readouterr().out


def test_cancel_on_a_plan_says_the_platform_cancels_none_yet(platform, run_command):
    # `evals.*` carries no cancel route, so the command says so rather than reporting a not_found
    # about a plan the platform holds.
    platform.answer_by_route({routes.SUBMISSIONS_CANCEL: NO_SUBMISSION, routes.EVALS_GET: (PLAN_PAYLOAD, 200)})

    with pytest.raises(SystemExit, match='cancels no plan yet'):
        run_command(cancel, id=ID)


def test_a_refusal_that_is_not_a_missing_record_is_not_read_as_the_other_kind(platform, run_command):
    platform.answer({'error': {'code': 'forbidden', 'message': 'no grant'}}, status=403)

    with pytest.raises(SystemExit, match='forbidden'):
        run_command(status, id=ID)


def test_an_id_the_parser_read_as_a_number_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='hexadecimal'):
        run_command(status, id=1234567890123456)


def test_an_id_that_is_not_hexadecimal_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='not an id'):
        run_command(cancel, id='zz')


def test_a_refusal_by_the_platform_exits_with_its_message(platform, run_command):
    platform.answer({'error': {'code': 'quota_exceeded', 'message': 'no submissions left today'}}, status=429)

    with pytest.raises(SystemExit) as exit_info:
        run_command(list_runs)

    assert str(exit_info.value) == 'quota_exceeded: no submissions left today'


def test_a_command_needing_a_key_names_the_variable_that_holds_it(platform, run_command, monkeypatch):
    monkeypatch.delenv(gateway_module.API_KEY_ENV)

    with pytest.raises(SystemExit, match=gateway_module.API_KEY_ENV):
        run_command(list_runs)


def test_an_unconfigured_url_leaves_the_client_on_its_default_platform(platform, run_command, monkeypatch):
    # A user should never have to know a URL: with nothing set, the client reaches the platform.
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer({'submissions': [], 'plans': []})

    run_command(list_runs)

    assert platform.base_url is None


def test_the_platform_url_argument_overrides_the_environment(platform, run_command):
    platform.answer({'submissions': [], 'plans': []})

    run_command(list_runs, platform_url='http://other.test')

    assert platform.base_url == 'http://other.test'
