"""`positronic eval status|list|cancel|catalog`: reading back what the platform is doing, over a stub transport.

The gateway plumbing every command shares — the key, the URL, a refusal — is checked here, on the
commands that require a key.
"""

import pytest
from platform_client import routes

from positronic.cli.account import gateway as gateway_module
from positronic.cli.conftest import AT, ID
from positronic.cli.eval.submissions import cancel, catalog, list_submissions, status


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


NO_SUBMISSION = ({'error': {'code': 'not_found', 'message': 'no such submission'}}, 404)
ROW = {'id': ID, 'user_id': 'a0', 'status': 'running', 'received_at': AT}


def test_status_prints_what_a_blocked_run_waits_on(platform, run_command, capsys):
    platform.answer({'id': ID, 'status': 'blocked', 'reason': 'the rig is not ready'})

    run_command(status, id=ID)

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'submission {ID} blocked'
    assert '  reason: the rig is not ready' in out


def test_list_prints_one_labelled_line_per_run(platform, run_command, capsys):
    # A run the platform executes itself reports no episode count, so only the rig's carries a tail.
    platform.answer({
        'submissions': [
            {**ROW, 'alias': 'demo', 'eval': 'fake.smoke'},
            {**ROW, 'id': '1a', 'episodes': {'total': 20, 'done': 0, 'outstanding': 20}},
        ]
    })

    run_command(list_submissions)

    assert platform.request.url.path == routes.SUBMISSIONS_LIST
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f'submission {ID} 2026-03-04 05:06 running fake.smoke demo',
        'submission 1a 2026-03-04 05:06 running 0/20 episodes',
    ]


def test_list_reads_every_page(platform, run_command, capsys):
    platform.answer_in_turn(
        routes.SUBMISSIONS_LIST,
        [({'submissions': [{**ROW, 'id': '1a'}], 'next': '1a'}, 200), ({'submissions': [{**ROW, 'id': '2b'}]}, 200)],
    )

    run_command(list_submissions)

    assert platform.paths.count(routes.SUBMISSIONS_LIST) == 2
    assert platform.request.url.params['after'] == '1a'
    assert capsys.readouterr().out.splitlines() == [
        'submission 1a 2026-03-04 05:06 running',
        'submission 2b 2026-03-04 05:06 running',
    ]


def test_a_response_the_client_cannot_read_is_a_refusal_without_the_body(platform, run_command):
    platform.answer_by_route({routes.SUBMISSIONS_LIST: ({'submissions': 'not a list', 'secret': 'x'}, 200)})
    with pytest.raises(SystemExit, match='cannot read: submissions') as caught:
        run_command(list_submissions)
    assert 'secret' not in str(caught.value)


def test_catalog_prints_the_evals_where_the_key_has_no_grant_for_the_tasks(platform, run_command, capsys):
    platform.answer_by_route({
        routes.CATALOG_EVALS: (
            {'evals': [{'id': 'fake.smoke', 'embodiment': 'franka', 'tasks': ['a'], 'composable': True}]},
            200,
        ),
        routes.CATALOG_TASKS: ({'error': {'code': 'forbidden', 'message': 'no customer grant'}}, 403),
    })

    run_command(catalog)

    out, err = capsys.readouterr()
    assert '"id": "fake.smoke"' in out
    assert err.strip() == 'tasks: forbidden: no customer grant'


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


def test_cancel_reports_a_run_the_platform_does_not_hold(platform, run_command):
    platform.answer_by_route({routes.SUBMISSIONS_CANCEL: NO_SUBMISSION})

    with pytest.raises(SystemExit, match='no such submission'):
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
