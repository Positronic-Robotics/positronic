"""`positronic eval status|list|cancel|catalog`: reading back what the platform is doing, over a stub transport.

The gateway plumbing every command shares — the key, the URL, a refusal — is checked here, on the
commands that require a key.
"""

import json

import httpx
import pytest
from platform_client import routes
from platform_client.catalog import EvalListResponse, EvalSummary, TaskListResponse, TaskSummary
from platform_client.client import PlatformClient
from platform_client.enums import ErrorCode, ReasonCode, RigShape, SubmissionStatus
from platform_client.ids import SubmissionId
from platform_client.requests import CancelRequest
from platform_client.responses import (
    ArtifactRefs,
    BlockedSubmissionView,
    CancelResponse,
    EpisodeCounts,
    ErroredSubmissionView,
    FinishedSubmissionView,
    PendingSubmissionView,
    ReplayLink,
    ReportLink,
    Scores,
    SubmissionListResponse,
    SubmissionListRow,
    SubmissionView,
)
from platform_client.tasks import TaskRef
from positronic_evals.evals import EvalRef
from pydantic import TypeAdapter

from positronic.cli import _internal_main
from positronic.cli.account import gateway as gateway_module
from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.conftest import AT, BASE, ID, SUBMISSION, USER, WHEN, refusal
from positronic.cli.eval.submissions import CatalogView, cancel, catalog, list_submissions, status

PENDING = PendingSubmissionView(id=SUBMISSION, received_at=WHEN, queued_at=WHEN, queue_position=3)
PARTIAL = ErroredSubmissionView(
    id=SUBMISSION,
    reason_code=ReasonCode.policy_inference_crash,
    reason='the policy session dropped; 6 of 12 episodes were published and scored',
    scores=Scores(primary=0.5),
    episodes=EpisodeCounts(total=12, done=6, outstanding=6),
)


def test_status_prints_the_fields_of_the_variant_it_got(platform, run_command, capsys):
    platform.answer(PENDING)

    run_command(status, id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_GET
    assert platform.request.url.params['id'] == ID
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'submission {ID} pending'
    assert '  queue_position: 3' in out
    # The header carries the id and the status, so the body below it repeats neither.
    assert [line for line in out[1:] if line.startswith(('  id:', '  status:'))] == []


def test_status_prints_the_score_of_a_failed_run_over_the_episodes_that_closed(platform, run_command, capsys):
    platform.answer(PARTIAL)

    run_command(status, id=ID)

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'submission {ID} errored'
    assert "  scores: {'primary': 0.5}" in out
    assert "  episodes: {'total': 12, 'done': 6, 'outstanding': 6}" in out


def test_status_prints_no_score_for_a_failed_run_that_published_no_episodes(platform, run_command, capsys):
    platform.answer(ErroredSubmissionView(id=SUBMISSION, reason_code=ReasonCode.policy_oom, reason='out of memory'))

    run_command(status, id=ID)

    assert capsys.readouterr().out.splitlines() == [
        f'submission {ID} errored',
        "  episodes: {'total': 0, 'done': 0, 'outstanding': 0}",
        '  runs: []',
        '  resolved: None',
        '  reason_code: policy_oom',
        '  reason: out of memory',
        '  artifacts: None',
        '  scores: None',
    ]


NO_SUBMISSION = (refusal(ErrorCode.not_found, 'no such submission'), 404)


def _row(id: str = ID, **fields) -> SubmissionListRow:
    """One running row of `submissions.list`, with `fields` added."""
    return SubmissionListRow(
        id=SubmissionId.parse(id), user_id=USER, status=SubmissionStatus.running, received_at=WHEN, **fields
    )


def test_status_prints_what_a_blocked_run_waits_on(platform, run_command, capsys):
    platform.answer(BlockedSubmissionView(id=SUBMISSION, reason='the rig is not ready'))

    run_command(status, id=ID)

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f'submission {ID} blocked'
    assert '  reason: the rig is not ready' in out


def test_status_prints_the_replay_and_the_report_of_a_finished_run(platform, run_command, capsys):
    replay, report = 'https://replays.example/r/token/', 'https://reports.example/r/token/report.html'
    platform.answer(
        FinishedSubmissionView(
            id=SUBMISSION,
            artifacts=ArtifactRefs(result='s3://b/episodes/'),
            replay=ReplayLink(url=replay, expires_at=WHEN),
            report=ReportLink(url=report, expires_at=WHEN),
        )
    )

    run_command(status, id=ID)

    out = capsys.readouterr().out.splitlines()
    assert f"  replay: {{'url': '{replay}', 'expires_at': '{AT}'}}" in out
    assert f"  report: {{'url': '{report}', 'expires_at': '{AT}'}}" in out


def test_list_prints_one_labelled_line_per_run(platform, run_command, capsys):
    # One row of each kind: the platform-run row and the rig row that carries a count.
    platform.answer(
        SubmissionListResponse(
            submissions=[
                _row(alias='demo', eval=EvalRef('fake.smoke')),
                _row('1a', episodes=EpisodeCounts(total=20, done=0, outstanding=20)),
            ]
        )
    )

    run_command(list_submissions)

    assert platform.request.url.path == routes.SUBMISSIONS_LIST
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f'submission {ID} 2026-03-04 05:06 running fake.smoke demo',
        'submission 1a 2026-03-04 05:06 running 0/20 episodes',
    ]


FIRST_PAGE = SubmissionListResponse(submissions=[_row('1a')], next=SubmissionId.parse('1a'))
LAST_PAGE = SubmissionListResponse(submissions=[_row('2b')])


def test_list_reads_every_page(platform, run_command, capsys):
    platform.answer_in_turn(routes.SUBMISSIONS_LIST, [(FIRST_PAGE, 200), (LAST_PAGE, 200)])

    run_command(list_submissions)

    assert platform.paths.count(routes.SUBMISSIONS_LIST) == 2
    assert platform.request.url.params['after'] == '1a'
    assert capsys.readouterr().out.splitlines() == [
        'submission 1a 2026-03-04 05:06 running',
        'submission 2b 2026-03-04 05:06 running',
    ]


def test_a_response_the_client_cannot_read_is_a_refusal_without_the_body(platform, run_command):
    # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
    platform.answer_by_route({routes.SUBMISSIONS_LIST: ({'submissions': 'not a list', 'secret': 'x'}, 200)})
    with pytest.raises(CommandFailed, match='cannot read: list_type') as caught:
        run_command(list_submissions)
    assert 'secret' not in str(caught.value)
    assert caught.value.fault is Fault.PLATFORM


EVAL = EvalSummary(id=EvalRef('fake.smoke'), embodiment='franka', tasks=['a'], composable=True)
TASK = TaskSummary(id=TaskRef('eight-spoons-into-grey-tote'), embodiments=[RigShape.franka], task='spoons')
EVALS_ANSWER = (EvalListResponse(evals=[EVAL]), 200)
TASKS_ANSWER = (TaskListResponse(tasks=[TASK]), 200)
NO_GRANT = (refusal(ErrorCode.forbidden, 'no customer grant'), 403)


def test_catalog_prints_the_evals_where_the_key_has_no_grant_for_the_tasks(platform, run_command, capsys):
    platform.answer_by_route({routes.CATALOG_EVALS: EVALS_ANSWER, routes.CATALOG_TASKS: NO_GRANT})

    run_command(catalog)

    out, err = capsys.readouterr()
    assert EvalListResponse.model_validate_json(out).evals == [EVAL]
    assert err.strip() == 'tasks: forbidden: no customer grant'


def test_catalog_prints_what_the_key_may_name(platform, run_command, capsys):
    two_arm = TaskSummary(id=TaskRef('fold-the-towel'), embodiments=[RigShape.yam], task='towel')
    tasks = TaskListResponse(tasks=[TASK, two_arm])
    platform.answer_by_route({routes.CATALOG_EVALS: EVALS_ANSWER, routes.CATALOG_TASKS: (tasks, 200)})

    run_command(catalog)

    assert platform.request.url.path == routes.CATALOG_TASKS
    decoder = json.JSONDecoder()
    out = capsys.readouterr().out
    evals, end = decoder.raw_decode(out)
    printed_tasks, _ = decoder.raw_decode(out[end:].lstrip())
    assert EvalListResponse.model_validate(evals).evals == [EVAL]
    assert [task['embodiments'] for task in printed_tasks['tasks']] == [['franka'], ['yam']]


def test_catalog_json_prints_the_evals_and_the_tasks_as_one_value(platform, run_command, capsys):
    platform.answer_by_route({routes.CATALOG_EVALS: EVALS_ANSWER, routes.CATALOG_TASKS: TASKS_ANSWER})

    run_command(catalog, json=True)

    assert CatalogView.model_validate_json(capsys.readouterr().out) == CatalogView(evals=[EVAL], tasks=[TASK])


def test_catalog_json_prints_null_tasks_where_the_key_has_no_grant_for_them(platform, run_command, capsys):
    platform.answer_by_route({routes.CATALOG_EVALS: EVALS_ANSWER, routes.CATALOG_TASKS: NO_GRANT})

    run_command(catalog, json=True)

    out, err = capsys.readouterr()
    assert CatalogView.model_validate_json(out) == CatalogView(evals=[EVAL], tasks=None)
    assert err.strip() == 'tasks: forbidden: no customer grant'


def test_cancel_reports_whether_the_quota_came_back(platform, run_command, capsys):
    platform.answer(CancelResponse(status=SubmissionStatus.cancelled, refunded=True))

    run_command(cancel, id=ID)

    assert platform.request.url.path == routes.SUBMISSIONS_CANCEL
    assert CancelRequest.model_validate(platform.body) == CancelRequest(id=SUBMISSION)
    assert 'cancelled, quota refunded' in capsys.readouterr().out


def test_cancel_reports_a_run_the_platform_does_not_hold(platform, run_command):
    platform.answer_by_route({routes.SUBMISSIONS_CANCEL: NO_SUBMISSION})

    with pytest.raises(SystemExit, match='no such submission'):
        run_command(cancel, id=ID)


def test_a_refusal_that_is_not_a_missing_record_is_not_read_as_the_other_kind(platform, run_command):
    platform.answer(refusal(ErrorCode.forbidden, 'no grant'), status=403)

    with pytest.raises(SystemExit, match='forbidden'):
        run_command(status, id=ID)


def test_an_id_the_parser_read_as_a_number_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='hexadecimal'):
        run_command(status, id=1234567890123456)


def test_an_id_that_is_not_hexadecimal_is_refused(platform, run_command):
    with pytest.raises(SystemExit, match='not an id'):
        run_command(cancel, id='zz')


def test_a_refusal_by_the_platform_exits_with_its_message(platform, run_command):
    platform.answer(refusal(ErrorCode.quota_exceeded, 'no submissions left today'), status=429)

    with pytest.raises(SystemExit) as exit_info:
        run_command(list_submissions)

    assert str(exit_info.value) == 'quota_exceeded: no submissions left today'


def test_a_command_needing_a_key_names_the_variable_that_holds_it(platform, run_command, monkeypatch):
    monkeypatch.delenv(gateway_module.API_KEY_ENV)

    with pytest.raises(CommandFailed, match=gateway_module.API_KEY_ENV) as exit_info:
        run_command(list_submissions)

    assert exit_info.value.fault is Fault.CALLER


@pytest.mark.parametrize(
    ('status_code', 'code', 'fault'),
    [
        (400, ErrorCode.bad_request, Fault.CALLER),
        (401, ErrorCode.unauthorized, Fault.CALLER),
        (404, ErrorCode.not_found, Fault.CALLER),
        (429, ErrorCode.quota_exceeded, Fault.CALLER),
        (500, ErrorCode.internal_error, Fault.PLATFORM),
        (503, ErrorCode.upstream_unavailable, Fault.PLATFORM),
        # Each half decides alone: a 5xx with a caller's code, and a 4xx with the platform's own.
        (503, ErrorCode.bad_request, Fault.PLATFORM),
        (409, ErrorCode.internal_error, Fault.PLATFORM),
    ],
)
def test_a_refusal_exits_with_the_fault_its_status_and_code_name(platform, run_command, status_code, code, fault):
    platform.answer(refusal(code), status=status_code)

    with pytest.raises(CommandFailed) as exit_info:
        run_command(status, id=ID)

    assert exit_info.value.fault is fault


def test_a_failure_without_an_envelope_is_the_platforms_fault(platform, run_command):
    # A proxy in front of the platform answers in its own shape.
    platform.answer('<html>502 Bad Gateway</html>', status=502)

    with pytest.raises(CommandFailed) as exit_info:
        run_command(status, id=ID)

    assert exit_info.value.fault is Fault.PLATFORM


@pytest.mark.parametrize('as_json', [False, True], ids=['text', 'json'])
def test_a_failure_without_an_envelope_prints_no_body_and_no_envelope(platform, run_command, capsys, as_json):
    # A proxy's whole page is not a detail of a platform refusal, and no platform envelope exists to print.
    platform.answer('<html>502 Bad Gateway</html>', status=502)

    with pytest.raises(CommandFailed) as exit_info:
        run_command(status, id=ID, json=as_json)

    assert str(exit_info.value) == 'internal_error: unparseable error response (HTTP 502)'
    assert exit_info.value.fault is Fault.PLATFORM
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize(
    'no_answer', [httpx.ConnectError('connection refused'), httpx.ReadTimeout('timed out')], ids=['refused', 'timeout']
)
def test_a_platform_that_does_not_answer_is_the_platforms_fault(platform, run_command, no_answer):
    platform.no_answer = no_answer

    with pytest.raises(CommandFailed, match=f'{BASE} did not answer') as exit_info:
        run_command(status, id=ID)

    assert exit_info.value.fault is Fault.PLATFORM


def test_an_argument_refused_before_any_call_is_a_usage_fault(platform, run_command):
    with pytest.raises(CommandFailed) as exit_info:
        run_command(status, id='zz')

    assert exit_info.value.fault is Fault.USAGE
    assert platform.seen is None


@pytest.mark.parametrize(
    ('argv', 'answer', 'fault', 'printed'),
    [
        (['--id=zz'], None, Fault.USAGE, 'not an id'),
        ([f'--id={ID}'], NO_SUBMISSION, Fault.CALLER, 'no such submission'),
        ([f'--id={ID}'], (refusal(ErrorCode.internal_error, 'boom'), 500), Fault.PLATFORM, 'boom'),
    ],
    ids=['usage', 'caller', 'platform'],
)
def test_positronic_exits_with_the_fault_as_its_status(platform, monkeypatch, capsys, argv, answer, fault, printed):
    if answer is not None:
        platform.answer(answer[0], status=answer[1])
    monkeypatch.setattr('sys.argv', ['positronic', 'eval', 'status', *argv])

    with pytest.raises(SystemExit) as exit_info:
        _internal_main()

    assert exit_info.value.code == fault
    assert printed in capsys.readouterr().err


UNPARSEABLE_URL = 'http://gateway:bad'


@pytest.mark.parametrize('named_in', ['argument', 'environment'])
def test_a_platform_url_that_does_not_parse_exits_with_the_usage_status(platform, monkeypatch, capsys, named_in):
    # The real client parses the URL before any call, so the stub transport stays out of it.
    monkeypatch.setattr(gateway_module, 'PlatformClient', PlatformClient)
    argv = ['positronic', 'eval', 'status', f'--id={ID}']
    if named_in == 'argument':
        argv.append(f'--platform_url={UNPARSEABLE_URL}')
    else:
        monkeypatch.setenv(gateway_module.API_URL_ENV, UNPARSEABLE_URL)
    monkeypatch.setattr('sys.argv', argv)

    with pytest.raises(SystemExit) as exit_info:
        _internal_main()

    assert exit_info.value.code == Fault.USAGE
    assert UNPARSEABLE_URL in capsys.readouterr().err


def test_status_json_prints_the_view_the_platform_answered(platform, run_command, capsys):
    platform.answer(PENDING)

    run_command(status, id=ID, json=True)

    assert TypeAdapter(SubmissionView).validate_json(capsys.readouterr().out) == PENDING


def test_status_json_keeps_the_score_of_a_failed_run(platform, run_command, capsys):
    platform.answer(PARTIAL)

    run_command(status, id=ID, json=True)

    assert TypeAdapter(SubmissionView).validate_json(capsys.readouterr().out) == PARTIAL


def test_list_json_prints_every_page_as_one(platform, run_command, capsys):
    platform.answer_in_turn(routes.SUBMISSIONS_LIST, [(FIRST_PAGE, 200), (LAST_PAGE, 200)])

    run_command(list_submissions, json=True)

    page = SubmissionListResponse.model_validate_json(capsys.readouterr().out)
    assert [row.id for row in page.submissions] == [SubmissionId.parse('1a'), SubmissionId.parse('2b')]
    assert page.next is None


@pytest.mark.parametrize(
    ('command', 'args'),
    [(status, {'id': ID}), (list_submissions, {}), (catalog, {})],
    ids=['status', 'list', 'catalog'],
)
@pytest.mark.parametrize(('status_code', 'fault'), [(404, Fault.CALLER), (503, Fault.PLATFORM)])
def test_a_refusal_under_json_prints_its_envelope(platform, run_command, capsys, command, args, status_code, fault):
    envelope = refusal(ErrorCode.not_found, eval='none')
    platform.answer(envelope, status=status_code)

    with pytest.raises(SystemExit) as exit_info:
        run_command(command, json=True, **args)

    out, err = capsys.readouterr()
    assert json.loads(out) == envelope
    assert err == ''
    assert exit_info.value.code == fault


def test_an_unconfigured_url_leaves_the_client_on_its_default_platform(platform, run_command, monkeypatch):
    # A user should never have to know a URL: with nothing set, the client reaches the platform.
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer(SubmissionListResponse())

    run_command(list_submissions)

    assert platform.base_url is None


def test_the_platform_url_argument_overrides_the_environment(platform, run_command):
    platform.answer(SubmissionListResponse())

    run_command(list_submissions, platform_url='http://other.test')

    assert platform.base_url == 'http://other.test'
