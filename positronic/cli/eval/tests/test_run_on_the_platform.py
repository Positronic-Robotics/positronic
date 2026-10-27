"""`positronic eval run` with a policy image: the run goes to the platform, over a stub transport."""

import sys
from datetime import UTC, datetime

import configuronic as cfn
import pytest
from platform_client import routes
from platform_client.enums import BoardVisibility, ErrorCode, OnExhausted, QuotaSubject, ReasonCode, SubmissionStatus
from platform_client.errors import EVALS_DETAIL, QUOTA_DETAIL, TASKS_DETAIL
from platform_client.ids import SubmissionId
from platform_client.responses import (
    QUOTA_SUBMISSIONS_DAY,
    BoardListResponse,
    BoardSummary,
    QuotaLimit,
    SubmissionCreateResponse,
)

from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.conftest import ID, KEY, SUBMISSION, refusal, runs_of_four
from positronic.cli.eval.run import run


def _created(status: SubmissionStatus, **fields) -> SubmissionCreateResponse:
    """What `submissions.create` answers for the submission `ID`."""
    return SubmissionCreateResponse(submission_id=SUBMISSION, status=status, **fields)


def test_a_policy_image_sends_the_run_to_the_platform(platform, run_command, capsys):
    platform.answer(_created(SubmissionStatus.pending, policy_image_digest='sha256:abc'))

    created = run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')

    # Returned as well as printed, so a caller holding the function has the id without scraping stdout.
    assert created.submission_id == SubmissionId.parse(ID)
    assert platform.request.url.path == routes.SUBMISSIONS_CREATE
    assert platform.request.headers['authorization'] == f'Bearer {KEY}'
    # A policy image is one endpoint of a plan on the websocket wire, and the eval names the tasks the
    # catalogue expands.
    body = platform.body
    assert body['eval'] == 'fake.smoke' and body['tasks'] == []
    assert [(entry['name'], entry['kind'], entry['wire'], entry['image']) for entry in body['endpoints']] == [
        ('policy', 'image', 'websocket', 'org/p:v1')
    ]
    assert body['alias'] is None and body['transaction_key'] == 'retry-1'
    out = capsys.readouterr().out
    assert f'submission {ID} (pending)' in out
    assert 'digest sha256:abc' in out


def test_an_eval_is_a_name_beside_a_file_of_that_name(platform, run_command, tmp_path, monkeypatch):
    # `--eval` reads as a name wherever the command runs, so a file of that name beside it changes nothing.
    platform.answer(_created(SubmissionStatus.pending))
    (tmp_path / 'fake.smoke').write_text('tasks: []\n')
    monkeypatch.chdir(tmp_path)

    run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert platform.body['eval'] == 'fake.smoke'


def test_an_image_rejected_at_the_door_fails_the_command(platform, run_command, capsys):
    # Terminal at the door and charged: a zero exit would let a script read it as a run that happened.
    platform.answer(_created(SubmissionStatus.errored, reason_code=ReasonCode.image_unpullable))

    with pytest.raises(CommandFailed, match='rejected: image_unpullable') as exit_info:
        run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert f'submission {ID} (errored)' in capsys.readouterr().out
    assert exit_info.value.fault is Fault.CALLER


def test_a_replay_of_a_cancelled_submission_fails_the_command(platform, run_command):
    # An idempotent replay returns the original, which may since have been cancelled. It will never
    # produce a result, so it exits like any other terminal-without-a-result.
    platform.answer(_created(SubmissionStatus.cancelled))

    with pytest.raises(SystemExit, match='rejected: cancelled'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')


@pytest.mark.parametrize(
    ('reason', 'fault'),
    [
        (ReasonCode.internal_error, Fault.PLATFORM),
        (ReasonCode.quota_exceeded, Fault.PLATFORM),
        (ReasonCode.provision_wedged, Fault.PLATFORM),
        (ReasonCode.runner_unresponsive, Fault.PLATFORM),
        (ReasonCode.policy_oom, Fault.CALLER),
        (ReasonCode.wall_clock_exceeded, Fault.CALLER),
    ],
    ids=lambda value: value.name,
)
def test_a_replay_of_a_failed_submission_exits_with_the_fault_of_its_reason(platform, run_command, reason, fault):
    platform.answer(_created(SubmissionStatus.errored, reason_code=reason))

    with pytest.raises(CommandFailed, match=f'rejected: {reason.name}') as exit_info:
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')

    assert exit_info.value.fault is fault


def test_a_replay_of_a_finished_submission_succeeds(platform, run_command, capsys):
    # The boundary of the rule above: terminal is not the test, a missing result is.
    platform.answer(_created(SubmissionStatus.finished))

    run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')

    assert f'submission {ID} (finished)' in capsys.readouterr().out


def test_an_eval_the_platform_does_not_offer_is_answered_with_the_ones_it_does(platform, run_command):
    # The set lives on the server, so the refusal is where a caller learns the real names.
    offered = {EVALS_DETAIL: ['fake.smoke', 'robolab.public_subset']}
    platform.answer(refusal(ErrorCode.not_found, "unknown eval 'fake.smokey'", **offered), status=404)

    with pytest.raises(SystemExit) as exit_info:
        run_command(run, eval='fake.smokey', policy_image='org/p:v1')

    assert f'  {EVALS_DETAIL}: ["fake.smoke", "robolab.public_subset"]' in str(exit_info.value)


def _boards(*evals: str) -> BoardListResponse:
    """What `rankings.list` answers: one public board for each of `evals`."""
    return BoardListResponse(
        boards=[
            BoardSummary.model_validate({
                'board': f'board-{index}',
                'title': f'Board {index}',
                'eval': name,
                'primary_metric': 'success_rate',
                'visibility': BoardVisibility.public.name,
            })
            for index, name in enumerate(evals)
        ]
    )


def test_a_run_that_names_no_eval_runs_the_eval_that_the_board_ranks(platform, run_command, capsys):
    platform.answer_by_route({
        routes.RANKINGS_LIST: (_boards('fake.board'), 200),
        routes.SUBMISSIONS_CREATE: (_created(SubmissionStatus.pending), 200),
    })

    run_command(run, policy_image='org/p:v1', transaction_key='retry-1')

    assert platform.paths == [routes.RANKINGS_LIST, routes.SUBMISSIONS_CREATE]
    assert platform.body['eval'] == 'fake.board'
    assert 'eval fake.board, which the board ranks' in capsys.readouterr().out


def test_two_boards_over_one_eval_name_that_eval(platform, run_command):
    platform.answer_by_route({
        routes.RANKINGS_LIST: (_boards('fake.board', 'fake.board'), 200),
        routes.SUBMISSIONS_CREATE: (_created(SubmissionStatus.pending), 200),
    })

    run_command(run, policy_image='org/p:v1')

    assert platform.body['eval'] == 'fake.board'


@pytest.mark.parametrize(
    ('evals', 'named'), [((), 'no eval'), (('fake.one', 'fake.two'), 'fake.one, fake.two')], ids=['none', 'two']
)
def test_boards_that_rank_no_single_eval_leave_the_eval_to_the_caller(platform, run_command, evals, named):
    platform.answer_by_route({routes.RANKINGS_LIST: (_boards(*evals), 200)})

    with pytest.raises(CommandFailed, match=f'the boards rank {named}: pass --eval=<name>') as exit_info:
        run_command(run, policy_image='org/p:v1')

    assert exit_info.value.fault is Fault.USAGE
    assert platform.paths == [routes.RANKINGS_LIST]


def test_a_run_that_names_its_eval_does_not_read_the_boards(platform, run_command):
    platform.answer(_created(SubmissionStatus.pending))

    run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert routes.RANKINGS_LIST not in platform.paths


def test_a_private_run_that_names_no_eval_is_refused_before_anything_is_sent(platform, run_command):
    with pytest.raises(CommandFailed, match='a private run names its eval') as exit_info:
        run_command(run, policy_image='org/p:v1', org='acme')

    assert exit_info.value.fault is Fault.USAGE
    assert platform.seen is None


@pytest.mark.parametrize(
    ('status', 'code', 'details', 'printed'),
    [
        (400, ErrorCode.bad_request, {TASKS_DETAIL: ['marker-in-mug']}, f'  {TASKS_DETAIL}: ["marker-in-mug"]'),
        (
            429,
            ErrorCode.quota_exceeded,
            {
                QUOTA_DETAIL: QuotaLimit(
                    key=QUOTA_SUBMISSIONS_DAY,
                    meter='submissions',
                    unit='submission',
                    scale=1,
                    window='day',
                    subject=QuotaSubject.user,
                    limit=3,
                    used=3,
                    resets_at=datetime(2026, 3, 5, tzinfo=UTC),
                    on_exhausted=OnExhausted.block,
                ).model_dump(mode='json')
            },
            '"2026-03-05T00:00:00Z"',
        ),
        (
            422,
            ErrorCode.bad_request,
            {'fields': [{'type': 'missing', 'loc': ['body', 'eval'], 'msg': 'Field required'}]},
            '"eval"',
        ),
    ],
    ids=['tasks-on-offer', 'quota', 'fields'],
)
def test_a_refusal_prints_the_details_the_platform_sent(platform, run_command, status, code, details, printed):
    # The details name the next step, so a refusal that drops them leaves the caller to guess it.
    platform.answer(refusal(code, 'refused', **details), status=status)

    with pytest.raises(SystemExit) as exit_info:
        run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert str(exit_info.value).startswith(f'{code.name}: refused\n')
    assert printed in str(exit_info.value)


def test_a_refusal_with_no_details_is_one_line(platform, run_command):
    platform.answer(refusal(ErrorCode.unauthorized, 'invalid api key'), status=401)

    with pytest.raises(SystemExit) as exit_info:
        run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert str(exit_info.value) == 'unauthorized: invalid api key'


@pytest.mark.parametrize(
    ('typo', 'named'),
    [
        ({'policy_image': 'org/policy@'}, 'org/policy@'),
        ({'eval': 'fake smoke'}, 'fake smoke'),
        ({'transaction_key': ''}, 'transaction_key'),
    ],
    ids=['image', 'eval', 'transaction-key'],
)
def test_a_value_the_wire_types_refuse_is_a_refusal_naming_it_rather_than_a_traceback(
    platform, run_command, typo: dict, named: str
):
    # These are refused in the caller's own process, before anything is sent, so the user learns
    # which value they typed wrong instead of reading a stack trace out of a constructor.
    with pytest.raises(SystemExit) as exit_info:
        run_command(run, **{'eval': 'fake.smoke', 'policy_image': 'org/p:v1', **typo})
    assert named in str(exit_info.value)
    assert platform.seen is None


@pytest.mark.parametrize(
    'platform_only',
    [{'alias': 'demo'}, {'transaction_key': 'k'}, {'platform_url': 'http://x.test'}, {'sample_count': 5}],
)
def test_a_local_run_refuses_what_only_a_platform_run_can_mean(platform, run_command, platform_only: dict):
    # The mirror of the check below it: no place may drop another place's arguments in silence.
    with pytest.raises(SystemExit, match='a local run has no'):
        run_command(run, eval='fake.smoke', policy='a policy', **platform_only)
    assert platform.seen is None


def test_the_sample_flags_state_the_sample_the_platform_draws(platform, run_command):
    platform.answer(_created(SubmissionStatus.pending))

    run_command(run, eval='fake.smoke', policy_image='org/p:v1', sample_percent=10, sample_seed=7)

    assert platform.body['sample'] == {'count': None, 'percent': 10, 'seed': 7}


def test_a_sample_seed_alone_is_refused_before_anything_is_sent(platform, run_command):
    with pytest.raises(SystemExit, match='exactly one of the two'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', sample_seed=7)
    assert platform.seen is None


def test_a_run_with_neither_a_policy_nor_an_image_says_which_it_wants(platform, run_command):
    with pytest.raises(SystemExit, match='--policy is required'):
        run_command(run, eval='fake.smoke')
    assert platform.seen is None


def test_a_run_cannot_be_both_here_and_there(platform, run_command):
    with pytest.raises(SystemExit, match='pass one'):
        run_command(run, eval='fake.smoke', policy='a policy', policy_image='org/p:v1')
    assert platform.seen is None


@pytest.mark.parametrize('local_only', [{'timing': True}, {'output_dir': '/tmp/x'}, {'charge_inference_time': False}])
def test_a_platform_run_refuses_what_only_a_local_run_can_mean(platform, run_command, local_only: dict):
    # The platform owns its own trial sweep, output and telemetry, so silently dropping these would
    # hand back a run the caller believes they configured.
    with pytest.raises(SystemExit, match='a platform run has no'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', **local_only)
    assert platform.seen is None


@pytest.mark.parametrize('already_so', [{'timing': False}, {'charge_inference_time': True}])
def test_a_platform_run_takes_a_local_switch_stated_at_what_it_already_does(platform, run_command, already_so: dict):
    # A switch reads the same whether it was left at its default or stated there, and either way asks
    # for what the platform already does, so it refuses neither.
    platform.answer(_created(SubmissionStatus.pending))

    created = run_command(run, eval='fake.smoke', policy_image='org/p:v1', **already_so)

    assert created.submission_id == SubmissionId.parse(ID)


def test_the_eval_group_walks_to_run(platform, capsys, monkeypatch):
    platform.answer(_created(SubmissionStatus.pending))
    argv = ['positronic', 'eval', 'run', '--eval=fake.smoke', '--policy-image=org/p:v1']
    monkeypatch.setattr(sys, 'argv', argv)

    cfn.cli({'eval': {'run': run}})

    assert platform.request.url.path == routes.SUBMISSIONS_CREATE
    assert f'submission {ID} (pending)' in capsys.readouterr().out


def a_password_file(tmp_path, password: str = 'the-registry-password\n'):
    path = tmp_path / 'registry-password'
    path.write_text(password)
    return path


def test_a_private_image_sends_the_credential_the_registry_asks_for(platform, run_command, tmp_path):
    # The caller names the file, and the password itself first appears in the request.
    platform.answer(_created(SubmissionStatus.pending))
    password_file = a_password_file(tmp_path)

    run_command(
        run,
        eval='fake.smoke',
        policy_image='org/p:v1',
        registry_username='a-reader',
        registry_password_file=str(password_file),
    )

    assert platform.body['endpoints'][0]['image_credential'] == {
        'username': 'a-reader',
        'password': 'the-registry-password',
    }


def test_a_public_image_sends_no_credential(platform, run_command):
    platform.answer(_created(SubmissionStatus.pending))

    run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert platform.body['endpoints'][0]['image_credential'] is None


def test_half_a_credential_is_refused_before_the_submission(platform, run_command, tmp_path):
    with pytest.raises(SystemExit, match='pass both'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', registry_username='a-reader')
    with pytest.raises(SystemExit, match='pass both'):
        run_command(
            run, eval='fake.smoke', policy_image='org/p:v1', registry_password_file=str(a_password_file(tmp_path))
        )
    assert platform.seen is None


def test_a_credential_naming_no_such_file_is_refused_before_the_submission(platform, run_command, tmp_path):
    with pytest.raises(
        SystemExit, match='--registry-password-file names no readable password: the password file path names no file'
    ):
        run_command(
            run,
            eval='fake.smoke',
            policy_image='org/p:v1',
            registry_username='a-reader',
            registry_password_file=str(tmp_path / 'never-written'),
        )
    assert platform.seen is None


@pytest.mark.parametrize('not_a_path', [1, [], ['a-pasted-password'], 70914233])
def test_a_password_file_flag_that_is_no_text_is_refused_without_printing_it(platform, run_command, not_a_path):
    with pytest.raises(SystemExit, match='--registry-password-file names no file: pass its path') as refusal:
        run_command(
            run,
            eval='fake.smoke',
            policy_image='org/p:v1',
            registry_username='a-reader',
            registry_password_file=not_a_path,
        )
    assert str(refusal.value) == '--registry-password-file names no file: pass its path'
    assert platform.seen is None


def test_an_empty_username_is_refused_as_the_username(platform, run_command, tmp_path):
    with pytest.raises(SystemExit, match='username') as refusal:
        run_command(
            run,
            eval='fake.smoke',
            policy_image='org/p:v1',
            registry_username='',
            registry_password_file=str(a_password_file(tmp_path)),
        )
    assert '--registry-password-file' not in str(refusal.value)
    assert platform.seen is None


def test_a_password_pasted_as_its_file_is_not_printed(platform, run_command):
    pasted = 'Zx9QvT7Lm2Rk'
    with pytest.raises(SystemExit, match='--registry-password-file names no readable password') as refusal:
        run_command(
            run, eval='fake.smoke', policy_image='org/p:v1', registry_username='a-reader', registry_password_file=pasted
        )
    message = str(refusal.value)
    assert not [run_of_the_password for run_of_the_password in runs_of_four(pasted) if run_of_the_password in message]
    assert platform.seen is None


def test_a_credential_naming_no_such_home_is_refused_before_the_submission(platform, run_command):
    with pytest.raises(SystemExit, match='names no home directory'):
        run_command(
            run,
            eval='fake.smoke',
            policy_image='org/p:v1',
            registry_username='a-reader',
            registry_password_file='~no-such-user-on-this-machine/registry-password',
        )
    assert platform.seen is None
