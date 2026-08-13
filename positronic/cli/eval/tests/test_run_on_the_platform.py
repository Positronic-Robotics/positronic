"""`positronic eval run` with a policy image: the run goes to the platform, over a stub transport."""

import sys

import configuronic as cfn
import pytest
from platform_client import routes
from platform_client.ids import SubmissionId

from positronic.cli.conftest import ID, KEY
from positronic.cli.eval.run import run


def test_a_policy_image_sends_the_run_to_the_platform(platform, run_command, capsys):
    platform.answer({'submission_id': ID, 'status': 'pending', 'policy_image_digest': 'sha256:abc'})

    created = run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')

    # Returned as well as printed, so a caller holding the function has the id without scraping stdout.
    assert created.submission_id == SubmissionId.parse(ID)
    assert platform.request.url.path == routes.SUBMISSIONS_CREATE
    assert platform.request.headers['authorization'] == f'Bearer {KEY}'
    assert platform.body == {
        'policy_image': 'org/p:v1',
        'eval': 'fake.smoke',
        'alias': None,
        'transaction_key': 'retry-1',
    }
    out = capsys.readouterr().out
    assert f'submission {ID} (pending)' in out
    assert 'digest sha256:abc' in out


def test_an_image_rejected_at_the_door_fails_the_command(platform, run_command, capsys):
    # Terminal at the door and charged: a zero exit would let a script read it as a run that happened.
    platform.answer({'submission_id': ID, 'status': 'errored', 'reason_code': 'image_unpullable'})

    with pytest.raises(SystemExit, match='rejected: image_unpullable'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1')

    assert f'submission {ID} (errored)' in capsys.readouterr().out


def test_a_replay_of_a_cancelled_submission_fails_the_command(platform, run_command):
    # An idempotent replay returns the original, which may since have been cancelled. It will never
    # produce a result, so it exits like any other terminal-without-a-result.
    platform.answer({'submission_id': ID, 'status': 'cancelled'})

    with pytest.raises(SystemExit, match='rejected: cancelled'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')


def test_a_replay_of_a_finished_submission_succeeds(platform, run_command, capsys):
    # The boundary of the rule above: terminal is not the test, a missing result is.
    platform.answer({'submission_id': ID, 'status': 'finished'})

    run_command(run, eval='fake.smoke', policy_image='org/p:v1', transaction_key='retry-1')

    assert f'submission {ID} (finished)' in capsys.readouterr().out


def test_an_eval_the_platform_does_not_offer_is_answered_with_the_ones_it_does(platform, run_command):
    # The set lives on the server, so the refusal is where a caller learns the real names.
    platform.answer(
        {
            'error': {
                'code': 'not_found',
                'message': "unknown eval 'fake.smokey'",
                'details': {'evals': ['fake.smoke', 'robolab.public_subset']},
            }
        },
        status=404,
    )

    with pytest.raises(SystemExit) as exit_info:
        run_command(run, eval='fake.smokey', policy_image='org/p:v1')

    assert 'evals on offer: fake.smoke, robolab.public_subset' in str(exit_info.value)


def test_an_image_reference_the_registry_could_never_resolve_is_refused_here(platform, run_command):
    with pytest.raises(ValueError):
        run_command(run, eval='fake.smoke', policy_image='org/policy@')
    assert platform.seen is None


@pytest.mark.parametrize(
    'platform_only', [{'alias': 'demo'}, {'transaction_key': 'k'}, {'platform_url': 'http://x.test'}]
)
def test_a_local_run_refuses_what_only_a_platform_run_can_mean(platform, run_command, platform_only: dict):
    # The mirror of the check below it: neither half may drop the other's arguments in silence.
    with pytest.raises(SystemExit, match='a local run has no'):
        run_command(run, eval='fake.smoke', policy='a policy', **platform_only)
    assert platform.seen is None


def test_a_run_with_neither_a_policy_nor_an_image_says_which_it_wants(platform, run_command):
    with pytest.raises(SystemExit, match='--policy is required'):
        run_command(run, eval='fake.smoke')
    assert platform.seen is None


def test_a_run_cannot_be_both_here_and_there(platform, run_command):
    with pytest.raises(SystemExit, match='pass one'):
        run_command(run, eval='fake.smoke', policy='a policy', policy_image='org/p:v1')
    assert platform.seen is None


@pytest.mark.parametrize('local_only', [{'timing': True}, {'output_dir': '/tmp/x'}, {'inference_latency': True}])
def test_a_platform_run_refuses_what_only_a_local_run_can_mean(platform, run_command, local_only: dict):
    # The platform owns its own trial sweep, output and telemetry, so silently dropping these would
    # hand back a run the caller believes they configured.
    with pytest.raises(SystemExit, match='a platform run has no'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', **local_only)
    assert platform.seen is None


def test_the_eval_group_walks_to_run(platform, capsys, monkeypatch):
    platform.answer({'submission_id': ID, 'status': 'pending'})
    argv = ['positronic', 'eval', 'run', '--eval=fake.smoke', '--policy-image=org/p:v1']
    monkeypatch.setattr(sys, 'argv', argv)

    cfn.cli({'eval': {'run': run}})

    assert platform.request.url.path == routes.SUBMISSIONS_CREATE
    assert f'submission {ID} (pending)' in capsys.readouterr().out
