"""`positronic eval run` with a plan file: the plan is filed with the platform, over a stub transport."""

import json
from pathlib import Path

import pytest
from platform_client import routes
from platform_client.ids import SubmissionId

from positronic.cli.conftest import KEY
from positronic.cli.eval.plan import given
from positronic.cli.eval.run import run

SPOONS = 'eight-spoons-into-grey-tote'
FILED = {'submission_id': '2a', 'status': 'pending'}
BASELINE = {'host': 'baseline.example', 'port': 443, 'path': '/api/v1/session'}


def a_plan_file(directory: Path, name: str, payload: str) -> str:
    path = directory / name
    path.write_text(payload)
    return str(path)


PLAN_YAML = f"""
tasks:
  - {SPOONS}
endpoints:
  - name: baseline
    wire: websocket_tls
    address: {{host: baseline.example, port: 443, path: /api/v1/session}}
episodes_per_endpoint: 4
"""

PLAN_JSON = json.dumps({
    'tasks': [SPOONS],
    'endpoints': [{'name': 'baseline', 'wire': 'websocket_tls', 'address': BASELINE}],
    'episodes_per_endpoint': 4,
})


@pytest.mark.parametrize(('name', 'payload'), [('plan.yaml', PLAN_YAML), ('plan.json', PLAN_JSON)])
def test_a_plan_file_is_filed_whole(platform, run_command, tmp_path: Path, capsys, name: str, payload: str):
    platform.answer(FILED)

    filed = run_command(run, from_file=a_plan_file(tmp_path, name, payload))

    # Returned as well as printed, so a caller holding the function has the id without scraping stdout.
    assert filed.submission_id == SubmissionId(0x2A)
    assert platform.request.url.path == routes.SUBMISSIONS_CREATE
    assert platform.request.headers['authorization'] == f'Bearer {KEY}'
    assert platform.body['episodes_per_endpoint'] == 4
    assert [task['task_id'] for task in platform.body['tasks']] == [SPOONS]
    assert [(entry['wire'], entry['address']) for entry in platform.body['endpoints']] == [
        ('websocket_tls', {**BASELINE, 'query': ''})
    ]
    assert json.loads(capsys.readouterr().out)['submission_id'] == '2a'


def test_a_plan_file_naming_a_url_is_refused_before_it_is_filed(platform, run_command, tmp_path: Path):
    by_url = PLAN_YAML.replace(
        '    wire: websocket_tls\n    address: {host: baseline.example, port: 443, path: /api/v1/session}\n',
        '    url: wss://baseline.example/api/v1/session\n',
    )
    with pytest.raises(SystemExit, match='names a url'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', by_url))
    assert platform.seen is None


def test_a_plan_file_giving_a_key_twice_is_refused(platform, run_command, tmp_path: Path):
    # A YAML reader keeps the last of two equal keys, so the count the author meant would be lost.
    twice = PLAN_YAML + 'episodes_per_endpoint: 40\n'
    with pytest.raises(SystemExit, match="'episodes_per_endpoint' is given twice"):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', twice))


def test_a_transaction_key_beside_a_plan_file_is_filed_with_the_plan(platform, run_command, tmp_path: Path):
    platform.answer(FILED)

    run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), transaction_key='round-1')

    assert platform.body['transaction_key'] == 'round-1'


def test_a_plan_file_carrying_a_transaction_key_takes_no_flag(platform, run_command, tmp_path: Path):
    keyed = PLAN_YAML + 'transaction_key: round-1\n'
    with pytest.raises(SystemExit, match='carries transaction_key; drop --transaction-key'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', keyed), transaction_key='round-2')
    assert platform.seen is None


def test_an_alias_beside_a_plan_file_is_filed_with_the_plan(platform, run_command, tmp_path: Path):
    platform.answer(FILED)

    run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), alias='nightly')

    assert platform.body['alias'] == 'nightly'


def test_a_plan_file_carrying_an_alias_takes_no_flag(platform, run_command, tmp_path: Path):
    named = PLAN_YAML + 'alias: nightly\n'
    with pytest.raises(SystemExit, match='carries alias; drop --alias'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', named), alias='other')
    assert platform.seen is None


def test_a_plan_file_beside_a_policy_image_is_refused(platform, run_command, tmp_path: Path):
    # The platform runs an eval of its own by name, so it has no plan file to read.
    named = a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML)
    with pytest.raises(SystemExit, match='a platform run has no --from-file'):
        run_command(run, from_file=named, policy_image='org/p:v1')
    assert platform.seen is None


@pytest.mark.parametrize('eval', ['fake.smoke', 'plan.yaml'])
def test_an_eval_beside_a_plan_file_is_refused(platform, run_command, tmp_path: Path, eval: str):
    # A name and a second file alike: the file carries the whole plan, and `--eval` would be dropped.
    named = a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML)
    with pytest.raises(SystemExit, match='carries the whole plan; drop --eval'):
        run_command(run, from_file=named, eval=named if eval == 'plan.yaml' else eval)
    assert platform.seen is None


def test_a_file_that_is_not_a_plan_names_the_field(platform, run_command, tmp_path: Path):
    with pytest.raises(SystemExit, match='episodes_per_endpoint'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', f'tasks: [{SPOONS}]\n'))
    assert platform.seen is None


def test_a_file_that_reads_as_neither_yaml_nor_json_says_so(platform, run_command, tmp_path: Path):
    with pytest.raises(SystemExit, match='neither YAML nor JSON'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', 'tasks: [a\n  - ]]]\n'))
    assert platform.seen is None


def test_a_plan_file_that_is_not_there_names_it(platform, run_command, tmp_path: Path):
    with pytest.raises(SystemExit, match='absent.yaml'):
        run_command(run, from_file=str(tmp_path / 'absent.yaml'))
    assert platform.seen is None


def test_a_local_run_refuses_a_policy_wire(platform, run_command):
    with pytest.raises(SystemExit, match='a local run has no --policy-wire'):
        run_command(run, eval='fake.smoke', policy='a policy', policy_wire='websocket')
    assert platform.seen is None


@pytest.mark.parametrize(
    'elsewhere',
    [{'timing': True}, {'output_dir': '/tmp/x'}, {'charge_inference_time': False}, {'policy_wire': 'websocket'}],
)
def test_a_rig_run_refuses_what_only_another_place_can_mean(platform, run_command, tmp_path: Path, elsewhere: dict):
    # Each endpoint of a plan names its own wire, so a wire beside the file would be dropped.
    with pytest.raises(SystemExit, match='a rig run has no'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), **elsewhere)
    assert platform.seen is None


@pytest.mark.parametrize('switch', [{'timing': False}, {'charge_inference_time': True}])
def test_a_rig_run_takes_a_switch_stated_at_what_it_already_does(platform, run_command, tmp_path: Path, switch: dict):
    # A switch stated at what a rig run already does reads as one left at its default, so the rig
    # takes it.
    platform.answer(FILED)

    run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), **switch)

    assert platform.request.url.path == routes.SUBMISSIONS_CREATE


def test_a_run_naming_no_policy_is_told_the_three_places(platform, run_command):
    with pytest.raises(SystemExit, match='--policy is required'):
        run_command(run)
    assert platform.seen is None


@pytest.mark.parametrize(('value', 'is_given'), [(None, False), (False, True), (0, True), ('', True), (True, True)])
def test_a_flag_is_given_unless_it_is_unset(value: object, is_given: bool):
    assert given(value) is is_given
