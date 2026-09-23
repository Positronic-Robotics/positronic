"""`positronic eval run` with a plan file: the plan is filed with the platform, over a stub transport."""

import json
from pathlib import Path

import pytest
from platform_client import routes
from platform_client.ids import SubmissionId

from positronic.cli.conftest import KEY, runs_of_four
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


# No run of this appears in a temp path or in the words a refusal is built from, so a match is the
# password and nothing else.
REGISTRY_PASSWORD = 'Zx9QvT7Lm2Rk'


# Each puts the password somewhere PyYAML quotes what it read: on the error's mark, or inside the
# message as the alias, anchor, tag or repeated key it could not resolve.
BROKEN_CREDENTIALS = {
    'given twice': f'      password: {REGISTRY_PASSWORD}\n      password: {REGISTRY_PASSWORD}\n',
    'a tab before the value': f'      password:\t{REGISTRY_PASSWORD}\n',
    'an unclosed quote': f'      password: "{REGISTRY_PASSWORD}\n',
    'a flow sequence left open': f'      password: [{REGISTRY_PASSWORD}\n',
    'an undefined alias': f'      password: *{REGISTRY_PASSWORD}\n',
    'a duplicate anchor': f'      password: &{REGISTRY_PASSWORD} x\n      other: &{REGISTRY_PASSWORD} y\n',
    'an unknown tag': f'      password: !{REGISTRY_PASSWORD} x\n',
    'a tag carrying a URI escape': f"      password: !a%22b'{REGISTRY_PASSWORD} x\n",
    'a repeated mapping key': f'      password:\n        {REGISTRY_PASSWORD}: a\n        {REGISTRY_PASSWORD}: b\n',
}


def a_plan_with_a_broken_credential(how: str) -> str:
    """A plan carrying a password a caller pasted into it, where the parser would quote it back.

    A plan names a password file, so this is a caller's mistake. The refusal still prints none of it.
    """
    broken = BROKEN_CREDENTIALS[how]
    return (
        f'tasks:\n  - {SPOONS}\nendpoints:\n  - name: baseline\n'
        '    image: registry.example/policy:v1\n    image_credential:\n      username: reader\n'
        f'{broken}episodes_per_endpoint: 4\n'
    )


@pytest.mark.parametrize('how', sorted(BROKEN_CREDENTIALS))
def test_a_malformed_plan_prints_no_part_of_its_registry_password(platform, run_command, tmp_path: Path, how: str):
    # A refusal is built from this file's own words and a position, so no run of the password can
    # reach it. Four characters catch a partial echo that a search for the whole string would miss.
    payload = a_plan_with_a_broken_credential(how)
    with pytest.raises(SystemExit) as refusal:
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    message = str(refusal.value)
    assert not [
        run_of_the_password for run_of_the_password in runs_of_four(REGISTRY_PASSWORD) if run_of_the_password in message
    ]
    assert platform.seen is None


def test_a_password_pasted_as_its_file_is_not_printed(platform, run_command, tmp_path: Path):
    payload = a_plan_with_credentials(a_credential_naming(REGISTRY_PASSWORD))
    with pytest.raises(SystemExit, match=r'image_credential\.password: .*names no file') as refusal:
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    message = str(refusal.value)
    assert not [
        run_of_the_password for run_of_the_password in runs_of_four(REGISTRY_PASSWORD) if run_of_the_password in message
    ]
    assert platform.seen is None


def test_a_malformed_plan_still_says_where_the_fault_is(platform, run_command, tmp_path: Path):
    # The redaction keeps the position: a refusal naming no line sends the author hunting.
    payload = a_plan_with_a_broken_credential('given twice')
    with pytest.raises(SystemExit, match=r'line \d+, column \d+'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))


def test_a_scalar_where_a_plan_lists_its_tasks_is_refused(platform, run_command, tmp_path: Path):
    payload = f'tasks: 1\nendpoints:\n  - name: baseline\n    url: {BASELINE}\nepisodes_per_endpoint: 4\n'
    with pytest.raises(SystemExit, match='tasks: Input should be a valid list'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    assert platform.seen is None


def a_plan_with_credentials(on_plan: str, on_task: str | None = None) -> str:
    """A plan whose image endpoint states `on_plan` as its credential, and whose task states `on_task`."""

    def endpoint(credential: str, indent: str) -> str:
        lines = [
            '- name: private',
            '  kind: image',
            '  wire: websocket',
            '  image: registry.example/policy:v1',
            '  image_credential:',
        ]
        lines += [f'    {line}' for line in credential.splitlines()]
        return ''.join(f'{indent}{line}\n' for line in lines)

    task = (
        f'  - {SPOONS}\n'
        if on_task is None
        else f'  - task_id: {SPOONS}\n    endpoints:\n' + endpoint(on_task, '      ')
    )
    return f'tasks:\n{task}endpoints:\n{endpoint(on_plan, "  ")}episodes_per_endpoint: 4\n'


def a_credential_naming(password_file: Path | str) -> str:
    return f'username: reader\npassword_file: {password_file}'


def test_a_plan_file_sends_the_password_each_of_its_files_holds(platform, run_command, tmp_path: Path):
    platform.answer(FILED)
    on_plan = tmp_path / 'plan-password'
    on_plan.write_text(f'{REGISTRY_PASSWORD}\n')
    on_task = tmp_path / 'task-password'
    on_task.write_text('the-task-password\n')

    run_command(
        run,
        from_file=a_plan_file(
            tmp_path, 'plan.yaml', a_plan_with_credentials(a_credential_naming(on_plan), a_credential_naming(on_task))
        ),
    )

    body = platform.body
    assert body['endpoints'][0]['image_credential'] == {'username': 'reader', 'password': REGISTRY_PASSWORD}
    assert body['tasks'][0]['endpoints'][0]['image_credential'] == {
        'username': 'reader',
        'password': 'the-task-password',
    }


@pytest.mark.parametrize('stated', ['[]', '1'])
def test_a_password_file_that_is_no_path_is_refused(platform, run_command, tmp_path: Path, stated: str):
    payload = a_plan_with_credentials(a_credential_naming(stated))
    with pytest.raises(SystemExit, match=r'image_credential\.password_file: Input is not a valid path'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    assert platform.seen is None


def test_a_plan_file_stating_a_password_is_refused_at_it(platform, run_command, tmp_path: Path):
    payload = a_plan_with_credentials(f'username: reader\npassword: {REGISTRY_PASSWORD}')
    with pytest.raises(SystemExit, match=r'image_credential\.password: Extra inputs are not permitted') as refusal:
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    assert REGISTRY_PASSWORD not in str(refusal.value)
    assert platform.seen is None


def test_a_password_file_under_no_such_home_is_refused(platform, run_command, tmp_path: Path):
    payload = a_plan_with_credentials(a_credential_naming('~no-such-user-on-this-machine/registry-password'))
    with pytest.raises(SystemExit, match='names no home directory'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    assert platform.seen is None


def test_a_password_file_that_is_not_there_is_refused_at_its_credential(platform, run_command, tmp_path: Path):
    on_plan = tmp_path / 'plan-password'
    on_plan.write_text(f'{REGISTRY_PASSWORD}\n')
    payload = a_plan_with_credentials(a_credential_naming(on_plan), a_credential_naming(tmp_path / 'never-written'))
    with pytest.raises(
        SystemExit, match=r'tasks\.0\.endpoints\.0\.image_credential\.password: .*names no file'
    ) as refusal:
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', payload))
    assert REGISTRY_PASSWORD not in str(refusal.value)
    assert platform.seen is None


def test_a_plan_file_that_is_not_there_names_it(platform, run_command, tmp_path: Path):
    with pytest.raises(SystemExit, match='absent.yaml'):
        run_command(run, from_file=str(tmp_path / 'absent.yaml'))
    assert platform.seen is None


@pytest.mark.parametrize('elsewhere', [{'timing': True}, {'output_dir': '/tmp/x'}, {'charge_inference_time': False}])
def test_a_rig_run_refuses_what_only_another_place_can_mean(platform, run_command, tmp_path: Path, elsewhere: dict):
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
