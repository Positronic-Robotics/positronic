"""`positronic eval run` with a policy URL: the plan is filed with the platform, over a stub transport."""

import json
from pathlib import Path

import pytest
from platform_client import routes
from platform_client.enums import CameraVantage, Placement
from platform_client.eval_plan import EvalPlan
from platform_client.ids import PlanId

from positronic.cli.conftest import KEY
from positronic.cli.eval.plan import endpoint_of, flag_entries, given, scene_from_pairs
from positronic.cli.eval.run import run

SPOONS = 'eight-spoons-into-grey-tote'
MUG = 'marker-in-mug'
BASELINE = 'wss://baseline.example/ws'
CANDIDATE = 'wss://candidate.example/ws'
FILED = {'plan_id': '2a', 'status': 'received'}

FLAGS = {
    'policy_url': f'baseline={BASELINE},candidate={CANDIDATE}',
    'tasks': f'{SPOONS},{MUG}',
    'episodes': 10,
    'cap': 180,
    'preset': 'example_candidate',
    'scene': 'tote_placement=random,camera.side=left',
}


def a_plan_file(directory: Path, name: str, payload: str) -> str:
    path = directory / name
    path.write_text(payload)
    return str(path)


PLAN_YAML = f"""
tasks:
  - {SPOONS}
endpoints:
  - name: baseline
    url: {BASELINE}
episodes_per_endpoint: 4
"""


def test_the_flags_state_a_plan_and_it_is_filed(platform, run_command, capsys):
    platform.answer(FILED)

    filed = run_command(run, **FLAGS)

    # Returned as well as printed, so a caller holding the function has the id without scraping stdout.
    assert filed.plan_id == PlanId(0x2A)
    assert platform.request.url.path == routes.EVALS_RUN
    assert platform.request.headers['authorization'] == f'Bearer {KEY}'
    body = platform.body
    assert [task['task_id'] for task in body['tasks']] == [SPOONS, MUG]
    assert [(entry['name'], entry['url']) for entry in body['endpoints']] == [
        ('baseline', BASELINE),
        ('candidate', CANDIDATE),
    ]
    assert body['episodes_per_endpoint'] == 10
    assert body['cap_per_episode_sec'] == 180
    assert body['policy_preset'] == 'example_candidate'
    assert body['tote_placement'] == 'random'
    assert body['external_cameras'] == {'side': 'left'}
    assert json.loads(capsys.readouterr().out)['plan_id'] == '2a'


def test_a_bracketed_list_states_the_same_plan_as_the_comma_form(platform, run_command):
    # A CLI value is literal-evaluated, so `[a,b]` arrives as a list where the entries read as
    # names and as text where they do not. Both spell one plan.
    platform.answer(FILED)
    run_command(
        run, **{**FLAGS, 'tasks': [SPOONS, MUG], 'policy_url': [f'baseline={BASELINE}', f'candidate={CANDIDATE}']}
    )
    from_list = platform.body

    platform.answer(FILED)
    run_command(run, **FLAGS)

    assert from_list == platform.body


def test_a_bare_policy_url_is_named_for_its_place_in_the_list(platform, run_command):
    platform.answer(FILED)

    run_command(run, policy_url=f'{BASELINE},{CANDIDATE}', tasks=SPOONS, episodes=2)

    assert [(entry['name'], entry['url']) for entry in platform.body['endpoints']] == [
        ('policy1', BASELINE),
        ('policy2', CANDIDATE),
    ]


def test_a_url_carrying_a_query_is_not_read_as_a_label(platform, run_command):
    # A URL takes `=` in its query, so the part before the first one labels an endpoint only where
    # it names no scheme and no path.
    platform.answer(FILED)

    run_command(run, policy_url='wss://h/ws?mode=native', tasks=SPOONS, episodes=1)

    assert platform.body['endpoints'][0] == {
        'name': 'policy1',
        'kind': 'remote',
        'url': 'wss://h/ws?mode=native',
        'provider': None,
        'spec': None,
        'episodes_per_endpoint': None,
        'cap_per_episode_sec': None,
        'policy_preset': None,
        'tote_placement': None,
        'camera_vantage': None,
        'external_cameras': {},
        'clutter': None,
    }


PLAN_JSON = json.dumps({
    'tasks': [SPOONS],
    'endpoints': [{'name': 'baseline', 'url': BASELINE}],
    'episodes_per_endpoint': 4,
})


@pytest.mark.parametrize(('name', 'payload'), [('plan.yaml', PLAN_YAML), ('plan.json', PLAN_JSON)])
def test_a_plan_file_is_filed_whole(platform, run_command, tmp_path: Path, name: str, payload: str):
    platform.answer(FILED)

    run_command(run, from_file=a_plan_file(tmp_path, name, payload))

    assert platform.request.url.path == routes.EVALS_RUN
    assert platform.body['episodes_per_endpoint'] == 4
    assert [task['task_id'] for task in platform.body['tasks']] == [SPOONS]


def test_a_plan_file_giving_a_key_twice_is_refused(platform, run_command, tmp_path: Path):
    # A YAML reader keeps the last of two equal keys, so the count the author meant would be lost.
    twice = PLAN_YAML + 'episodes_per_endpoint: 40\n'
    with pytest.raises(SystemExit, match="'episodes_per_endpoint' is given twice"):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', twice))


def test_an_eval_naming_an_existing_file_is_that_file(platform, run_command, tmp_path: Path):
    platform.answer(FILED)

    run_command(run, eval=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML))

    assert platform.request.url.path == routes.EVALS_RUN
    assert platform.body['episodes_per_endpoint'] == 4


def test_a_plan_file_beside_a_plan_flag_is_refused(platform, run_command, tmp_path: Path):
    # One source states the plan; a flag beside a file would be ignored.
    with pytest.raises(SystemExit, match='carries the whole plan'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), episodes=3)
    assert platform.seen is None


def test_a_transaction_key_beside_a_plan_file_is_filed_with_the_plan(platform, run_command, tmp_path: Path):
    platform.answer(FILED)

    run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML), transaction_key='round-1')

    assert platform.body['transaction_key'] == 'round-1'


def test_a_plan_file_carrying_a_transaction_key_takes_no_flag(platform, run_command, tmp_path: Path):
    keyed = PLAN_YAML + 'transaction_key: round-1\n'
    with pytest.raises(SystemExit, match='carries transaction_key; drop --transaction-key'):
        run_command(run, from_file=a_plan_file(tmp_path, 'plan.yaml', keyed), transaction_key='round-2')
    assert platform.seen is None


def test_a_plan_file_beside_a_policy_image_is_refused(platform, run_command, tmp_path: Path):
    named = a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML)
    with pytest.raises(SystemExit, match='names a plan file, and --policy-image'):
        run_command(run, eval=named, policy_image='org/p:v1')
    assert platform.seen is None


def test_two_plan_files_are_refused(platform, run_command, tmp_path: Path):
    named = a_plan_file(tmp_path, 'plan.yaml', PLAN_YAML)
    with pytest.raises(SystemExit, match='each name a plan file'):
        run_command(run, from_file=named, eval=named)
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


def test_an_eval_name_that_is_no_file_is_refused_on_the_rig(platform, run_command):
    # The rig runs a plan, and `EvalPlan` carries no eval name, so a name reaches nothing there.
    with pytest.raises(SystemExit, match='names no plan file'):
        run_command(run, eval='fake.smoke', policy_url=BASELINE, tasks=SPOONS, episodes=1)
    assert platform.seen is None


def test_a_rig_run_states_its_tasks_its_endpoints_and_its_count(platform, run_command):
    with pytest.raises(SystemExit, match='--tasks, --policy-url and --episodes'):
        run_command(run, policy_url=BASELINE, episodes=1)
    assert platform.seen is None


@pytest.mark.parametrize('rig_only', [{'policy_url': BASELINE}, {'tasks': SPOONS}, {'episodes': 4}, {'cap': 60}])
def test_a_local_run_refuses_what_only_a_rig_run_can_mean(platform, run_command, rig_only: dict):
    with pytest.raises(SystemExit, match='a local run has no'):
        run_command(run, eval='fake.smoke', policy='a policy', **rig_only)
    assert platform.seen is None


@pytest.mark.parametrize(
    'rig_only',
    [{'policy_url': BASELINE}, {'preset': 'p'}, {'scene': 'tote_placement=left'}, {'episodes': 0}, {'cap': 0}],
)
def test_a_platform_run_refuses_what_only_a_rig_run_can_mean(platform, run_command, rig_only: dict):
    with pytest.raises(SystemExit, match='a platform run has no'):
        run_command(run, eval='fake.smoke', policy_image='org/p:v1', **rig_only)
    assert platform.seen is None


@pytest.mark.parametrize('elsewhere', [{'timing': True}, {'output_dir': '/tmp/x'}, {'alias': 'demo'}])
def test_a_rig_run_refuses_what_only_another_place_can_mean(platform, run_command, elsewhere: dict):
    with pytest.raises(SystemExit, match='a rig run has no'):
        run_command(run, policy_url=BASELINE, tasks=SPOONS, episodes=1, **elsewhere)
    assert platform.seen is None


def test_a_run_naming_no_policy_is_told_the_three_places(platform, run_command):
    with pytest.raises(SystemExit, match='--policy is required'):
        run_command(run)
    assert platform.seen is None


def test_a_transaction_key_makes_a_retry_return_the_first_plan(platform, run_command):
    platform.answer(FILED)

    run_command(run, **FLAGS, transaction_key='round-1')

    assert platform.body['transaction_key'] == 'round-1'


def test_a_task_id_that_could_never_be_a_catalogue_key_ends_the_command(platform, run_command):
    with pytest.raises(SystemExit, match='not a task id'):
        run_command(run, policy_url=BASELINE, tasks='Eight Spoons', episodes=1)
    assert platform.seen is None


# --- the flag readers ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'entries'),
    [
        ('a-b', ['a-b']),
        ('a-b,c-d', ['a-b', 'c-d']),
        ('[a-b,c-d]', ['a-b', 'c-d']),
        (['a-b', 'c-d'], ['a-b', 'c-d']),
        (None, []),
    ],
)
def test_a_repeatable_flag_reads_every_spelling_the_command_line_reaches(value: object, entries: list[str]):
    assert flag_entries(value, '--tasks') == entries


@pytest.mark.parametrize('value', ['a,,b', ' '])
def test_a_repeatable_flag_refuses_an_empty_entry(value: str):
    with pytest.raises(SystemExit, match='empty entry'):
        flag_entries(value, '--tasks')


def test_a_repeatable_flag_read_as_a_number_is_refused():
    with pytest.raises(SystemExit, match='quote'):
        flag_entries(10, '--tasks')


@pytest.mark.parametrize(('value', 'is_given'), [(None, False), (False, False), (0, True), ('', True), (True, True)])
def test_a_flag_is_given_unless_it_is_unset_or_a_switch_left_off(value: object, is_given: bool):
    assert given(value) is is_given


def test_a_labelled_url_takes_its_label():
    assert endpoint_of(f'baseline={BASELINE}', 1) == {'name': 'baseline', 'url': BASELINE}
    assert endpoint_of(BASELINE, 3) == {'name': 'policy3', 'url': BASELINE}


def test_scene_pairs_become_the_plans_own_fields():
    # The reader validates each side here, so a value outside the closed set never reaches the wire.
    assert scene_from_pairs(['tote_placement=random', 'camera_vantage=droid', 'camera.side=left']) == {
        'tote_placement': Placement.random,
        'camera_vantage': CameraVantage.droid,
        'external_cameras': {'side': Placement.left},
    }
    assert scene_from_pairs([]) == {}


@pytest.mark.parametrize('pair', ['tote_placement', 'tote=left', 'camera.=left', 'tote_placement=middle'])
def test_a_scene_pair_that_names_no_field_or_no_side_is_refused(pair: str):
    with pytest.raises(SystemExit):
        scene_from_pairs([pair])


@pytest.mark.parametrize('pairs', [['camera.side=left', 'side=right'], ['side=right', 'camera.side=left']])
def test_a_mount_spelled_like_a_scene_field_is_not_that_field(pairs: list[str]):
    # `side` is no scene field, so the refusal names it; a mount of that name is a different key.
    with pytest.raises(SystemExit, match="not 'side'"):
        scene_from_pairs(pairs)


@pytest.mark.parametrize(
    'pairs',
    [['camera.tote_placement=left', 'tote_placement=right'], ['tote_placement=right', 'camera.tote_placement=left']],
)
def test_a_mount_named_like_a_scene_field_is_read_in_either_order(pairs: list[str]):
    assert scene_from_pairs(pairs) == {
        'tote_placement': Placement.right,
        'external_cameras': {'tote_placement': Placement.left},
    }


@pytest.mark.parametrize(
    'pairs', [['tote_placement=left', 'tote_placement=right'], ['camera.side=left', 'camera.side=right']]
)
def test_a_scene_key_given_twice_is_refused(pairs: list[str]):
    with pytest.raises(SystemExit, match='given twice'):
        scene_from_pairs(pairs)


def test_the_scene_keys_are_fields_the_plan_declares():
    # A key this reader accepts that the model does not carry would be dropped in silence.
    scene = scene_from_pairs(['tote_placement=left', 'camera_vantage=droid', 'camera.side=right'])
    assert set(scene) <= set(EvalPlan.model_fields)
