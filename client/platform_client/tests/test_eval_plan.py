"""The plan's own rules: the count cascade, the bare-label shapes, and what each level may state."""

from __future__ import annotations

import pytest
from platform_client.enums import EndpointKind, Placement
from platform_client.eval_plan import Endpoint, EvalPlan, TaskNode
from platform_client.tasks import TaskRef
from pydantic import ValidationError

SPOONS = 'eight-spoons-into-grey-tote'
MUG = 'marker-in-mug'
BASELINE = {'name': 'baseline', 'url': 'wss://baseline.example/ws'}
CANDIDATE = {'name': 'candidate', 'url': 'wss://candidate.example/ws'}


def a_plan(**over) -> EvalPlan:
    fields = {'tasks': [SPOONS], 'endpoints': [BASELINE, CANDIDATE], 'episodes_per_endpoint': 10}
    return EvalPlan.model_validate({**fields, **over})


def test_the_plan_count_reaches_every_leaf():
    plan = a_plan(tasks=[SPOONS, MUG])
    assert plan.resolved_episodes_total == 40


def test_a_task_overrides_the_plan_and_an_endpoint_overrides_the_task():
    plan = a_plan(tasks=[SPOONS, {'task_id': MUG, 'episodes_per_endpoint': 2, 'endpoints': ['candidate']}])
    mug = plan.tasks[1]
    assert mug.endpoints is not None
    assert plan.task_endpoints(mug) == [Endpoint(name='candidate')]
    assert plan.episodes_on(mug, mug.endpoints[0]) == 2
    assert plan.resolved_episodes_total == 22


def test_a_plan_endpoint_count_beats_the_task_count_for_a_bare_label():
    """The definition the label names states the count for that endpoint on every task it runs."""
    plan = a_plan(
        tasks=[{'task_id': MUG, 'episodes_per_endpoint': 2, 'endpoints': ['candidate']}],
        endpoints=[BASELINE, {**CANDIDATE, 'episodes_per_endpoint': 12}],
    )
    mug = plan.tasks[0]
    assert mug.endpoints is not None
    assert plan.episodes_on(mug, mug.endpoints[0]) == 12


def test_a_stated_checksum_must_match_the_leaves():
    assert a_plan(episodes_total=20).episodes_total == 20
    with pytest.raises(ValidationError, match='leaves sum to 20'):
        a_plan(episodes_total=21)


def test_a_bare_task_id_and_a_bare_endpoint_label_are_the_short_forms():
    plan = a_plan(tasks=[{'task_id': SPOONS, 'endpoints': ['baseline']}])
    spoons = plan.tasks[0]
    assert spoons == TaskNode(task_id=TaskRef(SPOONS), endpoints=[Endpoint(name='baseline')])
    assert spoons.endpoints is not None and spoons.endpoints[0].kind is EndpointKind.remote


def test_a_plan_states_a_count():
    with pytest.raises(ValidationError, match='states episodes_per_endpoint'):
        EvalPlan.model_validate({'tasks': [SPOONS], 'endpoints': [BASELINE]})


def test_a_task_label_names_a_plan_endpoint():
    with pytest.raises(ValidationError, match='names phantom'):
        a_plan(tasks=[{'task_id': SPOONS, 'endpoints': ['phantom']}])


def test_an_endpoint_overrides_only_its_count():
    with pytest.raises(ValidationError, match='per-task properties'):
        a_plan(endpoints=[{**BASELINE, 'cap_per_episode_sec': 60}])


def test_a_served_endpoint_names_its_bring_up_and_no_url():
    served = Endpoint.model_validate({'name': 'pi05', 'kind': 'served', 'provider': 'cohost', 'spec': 'pi05-droid'})
    assert served.names_a_locator
    with pytest.raises(ValidationError, match='names no provider or no spec'):
        Endpoint.model_validate({'name': 'pi05', 'kind': 'served'})
    with pytest.raises(ValidationError, match='names a url'):
        Endpoint.model_validate({
            'name': 'pi05',
            'kind': 'served',
            'provider': 'cohost',
            'spec': 's',
            'url': 'wss://x/ws',
        })


def test_a_scene_is_flat_on_every_level():
    plan = a_plan(
        tote_placement='random',
        camera_vantage='phail',
        external_cameras={'side': 'left'},
        tasks=[{'task_id': SPOONS, 'tote_placement': 'left'}],
    )
    assert plan.tote_placement is Placement.random
    assert plan.tasks[0].tote_placement is Placement.left
    assert plan.external_cameras == {'side': Placement.left}
    sent = plan.model_dump(mode='json')
    assert sent['tote_placement'] == 'random'
    assert sent['camera_vantage'] == 'phail'
    assert sent['external_cameras'] == {'side': 'left'}


def test_a_scene_value_outside_the_closed_set_is_refused():
    with pytest.raises(ValidationError):
        a_plan(tote_placement='middle')


def test_every_cap_sits_under_the_ceiling():
    with pytest.raises(ValidationError, match='over the plan ceiling'):
        a_plan(max_cap_per_episode_sec=100, tasks=[{'task_id': SPOONS, 'cap_per_episode_sec': 120}])
    assert a_plan(max_cap_per_episode_sec=100, cap_per_episode_sec=100).cap_per_episode_sec == 100


def test_an_unknown_field_is_refused():
    with pytest.raises(ValidationError, match='extra'):
        a_plan(scene={'tote_placement': 'left'})


def test_an_endpoint_count_wins_and_one_without_takes_the_nearest_level():
    # The plan runs `own` at 3 and `bare` at its own 10. The task runs `bare` by label at the
    # task's 5, since the definition states none; `two` at its own 2; `five` at the task's 5.
    plan = a_plan(
        tasks=[
            'a',
            {
                'task_id': 'b',
                'episodes_per_endpoint': 5,
                'endpoints': [
                    'bare',
                    {'name': 'two', 'url': 'wss://two.example/ws', 'episodes_per_endpoint': 2},
                    {'name': 'five', 'url': 'wss://five.example/ws'},
                ],
            },
        ],
        endpoints=[{'name': 'own', 'url': 'wss://own.example/ws', 'episodes_per_endpoint': 3}, {'name': 'bare'}],
    )
    first, second = plan.tasks
    assert [plan.episodes_on(first, entry) for entry in plan.task_endpoints(first)] == [3, 10]
    assert [plan.episodes_on(second, entry) for entry in plan.task_endpoints(second)] == [5, 2, 5]
    assert plan.resolved_episodes_total == 25
    # The count leaves on the wire under its own name, and comes back.
    sent = plan.endpoints[0].model_dump(mode='json')
    assert sent['episodes_per_endpoint'] == 3 and Endpoint.model_validate(sent) == plan.endpoints[0]


def test_a_remote_endpoint_names_no_bring_up():
    with pytest.raises(ValidationError, match='only a served endpoint carries'):
        Endpoint(name='baseline', provider='droid_cohost')


def test_an_endpoint_url_names_a_host():
    """An address with no host reaches nothing, and it counts as a locator all the way to the
    platform, which refuses the plan after it is filed."""
    with pytest.raises(ValidationError, match='no host'):
        Endpoint(name='baseline', url='/ws')
    with pytest.raises(ValidationError, match='no host'):
        Endpoint(name='baseline', url='baseline.example/ws')


@pytest.mark.parametrize(
    'url', ['wss://baseline.example/ws', 'https://baseline.example/ws', 'http://localhost:8080/ws']
)
def test_an_absolute_endpoint_url_is_left_alone(url: str):
    """The boundary: the scheme is the platform's to judge — it dials wss:// as readily as https://
    — so this refuses an address with no host and nothing else."""
    assert Endpoint(name='baseline', url=url).url == url


def test_an_endpoint_says_whether_it_names_a_locator():
    assert Endpoint(name='baseline').names_a_locator is False
    assert Endpoint(name='baseline', url='wss://x/ws').names_a_locator is True


def test_a_plan_names_each_task_and_each_endpoint_once():
    with pytest.raises(ValidationError, match='more than once'):
        a_plan(tasks=[SPOONS, SPOONS])
    with pytest.raises(ValidationError, match='more than once'):
        a_plan(endpoints=[BASELINE, BASELINE])
    with pytest.raises(ValidationError, match='more than once'):
        TaskNode.model_validate({'task_id': SPOONS, 'endpoints': [{'name': 'e'}, {'name': 'e'}]})


def test_a_task_endpoint_naming_no_locator_names_one_the_plan_defines():
    with pytest.raises(ValidationError, match='name no endpoint the plan defines'):
        a_plan(tasks=[{'task_id': SPOONS, 'endpoints': ['elsewhere']}])
    # An entry that carries its own address needs no definition.
    a_plan(endpoints=[], tasks=[{'task_id': SPOONS, 'endpoints': [{'name': 'elsewhere', 'url': 'wss://x/ws'}]}])


def test_every_task_runs_on_at_least_one_endpoint():
    with pytest.raises(ValidationError, match='runs on no endpoint'):
        a_plan(endpoints=[])
    with pytest.raises(ValidationError):
        TaskNode.model_validate({'task_id': SPOONS, 'endpoints': []})


def test_a_count_below_one_is_refused_at_every_level():
    with pytest.raises(ValidationError):
        a_plan(episodes_per_endpoint=0)
    with pytest.raises(ValidationError):
        a_plan(tasks=[{'task_id': SPOONS, 'episodes_per_endpoint': 0}])
    with pytest.raises(ValidationError):
        Endpoint(name='e', episodes_per_endpoint=0)


def test_a_clutter_draw_needs_a_range():
    with pytest.raises(ValidationError, match='which is no range'):
        a_plan(clutter={'count_min': 8, 'count_max': 4})


def test_a_malformed_url_is_a_validation_error():
    with pytest.raises(ValidationError, match='is not a URL'):
        Endpoint.model_validate({'name': 'bad', 'url': 'http://host:bad'})
