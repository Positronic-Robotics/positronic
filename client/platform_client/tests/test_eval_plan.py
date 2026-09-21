"""The plan's own rules: the count cascade, the bare-label shapes, and what each level may state."""

from __future__ import annotations

import json

import pytest
from platform_client.enums import EndpointKind, Placement
from platform_client.eval_plan import (
    _ENDPOINT_OVERRIDES,
    _PER_TASK_ONLY,
    SENDING,
    Cascade,
    Endpoint,
    EvalPlan,
    RegistryCredential,
    TaskNode,
    plan_of_image,
)
from platform_client.evals import EvalRef
from platform_client.policy_images import PolicyImage
from platform_client.tasks import TaskRef
from pydantic import SecretStr, ValidationError

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


def test_a_plan_names_a_task():
    with pytest.raises(ValidationError, match='names at least one task'):
        a_plan(tasks=[])


def test_a_plan_states_a_count():
    with pytest.raises(ValidationError, match='states episodes_per_endpoint'):
        EvalPlan.model_validate({'tasks': [SPOONS], 'endpoints': [BASELINE]})


def test_a_task_label_names_a_plan_endpoint():
    with pytest.raises(ValidationError, match='names phantom'):
        a_plan(tasks=[{'task_id': SPOONS, 'endpoints': ['phantom']}])


def test_an_endpoint_overrides_only_its_count():
    with pytest.raises(ValidationError, match='per-task properties'):
        a_plan(endpoints=[{**BASELINE, 'cap_per_episode_sec': 60}])


def test_a_served_endpoint_names_its_spec_and_no_url():
    served = Endpoint.model_validate({'name': 'pi05', 'kind': 'served', 'spec': 'pi05-droid'})
    assert served.names_a_locator and served.provider is None
    assert Endpoint.model_validate({'name': 'pi05', 'kind': 'served'}).names_a_locator is False
    with pytest.raises(ValidationError, match='names a url'):
        Endpoint.model_validate({
            'name': 'pi05',
            'kind': 'served',
            'provider': 'cohost',
            'spec': 's',
            'url': 'wss://x/ws',
        })


def test_an_endpoint_states_only_a_locator_and_its_own_count():
    # The check names what an endpoint may state, so a field added to `Cascade` is refused here
    # rather than accepted in silence, and the refusal names every field it found.
    with pytest.raises(ValidationError, match='which are per-task properties'):
        Endpoint.model_validate({**BASELINE, 'policy_preset': 'p'})
    with pytest.raises(ValidationError, match='cap_per_episode_sec, tote_placement'):
        Endpoint.model_validate({**BASELINE, 'tote_placement': 'left', 'cap_per_episode_sec': 30})
    allowed = Endpoint.model_validate({**BASELINE, 'episodes_per_endpoint': 2})
    assert allowed.episodes_per_endpoint == 2


def test_an_endpoint_reads_back_from_its_own_dump():
    # A dump names every field, so the check reads what each one carries: a plan the client sends
    # is one the platform validates from that JSON.
    entry = Endpoint.model_validate({**BASELINE, 'episodes_per_endpoint': 2})
    assert Endpoint.model_validate(entry.model_dump(mode='json')) == entry


A_PER_TASK_VALUE = {
    'cap_per_episode_sec': 60,
    'policy_preset': 'other',
    'tote_placement': 'left',
    'camera_vantage': 'phail',
    'external_cameras': {'side': 'left'},
    'clutter': {'count_min': 1, 'count_max': 2},
}


def test_an_endpoint_refuses_every_per_task_property():
    """Every property `Cascade` carries but the count, driven over the model rather than a list.

    A set that repeats its names passes a membership check and goes on passing once the model
    carries a name nothing added to it. The coverage assert is what a rename or an addition fails
    on, and it covers the one name the refused set still spells as a string.
    """
    assert _ENDPOINT_OVERRIDES in Cascade.model_fields
    assert set(A_PER_TASK_VALUE) == _PER_TASK_ONLY
    # An endpoint's own fields do not cascade, so a field added to `Endpoint` — `image_credential`
    # among them — is stated on one with nothing to add here.
    assert not _PER_TASK_ONLY & (set(Endpoint.model_fields) - set(Cascade.model_fields))
    for name, value in A_PER_TASK_VALUE.items():
        with pytest.raises(ValidationError, match='per-task properties'):
            Endpoint.model_validate({**BASELINE, name: value})


def test_an_endpoint_states_the_one_property_it_overrides():
    """The boundary of the rule above: the count is the cascading property an endpoint may state."""
    assert Endpoint.model_validate({**BASELINE, _ENDPOINT_OVERRIDES: 2}).episodes_per_endpoint == 2


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
        endpoints=[
            {'name': 'own', 'url': 'wss://own.example/ws', 'episodes_per_endpoint': 3},
            {'name': 'bare', 'url': 'wss://bare.example/ws'},
        ],
    )
    first, second = plan.tasks
    assert [plan.episodes_on(first, entry) for entry in plan.task_endpoints(first)] == [3, 10]
    assert [plan.episodes_on(second, entry) for entry in plan.task_endpoints(second)] == [5, 2, 5]
    assert plan.resolved_episodes_total == 25
    # The count is serialized under its own field name and round-trips.
    sent = plan.endpoints[0].model_dump(mode='json')
    assert sent['episodes_per_endpoint'] == 3 and Endpoint.model_validate(sent) == plan.endpoints[0]


def test_a_remote_endpoint_names_no_bring_up():
    with pytest.raises(ValidationError, match='only a served or an image endpoint carries'):
        Endpoint(name='baseline', provider='droid_cohost')
    with pytest.raises(ValidationError, match='only a served or an image endpoint carries'):
        Endpoint(name='baseline', image=PolicyImage('org/policy:v1'))


def test_an_image_endpoint_names_the_image_and_nothing_else():
    entry = Endpoint(name='policy', kind=EndpointKind.image, image=PolicyImage('org/policy@sha256:abc'))
    assert entry.names_a_locator and entry.url is None
    with pytest.raises(ValidationError, match='the platform runs the image'):
        Endpoint(name='policy', kind=EndpointKind.image, image=PolicyImage('org/p:v1'), url='wss://h/ws')
    with pytest.raises(ValidationError, match='only an image endpoint carries'):
        Endpoint(name='policy', kind=EndpointKind.served, spec='pi05', image=PolicyImage('org/p:v1'))


def test_a_plan_of_an_image_names_the_eval_and_states_no_task():
    # The catalogue expands the name into the tasks and the count each takes, so the plan states
    # neither.
    plan = plan_of_image(PolicyImage('org/policy@sha256:abc'), EvalRef('robolab.public_subset'), alias='demo')
    assert plan.names_an_eval and not plan.tasks and plan.episodes_per_endpoint is None
    assert [entry.image for entry in plan.endpoints] == ['org/policy@sha256:abc']
    assert plan.alias == 'demo'
    assert EvalPlan.model_validate(plan.model_dump(mode='json')) == plan


def test_a_plan_takes_its_tasks_from_itself_or_from_an_eval_and_not_from_both():
    with pytest.raises(ValidationError, match='it takes its tasks from one'):
        a_plan(eval='robolab.public_subset')
    with pytest.raises(ValidationError, match='names at least one task, or the eval'):
        EvalPlan.model_validate({'endpoints': [BASELINE], 'episodes_per_endpoint': 1})


def test_an_endpoint_url_names_a_host():
    """An address with no host reaches nothing. Without this check it counts as a locator, and the
    platform refuses the plan only after it is filed."""
    with pytest.raises(ValidationError, match='no host'):
        Endpoint(name='baseline', url='/ws')
    with pytest.raises(ValidationError, match='no host'):
        Endpoint(name='baseline', url='baseline.example/ws')


@pytest.mark.parametrize(
    'url', ['wss://baseline.example/ws', 'https://baseline.example/ws', 'http://localhost:8080/ws']
)
def test_an_absolute_endpoint_url_is_left_alone(url: str):
    """The platform judges the scheme, and it dials wss:// and https:// alike, so the client refuses
    only an address with no host."""
    assert Endpoint(name='baseline', url=url).url == url


@pytest.mark.parametrize('entry', [{'name': 'bare'}, {'name': 'bare', 'kind': 'served'}])
def test_a_plan_endpoint_states_where_its_policy_comes_from(entry: dict):
    with pytest.raises(ValidationError, match='states where its policy comes from'):
        a_plan(endpoints=[entry])


def test_an_endpoint_says_whether_it_names_a_locator():
    assert Endpoint(name='baseline').names_a_locator is False
    assert Endpoint(name='baseline', url='wss://x/ws').names_a_locator is True


def test_a_plan_names_each_endpoint_once():
    with pytest.raises(ValidationError, match='more than once'):
        a_plan(endpoints=[BASELINE, BASELINE])
    with pytest.raises(ValidationError, match='more than once'):
        TaskNode.model_validate({'task_id': SPOONS, 'endpoints': [{'name': 'e'}, {'name': 'e'}]})


def test_a_plan_may_name_one_task_twice():
    """Two nodes of one task, each with its own scene, is a plan the platform runs. The list keeps
    them apart and in order; the id cannot."""
    plan = a_plan(tasks=[SPOONS, {'task_id': SPOONS, 'episodes_per_endpoint': 2}])

    assert [task.task_id for task in plan.tasks] == [SPOONS, SPOONS]
    assert [task.episodes_per_endpoint for task in plan.tasks] == [None, 2]
    assert plan.resolved_episodes_total == 24


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


IMAGE_ENDPOINT = [{'name': 'policy', 'kind': 'image', 'image': 'org/p:v1'}]


def test_a_plan_naming_an_eval_states_the_policy_that_runs_it():
    # The catalogue supplies a named eval's tasks, so `tasks` is empty and every check that iterates
    # them reads nothing. Without this the plan files with no policy at all.
    with pytest.raises(ValidationError, match='defines no endpoint'):
        EvalPlan.model_validate({'eval': 'robolab.public_subset'})
    # One plan-level endpoint is enough, which is the shape `plan_of_image` builds.
    EvalPlan.model_validate({'eval': 'robolab.public_subset', 'endpoints': IMAGE_ENDPOINT})


def test_the_plan_own_cap_is_checked_against_the_ceiling_with_no_task_to_carry_it():
    # The same empty-`tasks` seam: the cap a catalogue task would inherit is stated on the plan, so
    # it is checked there rather than through a task the plan does not have.
    with pytest.raises(ValidationError, match='over its own ceiling'):
        EvalPlan.model_validate({
            'eval': 'robolab.public_subset',
            'endpoints': IMAGE_ENDPOINT,
            'cap_per_episode_sec': 120,
            'max_cap_per_episode_sec': 100,
        })
    # A cap under the ceiling passes, and a task's own override is still checked.
    EvalPlan.model_validate({
        'eval': 'robolab.public_subset',
        'endpoints': IMAGE_ENDPOINT,
        'cap_per_episode_sec': 90,
        'max_cap_per_episode_sec': 100,
    })


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


# A value no assertion below may find in a rendering of a plan.
A_PASSWORD = 'the-registry-password'
A_CREDENTIAL = {'username': 'a-reader', 'password': A_PASSWORD}


def an_image_endpoint(**over) -> dict:
    return {'name': 'policy', 'kind': 'image', 'image': 'org/policy:v1', **over}


def test_an_image_endpoint_carries_a_credential_for_a_private_registry():
    endpoint = Endpoint.model_validate(an_image_endpoint(image_credential=A_CREDENTIAL))
    assert endpoint.image_credential is not None
    assert endpoint.image_credential.username == 'a-reader'
    assert endpoint.image_credential.password.get_secret_value() == A_PASSWORD


def test_an_entry_that_names_no_image_may_state_no_credential():
    """The boundary of the rule above: a credential opens the image its own entry names."""
    with pytest.raises(ValidationError, match='names no image'):
        Endpoint.model_validate({'name': 'remote', 'url': 'wss://host/ws', 'image_credential': A_CREDENTIAL})
    with pytest.raises(ValidationError, match='names no image'):
        Endpoint.model_validate({'name': 'policy', 'image_credential': A_CREDENTIAL})


def test_a_per_task_entry_states_a_credential_for_the_image_it_names():
    """An endpoint's own field, so the count rule does not read it as a per-task property."""
    assert 'image_credential' not in _PER_TASK_ONLY
    plan = EvalPlan.model_validate({
        'tasks': [{'task_id': SPOONS, 'endpoints': [an_image_endpoint(image_credential=A_CREDENTIAL)]}],
        'episodes_per_endpoint': 4,
    })
    entry = plan.tasks[0].endpoints[0] if plan.tasks[0].endpoints else None
    assert entry is not None and entry.image_credential is not None


def test_no_rendering_of_a_plan_carries_the_password():
    """Every way a plan reaches a log, an error or a store renders the password as a mask."""
    plan = EvalPlan.model_validate({
        'eval': 'robolab.public_subset',
        'endpoints': [an_image_endpoint(image_credential=A_CREDENTIAL)],
    })
    assert A_PASSWORD not in repr(plan)
    assert A_PASSWORD not in str(plan)
    assert A_PASSWORD not in plan.model_dump_json()
    assert A_PASSWORD not in str(plan.model_dump())
    assert A_PASSWORD not in str(plan.model_dump(mode='json'))


def test_a_credential_survives_the_model_it_is_read_into():
    """The mask is a rendering, not the value: the platform still reads what opens the registry."""
    plan = EvalPlan.model_validate({
        'eval': 'robolab.public_subset',
        'endpoints': [an_image_endpoint(image_credential=A_CREDENTIAL)],
    })
    credential = plan.endpoints[0].image_credential
    assert credential is not None
    assert credential.password.get_secret_value() == A_PASSWORD


def test_an_empty_half_of_a_credential_is_refused():
    with pytest.raises(ValidationError):
        Endpoint.model_validate(an_image_endpoint(image_credential={'username': '', 'password': A_PASSWORD}))
    with pytest.raises(ValidationError):
        Endpoint.model_validate(an_image_endpoint(image_credential={'username': 'a-reader', 'password': ''}))


def test_plan_of_image_carries_the_credential_onto_its_one_endpoint():
    plan = plan_of_image(
        PolicyImage('org/policy:v1'),
        EvalRef('robolab.public_subset'),
        credential=RegistryCredential(username='a-reader', password=SecretStr(A_PASSWORD)),
    )
    credential = plan.endpoints[0].image_credential
    assert credential is not None
    assert credential.password.get_secret_value() == A_PASSWORD


def test_a_refused_endpoint_reports_no_password():
    """A model-level validator is handed the whole input dict, before any field is coerced.

    So the password the error would echo is the plaintext the caller typed, which `SecretStr`
    reaches nowhere: a command that prints the exception writes it to the terminal, and a `logging`
    call that takes the exception writes it to the log.
    """
    with pytest.raises(ValidationError) as caught:
        Endpoint.model_validate(an_image_endpoint(url='not-absolute', image_credential=A_CREDENTIAL))
    error = caught.value
    assert A_PASSWORD not in str(error)
    assert A_PASSWORD not in repr(error)
    # Hiding the input costs the echoed value alone; the error still says what was wrong.
    assert 'has no host' in str(error)


def test_a_refused_plan_reports_no_password():
    """The shape a caller validates: the endpoint is nested, and the error names its place."""
    with pytest.raises(ValidationError) as caught:
        EvalPlan.model_validate({
            'eval': 'robolab.public_subset',
            'endpoints': [an_image_endpoint(url='not-absolute', image_credential=A_CREDENTIAL)],
        })
    error = caught.value
    assert A_PASSWORD not in str(error)
    assert A_PASSWORD not in repr(error)
    assert 'endpoints.0' in str(error)


def test_a_caller_that_asks_for_the_input_is_given_it():
    """The boundary of the rule above: `errors()` and `json()` carry the input on request.

    Pydantic takes `include_input` per call rather than from the model, so this is the caller's to
    drop, and the config reaches only what a model renders on its own.
    """
    with pytest.raises(ValidationError) as caught:
        Endpoint.model_validate(an_image_endpoint(url='not-absolute', image_credential=A_CREDENTIAL))
    error = caught.value
    assert A_PASSWORD in repr(error.errors())
    assert A_PASSWORD not in repr(error.errors(include_input=False))


def test_a_plan_serialised_by_hand_refuses_to_write_the_password():
    """`model_dump` yields the `SecretStr`, which `json` will not encode.

    Masking covers the renderings a model controls. This covers the one it does not: a caller that
    takes the dump apart itself raises here and writes nothing, so no store, log or payload built
    that way can carry the value.
    """
    plan = EvalPlan.model_validate({
        'eval': 'robolab.public_subset',
        'endpoints': [an_image_endpoint(image_credential=A_CREDENTIAL)],
    })
    with pytest.raises(TypeError):
        json.dumps(plan.model_dump())


def test_only_the_send_path_serialises_the_password_as_itself():
    """The value travels in the request that carries it, and in no other rendering.

    A mask everywhere else keeps the password out of a log and a store. A mask HERE would hand the
    platform a credential that opens nothing.
    """
    plan = EvalPlan.model_validate({
        'eval': 'robolab.public_subset',
        'endpoints': [an_image_endpoint(image_credential=A_CREDENTIAL)],
    })
    sent = plan.model_dump(mode='json', context={SENDING: True})
    assert sent['endpoints'][0]['image_credential']['password'] == A_PASSWORD
    assert A_PASSWORD not in json.dumps(plan.model_dump(mode='json'))
    assert A_PASSWORD not in plan.model_dump_json()
