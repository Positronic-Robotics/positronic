"""The plan's own rules: the count cascade, the bare-label shapes, and what each level may state."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from platform_client.enums import EndpointKind, Placement, Wire
from platform_client.eval_plan import (
    _ENDPOINT_MAY_STATE,
    ADDRESS_OF_WIRE,
    Endpoint,
    EvalPlan,
    HostPortAddress,
    RoboarenaAddress,
    TaskNode,
    UnixSocketAddress,
    plan_of_image,
)
from platform_client.evals import EvalRef
from platform_client.policy_images import PolicyImage
from platform_client.slug import slug_of
from platform_client.tasks import TaskRef
from positronic_wire import registry
from pydantic import ValidationError

SPOONS = 'eight-spoons-into-grey-tote'
MUG = 'marker-in-mug'
SESSION = '/api/v1/session'


def remote(name: str, **over) -> dict:
    """A remote entry on the websocket wire, at ``name``'s own host."""
    address = {'host': f'{name}.example', 'port': 8000, 'path': SESSION}
    return {'name': name, 'wire': 'websocket', 'address': address, **over}


BASELINE = remote('baseline')
CANDIDATE = remote('candidate')


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


def test_a_served_endpoint_names_its_spec_and_its_wire_and_no_address():
    served = Endpoint.model_validate({'name': 'pi05', 'kind': 'served', 'spec': 'pi05-droid', 'wire': 'grpc'})
    assert served.names_a_locator and served.provider is None and served.wire is Wire.grpc
    assert Endpoint.model_validate({'name': 'pi05', 'kind': 'served'}).names_a_locator is False
    with pytest.raises(ValidationError, match='the platform records the address it serves at'):
        Endpoint.model_validate({**remote('pi05'), 'kind': 'served', 'provider': 'cohost', 'spec': 's'})


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


def test_the_fields_an_endpoint_may_state_are_fields_it_declares():
    # A name here the model does not carry would let the per-task field of that name through.
    assert _ENDPOINT_MAY_STATE <= set(Endpoint.model_fields)


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
                'endpoints': ['bare', remote('two', episodes_per_endpoint=2), remote('five')],
            },
        ],
        endpoints=[remote('own', episodes_per_endpoint=3), remote('bare')],
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
        Endpoint(name='baseline', wire=Wire.websocket, image=PolicyImage('org/policy:v1'))


def test_an_image_endpoint_names_the_image_and_its_wire_and_nothing_else():
    entry = Endpoint(
        name='policy', kind=EndpointKind.image, wire=Wire.websocket, image=PolicyImage('org/policy@sha256:abc')
    )
    assert entry.names_a_locator and entry.address is None
    with pytest.raises(ValidationError, match='the platform runs the image'):
        Endpoint.model_validate({**BASELINE, 'kind': 'image', 'image': 'org/p:v1'})
    with pytest.raises(ValidationError, match='only an image endpoint carries'):
        Endpoint(name='policy', kind=EndpointKind.served, spec='pi05', wire=Wire.grpc, image=PolicyImage('org/p:v1'))


def test_a_plan_of_an_image_names_the_eval_and_states_no_task():
    # The catalogue expands the name into the tasks and the count each takes, so the plan states
    # neither.
    plan = plan_of_image(
        PolicyImage('org/policy@sha256:abc'), EvalRef('robolab.public_subset'), Wire.websocket, alias='demo'
    )
    assert plan.names_an_eval and not plan.tasks and plan.episodes_per_endpoint is None
    assert [(entry.image, entry.wire) for entry in plan.endpoints] == [('org/policy@sha256:abc', Wire.websocket)]
    assert plan.alias == 'demo'
    assert EvalPlan.model_validate(plan.model_dump(mode='json')) == plan


def test_a_plan_takes_its_tasks_from_itself_or_from_an_eval_and_not_from_both():
    with pytest.raises(ValidationError, match='it takes its tasks from one'):
        a_plan(eval='robolab.public_subset')
    with pytest.raises(ValidationError, match='names at least one task, or the eval'):
        EvalPlan.model_validate({'endpoints': [BASELINE], 'episodes_per_endpoint': 1})


@pytest.mark.parametrize(
    ('wire', 'address', 'dialled'),
    [
        ('websocket_tls', {'host': 'h.example', 'port': 443, 'path': SESSION, 'query': 'mode=native'}, HostPortAddress),
        ('grpc', {'host': '::1', 'port': 50051, 'path': f'{SESSION}/org/model'}, HostPortAddress),
        ('websocket_unix', {'uds': '/run/policy.sock', 'path': SESSION}, UnixSocketAddress),
        ('roboarena', {'host': 'h.example', 'port': 8000}, RoboarenaAddress),
    ],
)
def test_a_remote_endpoint_names_its_wire_then_the_fields_that_wire_dials(wire: str, address: dict, dialled: type):
    entry = Endpoint.model_validate({'name': 'baseline', 'wire': wire, 'address': address})
    assert type(entry.address) is dialled
    assert Endpoint.model_validate(entry.model_dump(mode='json')) == entry


def test_a_url_is_refused_with_the_fields_to_write_instead():
    with pytest.raises(
        ValidationError, match=r'names a url.*websocket: host, port, path, query.*roboarena: host, port'
    ):
        Endpoint.model_validate({'name': 'baseline', 'url': 'wss://baseline.example/ws'})
    with pytest.raises(ValidationError, match='carries a scheme or a path'):
        Endpoint.model_validate(remote('baseline') | {'address': {**BASELINE['address'], 'host': 'wss://h.example'}})
    with pytest.raises(ValidationError, match='is no session route'):
        Endpoint.model_validate(remote('baseline') | {'address': {**BASELINE['address'], 'path': 'wss://h/ws'}})


def test_an_address_carries_the_fields_its_wire_dials_and_no_other():
    with pytest.raises(
        ValidationError, match='roboarena wire, which dials host, port; the address carries host, port, path'
    ):
        Endpoint.model_validate(remote('baseline', wire='roboarena'))
    with pytest.raises(
        ValidationError, match='grpc wire, which dials host, port, path, query; the address carries host, port'
    ):
        Endpoint.model_validate({'name': 'baseline', 'wire': 'grpc', 'address': {'host': 'h', 'port': 1}})
    with pytest.raises(ValidationError, match='which dials host, port, path, query; the address carries uds'):
        Endpoint(name='baseline', wire=Wire.grpc, address=UnixSocketAddress(uds=Path('/run/p.sock'), path=SESSION))


def test_an_endpoint_that_runs_a_policy_names_its_wire():
    with pytest.raises(ValidationError, match='names no wire; name one of websocket, websocket_tls'):
        Endpoint.model_validate({'name': 'baseline', 'address': BASELINE['address']})
    with pytest.raises(ValidationError, match='names no wire'):
        Endpoint.model_validate({'name': 'pi05', 'kind': 'served', 'spec': 'pi05-droid'})
    with pytest.raises(ValidationError, match='names no wire'):
        Endpoint.model_validate(IMAGE_ENDPOINT[0] | {'wire': None})
    with pytest.raises(ValidationError, match='names a wire and nothing that runs on it'):
        Endpoint.model_validate({'name': 'baseline', 'wire': 'grpc'})


def test_a_socket_path_is_absolute():
    with pytest.raises(ValidationError, match='relative socket path'):
        Endpoint.model_validate({
            'name': 'baseline',
            'wire': 'websocket_unix',
            'address': {'uds': 'policy.sock', 'path': SESSION},
        })


def test_each_wire_dials_the_fields_its_registry_member_declares():
    # A field the platform client carries and the wire does not is one a caller writes and no dial reads.
    declared = {}
    for name, member in registry.CLIENT_WIRES.items():
        assert dataclasses.is_dataclass(member.ADDRESS)
        declared[name] = [field.name for field in dataclasses.fields(member.ADDRESS)]
    carried = {slug_of(wire): list(address.model_fields) for wire, address in ADDRESS_OF_WIRE.items()}
    assert carried == declared


@pytest.mark.parametrize('entry', [{'name': 'bare'}, {'name': 'bare', 'kind': 'served'}])
def test_a_plan_endpoint_states_where_its_policy_comes_from(entry: dict):
    with pytest.raises(ValidationError, match='states where its policy comes from'):
        a_plan(endpoints=[entry])


def test_an_endpoint_says_whether_it_names_a_locator():
    assert Endpoint(name='baseline').names_a_locator is False
    assert Endpoint.model_validate(BASELINE).names_a_locator is True


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
    a_plan(endpoints=[], tasks=[{'task_id': SPOONS, 'endpoints': [remote('elsewhere')]}])


def test_every_task_runs_on_at_least_one_endpoint():
    with pytest.raises(ValidationError, match='runs on no endpoint'):
        a_plan(endpoints=[])
    with pytest.raises(ValidationError):
        TaskNode.model_validate({'task_id': SPOONS, 'endpoints': []})


IMAGE_ENDPOINT = [{'name': 'policy', 'kind': 'image', 'image': 'org/p:v1', 'wire': 'websocket'}]


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
