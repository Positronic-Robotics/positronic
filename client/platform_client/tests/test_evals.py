"""The eval reference a submission names, and the definitions of the public evals."""

from __future__ import annotations

import pytest
from platform_client.eval_plan import EvalPlan, plan_of_image
from platform_client.evals import (
    MOLMO_CONFIG,
    MOLMO_EPISODE_INDEX_KEY,
    PUBLIC_EVALS,
    EvalDefinition,
    EvalRef,
    ScorerRef,
    public_eval,
)
from platform_client.policy_images import PolicyImage
from pydantic import ValidationError


def test_an_eval_name_is_a_str_carrying_its_type():
    ref = EvalRef('robolab.public_subset')
    assert ref == 'robolab.public_subset'
    assert isinstance(ref, str)


@pytest.mark.parametrize('value', ['', ' ', 'two words', 'trailing '])
def test_a_value_that_could_never_name_an_eval_is_refused_here(value: str):
    with pytest.raises(ValueError):
        EvalRef(value)


def test_a_name_this_client_has_never_heard_of_still_reaches_the_platform():
    # The set lives on the server; a client that curated its own copy would refuse a newly offered
    # eval until someone remembered to release it.
    plan = plan_of_image(PolicyImage('org/policy:v1'), EvalRef('an.eval.shipped.this.morning'))
    assert plan.eval == 'an.eval.shipped.this.morning'


def test_the_boundary_refuses_an_empty_eval():
    with pytest.raises(ValidationError):
        EvalPlan.model_validate({'eval': '', 'endpoints': [{'name': 'policy', 'kind': 'image', 'image': 'org/p:v1'}]})


@pytest.mark.parametrize('value', ['', ' ', 'two words'])
def test_a_value_that_could_never_name_a_scorer_is_refused_here(value: str):
    with pytest.raises(ValueError):
        ScorerRef(value)


BOARD = EvalRef('molmo.franka_pick_mini')
SMOKE = EvalRef('molmo.franka_pick_mini_smoke')


def test_the_smoke_eval_runs_the_first_five_benchmark_episodes_once_each():
    definition = public_eval(SMOKE)

    assert definition.config == '.sim.molmo.benchmarks'
    assert (definition.args['episodes'], definition.args['trial_count']) == ([0, 1, 2, 3, 4], 1)
    assert definition.time_limit_s == 7200


def test_the_board_eval_runs_the_first_twenty_benchmark_episodes_once_each():
    definition = public_eval(BOARD)

    assert definition.config == '.sim.molmo.benchmarks'
    assert (definition.args['episodes'], definition.args['trial_count']) == (list(range(20)), 1)
    assert definition.time_limit_s == 7200


@pytest.mark.parametrize('name', [name for name in PUBLIC_EVALS if public_eval(name).config == MOLMO_CONFIG])
def test_a_molmo_definition_runs_each_episode_its_arguments_name_trial_count_times(name: EvalRef):
    definition = public_eval(name)
    episodes, trial_count = definition.args['episodes'], definition.args['trial_count']
    assert isinstance(episodes, list) and isinstance(trial_count, int)

    (task,) = definition.tasks
    assert task.trials == [{MOLMO_EPISODE_INDEX_KEY: i} for i in episodes for _ in range(trial_count)]


def test_a_name_the_public_code_does_not_define_is_refused_with_the_public_names():
    with pytest.raises(LookupError) as refused:
        public_eval(EvalRef('molmo.held_out'))

    assert "'molmo.held_out' is not a public eval" in str(refused.value)
    assert all(name in str(refused.value) for name in PUBLIC_EVALS)


def test_a_public_definition_is_handed_out_as_a_copy():
    before = public_eval(SMOKE)
    handed = public_eval(SMOKE)
    episodes = handed.args['episodes']
    assert isinstance(episodes, list)
    episodes.append(999)
    handed.tasks[0].trials.clear()

    assert public_eval(SMOKE) == before


@pytest.mark.parametrize(
    'change',
    [
        {'unknown_field': 1},
        {'config': 'sim.molmo.benchmarks'},
        {'config': '@positronic.cfg.eval.sim.molmo.benchmarks'},
        {'positronic_revision': 'a9e13e8a'},
        {'time_limit_s': 0},
        {'tasks': []},
        {'scorer': ''},
    ],
)
def test_a_definition_refuses_a_field_no_run_can_use(change: dict):
    stated = public_eval(SMOKE).model_dump()

    with pytest.raises(ValidationError):
        EvalDefinition.model_validate({**stated, **change})
