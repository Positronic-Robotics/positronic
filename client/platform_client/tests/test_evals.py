"""The eval reference a submission names."""

from __future__ import annotations

import pytest
from platform_client.eval_plan import EvalPlan, plan_of_image
from platform_client.evals import EvalRef
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
