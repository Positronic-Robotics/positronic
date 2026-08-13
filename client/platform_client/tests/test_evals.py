"""The eval reference a submission names."""

from __future__ import annotations

import pytest
from platform_client.evals import EvalRef
from platform_client.requests import SubmissionCreateRequest
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
    payload = {'policy_image': 'org/policy:v1', 'eval': 'an.eval.shipped.this.morning'}
    assert SubmissionCreateRequest.model_validate(payload).eval == 'an.eval.shipped.this.morning'


def test_the_boundary_refuses_an_empty_eval():
    with pytest.raises(ValidationError):
        SubmissionCreateRequest.model_validate({'policy_image': 'org/policy:v1', 'eval': ''})
