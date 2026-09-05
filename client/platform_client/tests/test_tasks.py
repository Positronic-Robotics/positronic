"""The task reference a rollout request names."""

from __future__ import annotations

import pytest
from platform_client.requests import TaskAsk
from platform_client.tasks import TaskRef
from pydantic import ValidationError


def test_a_task_id_is_a_str_carrying_its_type():
    ref = TaskRef('eight-spoons-into-grey-tote')
    assert ref == 'eight-spoons-into-grey-tote'
    assert isinstance(ref, str)


@pytest.mark.parametrize('value', ['', 'Eight-Spoons', 'two words', 'a--b', '-lead', 'trail-', 'a-b-c-d-e-f'])
def test_a_value_that_could_never_be_a_catalogue_key_is_refused_here(value: str):
    with pytest.raises(ValueError):
        TaskRef(value)


def test_an_id_this_client_has_never_heard_of_still_reaches_the_platform():
    # The catalogue lives on the server; a client that curated its own copy would refuse a task
    # added this morning until someone remembered to release it.
    assert TaskAsk.model_validate({'task_id': 'a-task-added-this-morning'}).task_id == 'a-task-added-this-morning'


def test_the_boundary_refuses_a_malformed_id():
    with pytest.raises(ValidationError):
        TaskAsk.model_validate({'task_id': 'Eight Spoons'})
