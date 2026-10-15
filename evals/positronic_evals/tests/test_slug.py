"""Slugged: slug in, member out, unknown rejected — the 422 the API boundary owes a bad value."""

from __future__ import annotations

import pytest
from positronic_evals.labels import Outcome, Stage
from positronic_evals.slug import Slugged, members_by_slug, slug_of
from pydantic import BaseModel, ValidationError


class Holder(BaseModel):
    stage: Slugged[Stage]
    outcome: Slugged[Outcome] | None = None


def test_a_slug_validates_into_the_member():
    assert Holder.model_validate({'stage': 'contact'}).stage is Stage.contact
    holder = Holder.model_validate({'stage': 'reaching', 'outcome': 'out_of_time'})
    assert holder.outcome is Outcome.out_of_time


def test_a_member_passes_through():
    assert Holder(stage=Stage.at_target).stage is Stage.at_target


def test_serialization_carries_the_slug_not_the_int():
    holder = Holder(stage=Stage.at_target, outcome=Outcome.out_of_time)
    assert holder.model_dump(mode='json') == {'stage': 'at_target', 'outcome': 'out_of_time'}
    assert holder.model_dump() == {'stage': 'at_target', 'outcome': 'out_of_time'}


@pytest.mark.parametrize('value', ['nonsense', 'Contact', 'CONTACT', 'cont', '', ' contact'])
def test_an_unknown_slug_is_rejected(value: str):
    with pytest.raises(ValidationError):
        Holder.model_validate({'stage': value})


def test_the_sentinel_is_not_a_wire_value():
    with pytest.raises(ValidationError):
        Holder.model_validate({'stage': 'invalid'})
    with pytest.raises(ValidationError):
        Holder(stage=Stage.INVALID)


def test_the_stored_int_is_not_accepted_as_input():
    # The wire vocabulary is slugs; an int here means a caller bypassed the boundary.
    with pytest.raises(ValidationError):
        Holder.model_validate({'stage': 30})


def test_a_holder_round_trips_through_its_json_form():
    holder = Holder(stage=Stage.floating, outcome=Outcome.safety)
    assert Holder.model_validate(holder.model_dump(mode='json')) == holder


def test_the_wire_vocabulary_excludes_the_sentinel():
    assert 'invalid' not in members_by_slug(Stage)
    assert set(members_by_slug(Stage)) == {m.name for m in Stage} - {'INVALID'}


def test_slug_of_is_the_lowercased_name():
    assert slug_of(Outcome.out_of_time) == 'out_of_time'


def test_the_generated_schema_advertises_the_slugs():
    schema = Holder.model_json_schema()['properties']['stage']
    assert set(schema.get('enum', [])) == set(members_by_slug(Stage))
