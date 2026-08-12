"""Slugged: slug in, member out, unknown rejected — the 422 the API boundary owes a bad value."""

from __future__ import annotations

import pytest
from platform_client.enums import ReasonCode, SubmissionStatus
from platform_client.slug import Slugged, members_by_slug, slug_of
from pydantic import BaseModel, ValidationError


class Holder(BaseModel):
    status: Slugged[SubmissionStatus]
    reason_code: Slugged[ReasonCode] | None = None


def test_a_slug_validates_into_the_member():
    assert Holder.model_validate({'status': 'running'}).status is SubmissionStatus.running
    holder = Holder.model_validate({'status': 'pending', 'reason_code': 'policy_oom'})
    assert holder.reason_code is ReasonCode.policy_oom


def test_a_member_passes_through():
    assert Holder(status=SubmissionStatus.finished).status is SubmissionStatus.finished


def test_serialization_carries_the_slug_not_the_int():
    holder = Holder(status=SubmissionStatus.finished, reason_code=ReasonCode.wall_clock_exceeded)
    assert holder.model_dump(mode='json') == {'status': 'finished', 'reason_code': 'wall_clock_exceeded'}
    assert holder.model_dump() == {'status': 'finished', 'reason_code': 'wall_clock_exceeded'}


@pytest.mark.parametrize('value', ['nonsense', 'Running', 'RUNNING', 'run', '', ' running'])
def test_an_unknown_slug_is_rejected(value: str):
    with pytest.raises(ValidationError):
        Holder.model_validate({'status': value})


def test_the_sentinel_is_not_a_wire_value():
    with pytest.raises(ValidationError):
        Holder.model_validate({'status': 'invalid'})
    with pytest.raises(ValidationError):
        Holder(status=SubmissionStatus.INVALID)


def test_the_stored_int_is_not_accepted_as_input():
    # The wire vocabulary is slugs; an int here means a caller bypassed the boundary.
    with pytest.raises(ValidationError):
        Holder.model_validate({'status': 3})


def test_a_holder_round_trips_through_its_json_form():
    holder = Holder(status=SubmissionStatus.errored, reason_code=ReasonCode.image_unpullable)
    assert Holder.model_validate(holder.model_dump(mode='json')) == holder


def test_the_wire_vocabulary_excludes_the_sentinel():
    assert 'invalid' not in members_by_slug(SubmissionStatus)
    assert set(members_by_slug(SubmissionStatus)) == {m.name for m in SubmissionStatus} - {'INVALID'}


def test_slug_of_is_the_lowercased_name():
    assert slug_of(ReasonCode.policy_setup_crash) == 'policy_setup_crash'


def test_the_generated_schema_advertises_the_slugs():
    schema = Holder.model_json_schema()['properties']['status']
    assert set(schema.get('enum', [])) == set(members_by_slug(SubmissionStatus))
