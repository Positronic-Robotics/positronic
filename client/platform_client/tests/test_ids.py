"""Id64: the range invariant, the hex wire form, and the pydantic round trip."""

from __future__ import annotations

import json

import pytest
from platform_client.ids import ID_LIMIT, Id64, SubmissionId, UserId
from pydantic import BaseModel, ValidationError


class ServiceId(Id64):
    """An id declared outside this package — what a service with ids of its own does with `Id64`."""

    __slots__ = ()


class Holder(BaseModel):
    user_id: UserId
    other_id: ServiceId | None = None


def test_an_id_is_an_int_carrying_its_subclass():
    uid = UserId(255)
    assert uid == 255
    assert isinstance(uid, int)
    assert type(uid) is UserId


def test_the_wire_form_is_bare_lowercase_hex():
    assert UserId(255).to_str() == 'ff'
    assert str(UserId(255)) == 'ff'
    assert SubmissionId(ID_LIMIT - 1).to_str() == '7fffffffffffffff'
    assert not UserId(255).to_str().startswith(('0x', 'usr_'))


def test_parse_accepts_an_int_or_case_insensitive_hex():
    assert UserId.parse(255) == 255
    assert UserId.parse('ff') == 255
    assert UserId.parse('FF') == 255
    assert type(UserId.parse('ff')) is UserId


def test_parse_round_trips_through_the_wire_form():
    for value in (1, 255, 4096, ID_LIMIT - 1):
        assert SubmissionId.parse(SubmissionId(value).to_str()) == value


def test_repr_is_evaluable():
    uid = UserId(255)
    assert repr(uid) == "UserId.parse('ff')"
    assert eval(repr(uid)) == uid  # noqa: S307 - the repr contract is that it rebuilds the value


@pytest.mark.parametrize('value', [0, -1, ID_LIMIT, ID_LIMIT + 1, 2**64])
def test_out_of_range_is_rejected(value: int):
    with pytest.raises(ValueError):
        UserId(value)
    with pytest.raises(ValueError):
        UserId.parse(value)


def test_a_bool_is_not_an_id():
    with pytest.raises(TypeError):
        UserId(True)


def test_a_hex_string_needs_parse_not_the_constructor():
    with pytest.raises(TypeError):
        UserId('ff')  # pyright: ignore[reportArgumentType] - the refusal under test is of the wrong type


@pytest.mark.parametrize('value', ['', 'zz', '0xff', '+ff', '-ff', 'f_f', 'ff ff', 'usr_ff', '12.5'])
def test_a_non_hex_string_is_rejected(value: str):
    with pytest.raises(ValueError):
        UserId.parse(value)


def test_zero_is_rejected_in_both_forms():
    with pytest.raises(ValueError):
        UserId.parse('0')
    with pytest.raises(ValueError):
        UserId.parse(0)


def test_the_subclasses_are_distinct_types():
    assert {type(UserId(1)), type(SubmissionId(1)), type(ServiceId(1))} == {UserId, SubmissionId, ServiceId}
    assert not isinstance(UserId(1), SubmissionId)
    assert issubclass(UserId, Id64)


def test_pydantic_validates_an_int_or_a_hex_string_into_the_subclass():
    from_hex = Holder.model_validate({'user_id': 'ff'}).user_id
    from_int = Holder.model_validate({'user_id': 255}).user_id
    assert type(from_hex) is type(from_int) is UserId
    assert from_hex == from_int == 255


def test_pydantic_serializes_to_hex_in_both_dump_modes():
    holder = Holder(user_id=UserId(255), other_id=ServiceId(16))
    assert holder.model_dump(mode='json') == {'user_id': 'ff', 'other_id': '10'}
    assert holder.model_dump() == {'user_id': 'ff', 'other_id': '10'}


def test_the_json_payload_never_carries_a_number():
    payload = json.loads(Holder(user_id=UserId(ID_LIMIT - 1)).model_dump_json())
    assert payload['user_id'] == '7fffffffffffffff'
    assert isinstance(payload['user_id'], str)


def test_json_input_must_be_the_hex_string():
    assert Holder.model_validate_json('{"user_id": "ff"}').user_id == 255
    # A full int64 does not survive a JavaScript JSON parser, so a number is a server bug, not input.
    with pytest.raises(ValidationError):
        Holder.model_validate_json('{"user_id": 255}')


def test_pydantic_rejects_an_out_of_range_or_unparseable_id():
    with pytest.raises(ValidationError):
        Holder.model_validate({'user_id': 0})
    with pytest.raises(ValidationError):
        Holder.model_validate({'user_id': ID_LIMIT})
    with pytest.raises(ValidationError):
        Holder.model_validate({'user_id': 'nope'})


def test_a_model_round_trips_through_its_json_form():
    holder = Holder(user_id=UserId(1234), other_id=ServiceId(4321))
    assert Holder.model_validate(holder.model_dump(mode='json')) == holder
