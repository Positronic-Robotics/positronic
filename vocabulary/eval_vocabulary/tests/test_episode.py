import json

from eval_vocabulary.episode import EndedBy, PolicyEnd

# The values episodes on disk carry, pinned against the literal rather than against the enum under test.
WIRE_VALUES = ('complete', 'give_up')
ENDED_BY_WIRE_VALUES = ('operator', 'policy', 'station')


def test_the_policy_end_values_are_what_recordings_carry():
    assert tuple(end.value for end in PolicyEnd) == WIRE_VALUES


def test_the_ended_by_values_are_what_recordings_carry():
    assert tuple(ender.value for ender in EndedBy) == ENDED_BY_WIRE_VALUES
    # A static is written as JSON, and a member is written as its bare value.
    assert json.dumps(list(EndedBy)) == json.dumps(list(ENDED_BY_WIRE_VALUES))
