from eval_vocabulary.episode import PolicyEnd

# The values episodes on disk carry, pinned against the literal rather than against the enum under test.
WIRE_VALUES = ('complete', 'give_up')


def test_the_policy_end_values_are_what_recordings_carry():
    assert tuple(end.value for end in PolicyEnd) == WIRE_VALUES
