from eval_vocabulary.episode import EditOp, PolicyEnd

# The values episodes on disk carry, pinned against the literal rather than against the enum under test.
WIRE_VALUES = ('complete', 'give_up')


def test_the_policy_end_values_are_what_recordings_carry():
    assert tuple(end.value for end in PolicyEnd) == WIRE_VALUES


def test_the_edit_ops_are_what_edit_logs_carry():
    assert tuple(op.value for op in EditOp) == ('set_static', 'drop', 'undrop')
