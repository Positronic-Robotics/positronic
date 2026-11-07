from positronic.eval import keys


def test_a_caller_of_the_keys_still_reads_the_ended_by_names_it_was_written_against():
    assert keys.ENDED_BY_OPERATOR is keys.EndedBy.OPERATOR
    assert keys.ENDED_BY_POLICY is keys.EndedBy.POLICY
