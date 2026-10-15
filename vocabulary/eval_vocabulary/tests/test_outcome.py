from eval_vocabulary.outcome import ABSENT, VERDICTS, Outcome, is_scored

# The values episodes on disk carry, pinned against the literal rather than against the enum under
# test: a change here orphans every recording scored with the old word.
WIRE_VALUES = ('Success', 'Fail', 'Ran out of time', 'Safety', 'System', 'UNSCORED', 'Discarded')


def test_the_wire_values_are_what_recordings_carry():
    assert tuple(outcome.value for outcome in Outcome) == WIRE_VALUES


def test_every_outcome_is_a_verdict_or_one_of_the_two_that_are_not():
    assert set(VERDICTS) | {Outcome.UNSCORED, Outcome.DISCARDED} == set(Outcome)


def test_only_a_verdict_is_scored():
    assert [is_scored(outcome) for outcome in Outcome] == [True, True, True, True, True, False, False]


def test_an_absent_outcome_is_not_scored():
    assert not is_scored(None)
    assert is_scored(ABSENT) is False


def test_a_word_this_vocabulary_does_not_know_is_not_scored():
    assert not is_scored('Rescored by hand')


def test_a_member_compares_and_hashes_as_the_string_a_recording_holds():
    # A reader that never converts still finds the member: the badge table is keyed by members and
    # looked up with whatever the episode holds.
    assert 'Success' == Outcome.SUCCESS
    assert {Outcome.SUCCESS: 'green'}['Success'] == 'green'
