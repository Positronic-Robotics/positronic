import pytest
from eval_vocabulary.episode import (
    ENDED_BY,
    ENDED_BY_OPERATOR,
    ENDED_BY_POLICY,
    ENV_ENDS_ON_SUCCESS,
    POLICY_END,
    SUCCESS,
    TERMINATED,
    PolicyEnd,
)
from eval_vocabulary.outcome import ABSENT, OUTCOME, VERDICTS, Outcome, counted_outcome, is_scored

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


# The statics of trials that the policy ended before their time limit, with no verdict.
GAVE_UP = {ENDED_BY: ENDED_BY_POLICY, POLICY_END: PolicyEnd.GIVE_UP, TERMINATED: True}
COMPLETED = {ENDED_BY: ENDED_BY_POLICY, POLICY_END: PolicyEnd.COMPLETE, TERMINATED: True}


@pytest.mark.parametrize('verdict', [*VERDICTS, Outcome.DISCARDED])
def test_a_verdict_or_a_discard_counts_as_it_is_over_every_other_fact(verdict):
    assert counted_outcome({**GAVE_UP, OUTCOME: verdict, SUCCESS: True, ENV_ENDS_ON_SUCCESS: True}) is verdict
    assert counted_outcome({OUTCOME: verdict, TERMINATED: False}) is verdict


@pytest.mark.parametrize('value', [Outcome.UNSCORED, 'Rescored by hand', 3])
def test_an_unscored_outcome_or_a_value_this_vocabulary_does_not_know_decides_nothing(value):
    assert counted_outcome({OUTCOME: value, SUCCESS: False}) is Outcome.FAIL
    assert counted_outcome({OUTCOME: value}) is Outcome.UNSCORED


def test_a_trial_that_ran_out_of_time_counts_as_that_whatever_the_env_reported():
    assert counted_outcome({TERMINATED: False}) is Outcome.OUT_OF_TIME
    assert counted_outcome({TERMINATED: False, SUCCESS: True}) is Outcome.OUT_OF_TIME


@pytest.mark.parametrize(('success', 'counted'), [(True, Outcome.SUCCESS), (False, Outcome.FAIL)])
def test_the_env_report_counts_as_a_success_or_a_failure_over_a_policy_end(success, counted):
    assert counted_outcome({SUCCESS: success, TERMINATED: True}) is counted
    assert counted_outcome({**COMPLETED, ENV_ENDS_ON_SUCCESS: True, SUCCESS: success}) is counted


def test_an_env_report_that_is_not_a_bool_is_no_report():
    assert counted_outcome({SUCCESS: 'yes', TERMINATED: True}) is Outcome.UNSCORED


@pytest.mark.parametrize('ends_on_success', [None, False, True])
def test_a_give_up_counts_as_a_failure_in_every_env(ends_on_success):
    static = GAVE_UP if ends_on_success is None else {**GAVE_UP, ENV_ENDS_ON_SUCCESS: ends_on_success}
    assert counted_outcome(static) is Outcome.FAIL


def test_a_complete_in_an_env_that_ends_on_success_counts_as_a_failure():
    assert counted_outcome({**COMPLETED, ENV_ENDS_ON_SUCCESS: True}) is Outcome.FAIL


@pytest.mark.parametrize('static', [COMPLETED, {**COMPLETED, ENV_ENDS_ON_SUCCESS: False}])
def test_a_complete_that_nobody_confirmed_is_unscored(static):
    assert counted_outcome(static) is Outcome.UNSCORED


@pytest.mark.parametrize(
    'static',
    [
        {},
        {ENDED_BY: ENDED_BY_OPERATOR, TERMINATED: True, ENV_ENDS_ON_SUCCESS: True},
        {POLICY_END: PolicyEnd.GIVE_UP, TERMINATED: True},
    ],
)
def test_an_episode_that_no_policy_end_or_other_fact_decides_is_unscored(static):
    assert counted_outcome(static) is Outcome.UNSCORED
