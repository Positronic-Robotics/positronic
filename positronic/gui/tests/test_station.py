import itertools

import pytest

from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.gui.station import Outcome, Phase, Refused, Station, outcome_of, terminal_payload

CONFIGURED = 'put the cup in the tote'
OVERRIDE = 'put the red cup in the grey tote'


def _trials(instruction: str = CONFIGURED):
    """A trial source that numbers its draws in the trial's meta, so a test can tell the trials apart."""
    draws = itertools.count()
    return lambda: Task(instruction_source=instruction, timeout_sec=None, meta={'draw': next(draws)})


def test_start_sends_the_trial_drawn_before_it_and_draws_the_next():
    station = Station(_trials())
    task = station.start(now=10.0)
    assert task.instruction == CONFIGURED
    assert task.meta == {'draw': 0, eval_keys.TRIAL_INDEX: 0, eval_keys.INSTRUCTION_OVERRIDDEN: False}
    station.close(Outcome.PASS, now=12.0)
    assert station.start(now=20.0).meta['draw'] == 1


def test_an_override_replaces_the_instruction_until_it_is_reset():
    station = Station(_trials())
    station.set_override(OVERRIDE)
    for now in (1.0, 2.0):
        task = station.start(now)
        assert task.instruction == OVERRIDE
        assert task.meta[eval_keys.INSTRUCTION_OVERRIDDEN] is True
        station.close(Outcome.PASS, now + 0.5)
    assert station.view(now=3.0).override_since == 1

    station.set_override(None)
    task = station.start(now=4.0)
    assert task.instruction == CONFIGURED
    assert task.meta[eval_keys.INSTRUCTION_OVERRIDDEN] is False


def test_the_configured_text_is_no_override():
    station = Station(_trials())
    station.set_override(CONFIGURED)
    assert station.view(now=0.0).override is None


def test_a_changed_override_has_no_episode_until_one_sends_it():
    station = Station(_trials())
    station.set_override(OVERRIDE)
    station.start(now=1.0)
    station.close(Outcome.FAIL, now=2.0)
    station.set_override('another text')
    assert station.view(now=3.0).override_since is None
    station.start(now=4.0)
    assert station.view(now=5.0).override_since == 2


def test_the_instruction_is_locked_while_an_episode_runs():
    station = Station(_trials())
    station.start(now=1.0)
    with pytest.raises(Refused, match='locked'):
        station.set_override(OVERRIDE)


def test_each_attempt_takes_the_next_number_and_a_discarded_one_keeps_its_own():
    station = Station(_trials())
    for outcome in (Outcome.DISCARDED, Outcome.PASS):
        station.start(now=1.0)
        station.end(outcome)
        station.close(outcome, now=2.0)
    task = station.start(now=3.0)
    assert task.meta[eval_keys.TRIAL_INDEX] == 2
    assert [episode.number for episode in station.view(now=4.0).episodes] == [1, 2, 3]


def test_the_phase_follows_start_verdict_and_answer():
    station = Station(_trials())
    assert station.view(now=0.0).phase is Phase.READY
    station.start(now=1.0)
    assert station.view(now=1.5).phase is Phase.RUNNING
    station.end(Outcome.PASS)
    assert station.view(now=2.0).phase is Phase.ENDING
    station.close(Outcome.PASS, now=2.5)
    view = station.view(now=3.0)
    assert view.phase is Phase.READY
    assert (view.episodes[0].started_at, view.episodes[0].ended_at) == (1.0, 2.5)


def test_one_episode_runs_at_a_time_and_takes_one_verdict():
    station = Station(_trials())
    with pytest.raises(Refused):
        station.end(Outcome.PASS)
    station.start(now=1.0)
    with pytest.raises(Refused):
        station.start(now=1.5)
    station.end(Outcome.FAIL)
    with pytest.raises(Refused):
        station.end(Outcome.PASS)


def test_a_trial_source_that_fails_leaves_the_station_as_it_was():
    draws = iter([Task(instruction_source=CONFIGURED, timeout_sec=None)])
    station = Station(lambda: next(draws))
    with pytest.raises(StopIteration):
        station.start(now=1.0)
    view = station.view(now=2.0)
    assert view.phase is Phase.READY
    assert view.episodes == []


@pytest.mark.parametrize('verdict', [Outcome.PASS, Outcome.FAIL, Outcome.DISCARDED])
def test_the_outcome_reads_back_the_verdict_from_the_harness_answer(verdict):
    assert outcome_of({**terminal_payload(verdict), eval_keys.TERMINATED: True}) is verdict


def test_a_verdict_that_arrives_after_the_budget_reads_as_a_timeout():
    assert outcome_of({eval_keys.TERMINATED: False}) is Outcome.TIMEOUT


def test_the_verdicts_record_who_ended_the_episode():
    assert terminal_payload(Outcome.PASS) == {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, eval_keys.SUCCESS: True}
    assert terminal_payload(Outcome.DISCARDED) == {
        eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR,
        eval_keys.DISCARDED: True,
    }
