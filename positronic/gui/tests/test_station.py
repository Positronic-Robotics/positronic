import itertools
from collections.abc import Callable

import pytest
from eval_vocabulary.outcome import OUTCOME, Outcome

from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.gui import station as station_module
from positronic.gui.station import Phase, Refused, Station, Unrecorded, outcome_of, terminal_payload
from positronic.policy.harness import EpisodeEnd

CONFIGURED = 'put the cup in the tote'
OVERRIDE = 'put the red cup in the grey tote'
UID = 'recording-1'


def _trials(instruction: str = CONFIGURED):
    """A trial source that numbers its draws in the trial's meta, so a test can tell the trials apart."""
    draws = itertools.count()
    return lambda: Task(instruction_source=instruction, timeout_sec=None, meta={'draw': next(draws)})


class _Recordings:
    """The outcomes that a station writes into recordings, by uid."""

    def __init__(self):
        self.written: list[tuple[str, Outcome]] = []

    def __call__(self, uid: str, outcome: Outcome) -> None:
        self.written.append((uid, outcome))


def _station(trials=None, recordings: Callable[[str, Outcome], None] | None = None) -> Station:
    return Station(trials or _trials(), recordings or _Recordings())


def _judged(verdict, uid: str | None = UID) -> EpisodeEnd:
    """The harness's answer to an episode that the judge ended with ``verdict``."""
    return EpisodeEnd({**terminal_payload(verdict), eval_keys.TERMINATED: True}, uid)


POLICY_END = EpisodeEnd({eval_keys.ENDED_BY: eval_keys.ENDED_BY_POLICY, eval_keys.TERMINATED: True}, UID)
TIMED_OUT = EpisodeEnd({eval_keys.TERMINATED: False}, UID)


def test_start_sends_the_trial_drawn_before_it_and_draws_the_next():
    station = _station()
    task = station.start(now=10.0)
    assert task.instruction == CONFIGURED
    assert task.meta == {'draw': 0, eval_keys.TRIAL_INDEX: 0, eval_keys.INSTRUCTION_OVERRIDDEN: False}
    station.close(_judged(Outcome.SUCCESS), now=12.0)
    assert station.start(now=20.0).meta['draw'] == 1


def test_an_override_replaces_the_instruction_until_it_is_reset():
    station = _station()
    station.set_override(OVERRIDE)
    for now in (1.0, 2.0):
        task = station.start(now)
        assert task.instruction == OVERRIDE
        assert task.meta[eval_keys.INSTRUCTION_OVERRIDDEN] is True
        station.close(_judged(Outcome.SUCCESS), now + 0.5)
    assert station.view(now=3.0).override_since == 1

    station.set_override(None)
    task = station.start(now=4.0)
    assert task.instruction == CONFIGURED
    assert task.meta[eval_keys.INSTRUCTION_OVERRIDDEN] is False


def test_the_configured_text_is_no_override():
    station = _station()
    station.set_override(CONFIGURED)
    assert station.view(now=0.0).override is None


def test_a_changed_override_has_no_episode_until_one_sends_it():
    station = _station()
    station.set_override(OVERRIDE)
    station.start(now=1.0)
    station.close(_judged(Outcome.FAIL), now=2.0)
    station.set_override('another text')
    assert station.view(now=3.0).override_since is None
    station.start(now=4.0)
    assert station.view(now=5.0).override_since == 2


def test_the_instruction_is_locked_while_an_episode_runs():
    station = _station()
    station.start(now=1.0)
    with pytest.raises(Refused, match='locked'):
        station.set_override(OVERRIDE)


def test_each_attempt_takes_the_next_number():
    station = _station()
    for verdict in (Outcome.FAIL, Outcome.SUCCESS):
        station.start(now=1.0)
        station.end(verdict)
        station.close(_judged(verdict), now=2.0)
    task = station.start(now=3.0)
    assert task.meta[eval_keys.TRIAL_INDEX] == 2
    assert [episode.number for episode in station.view(now=4.0).episodes] == [1, 2, 3]


def test_the_phase_follows_start_verdict_and_answer():
    station = _station()
    assert station.view(now=0.0).phase is Phase.READY
    station.start(now=1.0)
    assert station.view(now=1.5).phase is Phase.RUNNING
    station.end(Outcome.SUCCESS)
    assert station.view(now=2.0).phase is Phase.ENDING
    station.close(_judged(Outcome.SUCCESS), now=2.5)
    view = station.view(now=3.0)
    assert view.phase is Phase.READY
    assert (view.episodes[0].started_at, view.episodes[0].ended_at) == (1.0, 2.5)


def test_a_failed_episode_keeps_its_error_and_the_next_start_opens_the_next_episode():
    station = _station()
    station.start(now=1.0)
    station.fail('RuntimeError: the arm stopped short of its target', now=2.0)
    view = station.view(now=3.0)
    assert view.phase is Phase.READY
    assert view.episodes[0].outcome is Unrecorded.ERROR
    assert view.episodes[0].error == 'RuntimeError: the arm stopped short of its target'
    assert station.start(now=4.0).meta[eval_keys.TRIAL_INDEX] == 1


def test_the_run_ends_only_after_the_open_episode_and_then_takes_no_start():
    station = _station()
    station.start(now=1.0)
    with pytest.raises(Refused, match='finish the episode first'):
        station.end_run()
    assert not station.run_ended
    station.close(_judged(Outcome.SUCCESS), now=2.0)
    station.end_run()
    assert station.run_ended
    assert station.view(now=3.0).phase is Phase.RUN_ENDED
    with pytest.raises(Refused, match='the run has ended'):
        station.start(now=4.0)


def test_one_episode_runs_at_a_time_and_accepts_verdicts_until_it_closes():
    station = _station()
    with pytest.raises(Refused):
        station.end(Outcome.SUCCESS)
    station.start(now=1.0)
    with pytest.raises(Refused):
        station.start(now=1.5)
    assert station.end(Outcome.FAIL) == terminal_payload(Outcome.FAIL)
    assert station.end(Outcome.DISCARDED) == terminal_payload(Outcome.DISCARDED)
    station.close(_judged(Outcome.DISCARDED), now=2.0)
    with pytest.raises(Refused):
        station.end(Outcome.SUCCESS)


def test_a_verdict_that_ends_the_episode_is_recorded_by_the_harness_and_not_written_again():
    recordings = _Recordings()
    station = _station(recordings=recordings)
    station.start(now=1.0)
    station.close(_judged(Outcome.DISCARDED), now=2.0)
    assert station.view(now=3.0).episodes[0].outcome is Outcome.DISCARDED
    assert recordings.written == []


def test_an_episode_that_runs_out_of_time_is_written_out_of_time():
    recordings = _Recordings()
    station = _station(recordings=recordings)
    station.start(now=1.0)
    station.close(TIMED_OUT, now=2.0)
    view = station.view(now=3.0)
    assert (view.phase, view.episodes[0].outcome) == (Phase.READY, Outcome.OUT_OF_TIME)
    assert recordings.written == [(UID, Outcome.OUT_OF_TIME)]


@pytest.mark.parametrize('verdict', [Outcome.SUCCESS, Outcome.FAIL, Outcome.DISCARDED])
def test_after_a_policy_end_the_verdict_is_written_into_the_recording(verdict):
    recordings = _Recordings()
    station = _station(recordings=recordings)
    station.start(now=1.0)
    station.close(POLICY_END, now=2.0)
    view = station.view(now=3.0)
    assert view.phase is Phase.VERDICT_PENDING
    assert (view.episodes[0].ended_at, view.episodes[0].outcome) == (2.0, None)
    assert recordings.written == []

    assert station.end(verdict) is None
    view = station.view(now=4.0)
    assert (view.phase, view.episodes[0].outcome) == (Phase.READY, verdict)
    assert recordings.written == [(UID, verdict)]


def test_a_pending_verdict_refuses_start_and_the_end_of_the_run():
    station = _station()
    station.start(now=1.0)
    station.close(POLICY_END, now=2.0)
    with pytest.raises(Refused, match='give the verdict'):
        station.start(now=3.0)
    with pytest.raises(Refused, match='finish the episode first'):
        station.end_run()
    station.end(Outcome.FAIL)
    with pytest.raises(Refused, match='no episode is running'):
        station.end(Outcome.SUCCESS)
    station.start(now=4.0)


def test_a_verdict_on_an_episode_that_recorded_nothing_writes_nothing():
    recordings = _Recordings()
    station = _station(recordings=recordings)
    station.start(now=1.0)
    station.close(EpisodeEnd(POLICY_END.terminal, None), now=2.0)
    station.end(Outcome.SUCCESS)
    station.start(now=3.0)
    station.close(EpisodeEnd(TIMED_OUT.terminal, None), now=4.0)
    assert recordings.written == []
    assert [episode.outcome for episode in station.view(now=5.0).episodes] == [Outcome.SUCCESS, Outcome.OUT_OF_TIME]


def test_a_failed_write_keeps_the_verdict_pending():
    def broken(uid: str, outcome: Outcome) -> None:
        raise OSError('disk full')

    station = _station(recordings=broken)
    station.start(now=1.0)
    station.close(POLICY_END, now=2.0)
    with pytest.raises(OSError, match='disk full'):
        station.end(Outcome.SUCCESS)
    assert station.view(now=3.0).phase is Phase.VERDICT_PENDING


def test_an_out_of_time_write_that_fails_closes_the_episode_and_shows_the_error_on_its_row(caplog):
    def broken(uid: str, outcome: Outcome) -> None:
        raise OSError('disk full')

    station = _station(recordings=broken)
    station.start(now=1.0)
    station.close(TIMED_OUT, now=2.0)
    view = station.view(now=3.0)
    assert view.phase is Phase.READY
    assert (view.episodes[0].outcome, view.episodes[0].error) == (Outcome.OUT_OF_TIME, 'OSError: disk full')
    assert 'disk full' in caplog.text


def test_a_verdict_pressed_while_a_policy_end_is_answered_closes_the_episode():
    recordings = _Recordings()
    station = _station(recordings=recordings)
    station.start(now=1.0)
    station.end(Outcome.DISCARDED)
    station.close(POLICY_END, now=2.0)
    view = station.view(now=3.0)
    assert (view.phase, view.episodes[0].outcome) == (Phase.READY, Outcome.DISCARDED)
    assert recordings.written == [(UID, Outcome.DISCARDED)]


def test_a_verdict_of_one_episode_does_not_close_the_next():
    station = _station()
    station.start(now=1.0)
    station.end(Outcome.SUCCESS)
    station.close(_judged(Outcome.SUCCESS), now=2.0)
    station.start(now=3.0)
    station.close(POLICY_END, now=4.0)
    assert station.view(now=5.0).phase is Phase.VERDICT_PENDING


def test_a_trial_source_that_fails_leaves_the_station_as_it_was():
    draws = iter([Task(instruction_source=CONFIGURED, timeout_sec=None)])
    station = _station(lambda: next(draws))
    with pytest.raises(StopIteration):
        station.start(now=1.0)
    view = station.view(now=2.0)
    assert view.phase is Phase.READY
    assert view.episodes == []


@pytest.mark.parametrize('verdict', [Outcome.SUCCESS, Outcome.FAIL, Outcome.DISCARDED])
def test_the_outcome_reads_back_the_verdict_from_the_harness_answer(verdict):
    assert outcome_of(_judged(verdict).terminal) is verdict


def test_a_verdict_that_arrives_after_the_budget_reads_as_out_of_time():
    assert outcome_of({eval_keys.TERMINATED: False}) is Outcome.OUT_OF_TIME


def test_an_answer_with_no_outcome_reads_as_none():
    assert outcome_of(POLICY_END.terminal) is None


def test_the_verdicts_record_who_ended_the_episode_and_the_outcome():
    for verdict in (Outcome.SUCCESS, Outcome.FAIL, Outcome.DISCARDED):
        assert terminal_payload(verdict) == {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, OUTCOME: verdict}


def _known_after_reset() -> str:
    return CONFIGURED


def test_a_trial_whose_instruction_is_known_only_after_its_reset_is_refused_at_startup():
    with pytest.raises(TypeError, match='string'):
        _station(lambda: Task(instruction_source=_known_after_reset, timeout_sec=None))


def test_a_later_trial_whose_instruction_is_known_only_after_its_reset_is_refused_at_start():
    draws = iter([
        Task(instruction_source=CONFIGURED, timeout_sec=None),
        Task(instruction_source=_known_after_reset, timeout_sec=None),
    ])
    station = _station(lambda: next(draws))
    with pytest.raises(TypeError, match='string'):
        station.start(now=1.0)
    view = station.view(now=2.0)
    assert view.phase is Phase.READY
    assert view.episodes == []


def test_each_change_advances_the_generation_and_a_view_does_not():
    station = _station()
    generations = [station.view(now=0.0).generation]

    def note():
        generations.append(station.view(now=0.0).generation)

    station.set_override(OVERRIDE)
    note()
    station.start(now=1.0)
    note()
    station.end(Outcome.SUCCESS)
    note()
    station.close(_judged(Outcome.SUCCESS), now=2.0)
    note()
    note()
    station.start(now=3.0)
    station.close(POLICY_END, now=4.0)
    note()
    station.end(Outcome.FAIL)
    note()
    assert generations == sorted(generations)
    assert len(set(generations)) == 7


def test_a_restarted_console_starts_above_the_generation_of_the_one_before_it(monkeypatch):
    monkeypatch.setattr(station_module.time, 'time_ns', lambda: 1_000_000_000_000)
    before = _station()
    for _ in range(1000):
        before.set_override(OVERRIDE)
    monkeypatch.setattr(station_module.time, 'time_ns', lambda: 1_001_000_000_000)
    assert _station().view(now=0.0).generation > before.view(now=0.0).generation
