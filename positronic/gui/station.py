"""What a station console holds about its run: the trial the next Start sends, the operator's instruction
override, and each episode so far."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import Any, Literal

from eval_vocabulary.outcome import OUTCOME, Outcome
from pydantic import BaseModel

from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.policy.harness import EpisodeEnd

logger = logging.getLogger(__name__)


class Unrecorded(StrEnum):
    """How an episode ended when it recorded nothing. Each other ending is an ``Outcome`` of ``eval_vocabulary``."""

    ERROR = 'error'
    # The operator aborted the episode before its policy started.
    ABORTED = 'aborted'


# What the judge gives an episode: success, fail, or a discard.
Verdict = Literal[Outcome.SUCCESS, Outcome.FAIL, Outcome.DISCARDED]
# How an episode ended. The console gives ``OUT_OF_TIME`` when the time limit ends an episode.
Ending = Verdict | Literal[Outcome.OUT_OF_TIME] | Unrecorded


def terminal_payload(verdict: Verdict) -> dict[str, Any]:
    """The ``done`` payload that ends an episode with ``verdict``. The harness records it in the episode's statics."""
    return {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, OUTCOME: verdict}


def outcome_of(terminal: dict[str, Any]) -> Verdict | Literal[Outcome.OUT_OF_TIME] | None:
    """The outcome that the harness's terminal payload carries. ``None`` when the judge must still give it."""
    if not terminal[eval_keys.TERMINATED]:
        return Outcome.OUT_OF_TIME
    return terminal.get(OUTCOME)


class Phase(StrEnum):
    READY = 'ready'
    # The harness readies the rig. The operator can abort the episode, and cannot give a verdict yet.
    STARTING = 'starting'
    RUNNING = 'running'
    # The judge ended the episode or the operator aborted it, and the harness has not answered yet.
    ENDING = 'ending'
    # The episode ended with no outcome, as on a policy END. The judge gives it before the next Start.
    VERDICT_PENDING = 'verdict_pending'
    # The operator ended the run. The console stops, and that stops the World.
    RUN_ENDED = 'run_ended'


class Episode(BaseModel):
    """One attempt. The operator's Starts number the attempts from 1."""

    number: int
    instruction: str
    overridden: bool
    started_at: float
    # ``ended_at`` stays ``None`` until the harness answers, and ``outcome`` until the episode has one.
    ended_at: float | None = None
    outcome: Ending | None = None
    # The error that failed an ``ERROR`` episode, or that stopped the console from writing the outcome into the
    # episode's recording, as the last line of its traceback.
    error: str | None = None


class RunView(BaseModel):
    """The station at ``now``, on the clock that stamps ``Episode.started_at``."""

    phase: Phase
    configured: str
    override: str | None
    # The first episode that sent the override in force. ``None`` until an episode sends it.
    override_since: int | None
    episodes: list[Episode]
    # Why the harness refused the last abort of the open episode. ``None`` until it refuses one.
    abort_refusal: str | None
    # Rises with each change to the run, so the page drops an answer older than one it has drawn.
    generation: int
    now: float


class Refused(RuntimeError):
    """The station's state does not allow the action. The message says why."""


class Station:
    """The console's record of one run. Each method takes a lock, so the web server and the control loop share it.

    ``next_task`` makes a trial before it is needed, so the page shows the instruction that the next Start sends.
    A trial must carry that instruction as a string: one known only after the trial's reset has nothing to show.
    An aborted episode does not use up its trial: the next Start sends it again.
    ``record(uid, outcome)`` writes ``outcome`` into the recording ``uid``. The station calls it for an outcome that it
    gives after the harness answered.
    """

    def __init__(self, next_task: Callable[[], Task], record: Callable[[str, Outcome], None]):
        self._next_task = next_task
        self._record = record
        # The first trial is the one of the open episode, or the one that the next Start sends.
        self._trials = deque([self._draw()])
        self._override: str | None = None
        self._override_since: int | None = None
        self._episodes: list[Episode] = []
        # The verdict that ends the open episode, from its press until the harness answers.
        self._ending_with: Verdict | None = None
        self._starting = False
        self._aborting = False
        self._abort_refusal: str | None = None
        # The recording of the episode that waits for a verdict. ``None`` when that episode recorded nothing.
        self._pending_uid: str | None = None
        self._run_ended = False
        # Starts at the wall clock in microseconds, so a restarted console starts above the one before it.
        self._generation = time.time_ns() // 1000
        self._lock = threading.Lock()

    def _draw(self) -> Task:
        trial = self._next_task()
        if not isinstance(trial.instruction_source, str):
            raise TypeError('the station console shows the instruction before Start, so it must be a string')
        return trial

    def _is_open(self) -> bool:
        return bool(self._episodes) and self._episodes[-1].outcome is None

    def _awaits_verdict(self) -> bool:
        """The harness answered the open episode with no outcome."""
        return self._is_open() and self._episodes[-1].ended_at is not None

    def set_override(self, text: str | None) -> None:
        """Send ``text`` in place of the trial's instruction from the next Start on. ``None`` or the trial's own
        instruction removes the override."""
        with self._lock:
            if self._is_open():
                raise Refused('the instruction is locked while an episode runs')
            override = None if text == self._trials[0].instruction else text
            if override != self._override:
                self._override, self._override_since = override, None
            self._generation += 1

    def start(self, now: float) -> Task:
        """Open the next episode and return its trial, with the override and the episode's number applied."""
        with self._lock:
            if self._run_ended:
                raise Refused('the run has ended')
            if self._awaits_verdict():
                raise Refused('give the verdict on the last episode first')
            if self._is_open():
                raise Refused('an episode is already running')
            if len(self._trials) == 1:
                self._trials.append(self._draw())
            trial = self._trials[0]
            number = len(self._episodes) + 1
            override = self._override
            if override is not None:
                trial = replace(trial, instruction_source=override)
                if self._override_since is None:
                    self._override_since = number
            overridden = override is not None
            episode = Episode(number=number, instruction=trial.instruction, overridden=overridden, started_at=now)
            self._episodes.append(episode)
            self._starting = True
            self._generation += 1
            meta = {**trial.meta, eval_keys.TRIAL_INDEX: number - 1, eval_keys.INSTRUCTION_OVERRIDDEN: overridden}
            return replace(trial, meta=meta)

    def policy_started(self) -> None:
        """Mark that the policy of the open episode runs."""
        with self._lock:
            if self._starting:
                self._starting = False
                self._generation += 1

    def abort(self) -> None:
        """Mark the open episode as ending. Only an episode whose policy has not started can abort."""
        with self._lock:
            if self._awaits_verdict():
                raise Refused('the episode has ended: give its verdict')
            if self._aborting:
                raise Refused('the abort of this episode is on its way')
            if self._ending_with is not None or not self._is_open() or not self._starting:
                raise Refused('no episode is starting')
            self._aborting = True
            self._abort_refusal = None
            self._generation += 1

    def abort_refused(self, reason: str) -> None:
        """Take back the abort of the open episode, which goes on, and keep ``reason`` for the page."""
        with self._lock:
            if not self._is_open():
                return
            self._aborting = False
            self._abort_refusal = reason
            self._generation += 1

    def end(self, verdict: Verdict) -> dict[str, Any]:
        """Mark the running episode as ending, and return the ``done`` payload that ends it with ``verdict``."""
        with self._lock:
            if self._awaits_verdict():
                raise Refused('the episode has ended: give its verdict')
            if not self._is_open():
                raise Refused('no episode is running')
            if self._starting:
                raise Refused('the policy has not started: abort the episode, or give the verdict when it runs')
            self._ending_with = verdict
            self._generation += 1
            return terminal_payload(verdict)

    def _closed(self, *, keep_trial: bool) -> None:
        """Clear what the station holds about the open episode while it runs."""
        if not keep_trial:
            self._trials.popleft()
        self._ending_with = None
        self._starting = self._aborting = False
        self._abort_refusal = None

    def close(self, end: EpisodeEnd, now: float) -> None:
        """Close the open episode on the harness's answer.

        An answer with no outcome takes the verdict pressed for this episode, or waits for the judge's verdict. The
        station writes an outcome that the recording does not hold, and the episode's row shows a write that failed.
        """
        with self._lock:
            outcome = outcome_of(end.terminal) or self._ending_with
            episode = self._episodes[-1]
            episode.ended_at, episode.outcome = now, outcome
            self._closed(keep_trial=False)
            if outcome is None:
                self._pending_uid = end.uid
            elif OUTCOME not in end.terminal and end.uid is not None:
                # rules-allow: swallowed-error — the recording is complete, and the row shows the error
                try:
                    self._record(end.uid, outcome)
                except (OSError, ValueError) as exc:
                    logger.error(f'The outcome {outcome} of episode {episode.number} is not in its recording: {exc}')
                    episode.error = f'{type(exc).__name__}: {exc}'
            self._generation += 1

    def fail(self, error: str, now: float) -> None:
        """Close the open episode, which the harness failed with ``error``."""
        with self._lock:
            episode = self._episodes[-1]
            episode.ended_at, episode.outcome, episode.error = now, Unrecorded.ERROR, error
            self._closed(keep_trial=False)
            self._generation += 1

    def aborted(self, now: float) -> None:
        """Close the open episode, which an abort ended before its policy started. The next Start sends its trial."""
        with self._lock:
            episode = self._episodes[-1]
            episode.ended_at, episode.outcome = now, Unrecorded.ABORTED
            self._closed(keep_trial=True)
            self._generation += 1

    def judge(self, verdict: Verdict) -> None:
        """Give ``verdict`` to the episode that waits for a verdict, and write it into that episode's recording."""
        with self._lock:
            if not self._awaits_verdict():
                raise Refused('no episode waits for a verdict')
            if self._pending_uid is not None:
                self._record(self._pending_uid, verdict)
            self._episodes[-1].outcome, self._pending_uid = verdict, None
            self._generation += 1

    def end_run(self) -> None:
        """End the run. An open episode must end first."""
        with self._lock:
            if self._is_open():
                raise Refused('finish the episode first, then end the run')
            self._run_ended = True
            self._generation += 1

    @property
    def run_ended(self) -> bool:
        with self._lock:
            return self._run_ended

    def _phase(self) -> Phase:
        if self._run_ended:
            return Phase.RUN_ENDED
        if self._awaits_verdict():
            return Phase.VERDICT_PENDING
        if self._ending_with is not None or self._aborting:
            return Phase.ENDING
        if not self._is_open():
            return Phase.READY
        return Phase.STARTING if self._starting else Phase.RUNNING

    def view(self, now: float) -> RunView:
        with self._lock:
            return RunView(
                phase=self._phase(),
                configured=self._trials[0].instruction,
                override=self._override,
                override_since=self._override_since,
                episodes=[episode.model_copy() for episode in self._episodes],
                abort_refusal=self._abort_refusal,
                generation=self._generation,
                now=now,
            )
