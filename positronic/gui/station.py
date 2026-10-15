"""What a station console holds about its run: the trial the next Start sends, the operator's instruction
override, and each episode so far."""

import logging
import threading
import time
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


# What the judge gives an episode: the verdict success or fail, or a discard, which leaves the episode unscored.
Judgement = Literal[Outcome.SUCCESS, Outcome.FAIL, Outcome.DISCARDED]
# How an episode ended. ``OUT_OF_TIME`` is what the page shows for ``eval.terminated = False``. Nothing writes it.
Ending = Judgement | Literal[Outcome.OUT_OF_TIME] | Unrecorded


def terminal_payload(judgement: Judgement) -> dict[str, Any]:
    """The ``done`` payload that ends an episode with ``judgement``. The harness records it in the episode's statics."""
    return {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, OUTCOME: judgement}


def outcome_of(terminal: dict[str, Any]) -> Judgement | Literal[Outcome.OUT_OF_TIME] | None:
    """The outcome that the harness's terminal payload carries. ``None`` when the judge must still give it."""
    if not terminal[eval_keys.TERMINATED]:
        return Outcome.OUT_OF_TIME
    return terminal.get(OUTCOME)


class Phase(StrEnum):
    READY = 'ready'
    RUNNING = 'running'
    # The judge ended the episode, and the harness has not answered yet.
    ENDING = 'ending'
    # The episode ended with no outcome, as when the policy ends it. The judge gives it before the next Start.
    JUDGEMENT_PENDING = 'judgement_pending'
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
    # The error that failed an ``ERROR`` episode, or that stopped the console from writing the judgement into the
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
    # Rises with each change to the run, so the page drops an answer older than one it has drawn.
    generation: int
    now: float


class Refused(RuntimeError):
    """The station's state does not allow the action. The message says why."""


class Station:
    """The console's record of one run. Each method takes a lock, so the web server and the control loop share it.

    ``next_task`` makes a trial before it is needed, so the page shows the instruction that the next Start sends.
    A trial must carry that instruction as a string: one known only after the trial's reset has nothing to show.
    ``record(uid, judgement)`` writes ``judgement`` into the recording ``uid``. The station calls it for a judgement
    that the recording does not hold.
    """

    def __init__(self, next_task: Callable[[], Task], record: Callable[[str, Judgement], None]):
        self._next_task = next_task
        self._record = record
        self._trial = self._draw()
        self._override: str | None = None
        self._override_since: int | None = None
        self._episodes: list[Episode] = []
        # The judgement that ends the open episode, from its press until the harness answers.
        self._ending_with: Judgement | None = None
        # The recording of the episode that waits for a judgement. ``None`` when that episode recorded nothing.
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

    def _awaits_judgement(self) -> bool:
        """The harness answered the open episode with no outcome."""
        return self._is_open() and self._episodes[-1].ended_at is not None

    def set_override(self, text: str | None) -> None:
        """Send ``text`` in place of the trial's instruction from the next Start on. ``None`` or the trial's own
        instruction removes the override."""
        with self._lock:
            if self._is_open():
                raise Refused('the instruction is locked while an episode runs')
            override = None if text == self._trial.instruction else text
            if override != self._override:
                self._override, self._override_since = override, None
            self._generation += 1

    def start(self, now: float) -> Task:
        """Open the next episode and return its trial, with the override and the episode's number applied."""
        with self._lock:
            if self._run_ended:
                raise Refused('the run has ended')
            if self._awaits_judgement():
                raise Refused('judge the last episode first')
            if self._is_open():
                raise Refused('an episode is already running')
            trial, self._trial = self._trial, self._draw()
            number = len(self._episodes) + 1
            override = self._override
            if override is not None:
                trial = replace(trial, instruction_source=override)
                if self._override_since is None:
                    self._override_since = number
            overridden = override is not None
            episode = Episode(number=number, instruction=trial.instruction, overridden=overridden, started_at=now)
            self._episodes.append(episode)
            self._generation += 1
            meta = {**trial.meta, eval_keys.TRIAL_INDEX: number - 1, eval_keys.INSTRUCTION_OVERRIDDEN: overridden}
            return replace(trial, meta=meta)

    def end(self, judgement: Judgement) -> dict[str, Any] | None:
        """Give ``judgement`` to the open episode. Return the ``done`` payload that ends a running episode, or ``None``
        for an episode that waits for a judgement. Refused when the judgement does not reach the recording."""
        with self._lock:
            if not self._is_open():
                raise Refused('no episode is running')
            if not self._awaits_judgement():
                self._ending_with = judgement
                self._generation += 1
                return terminal_payload(judgement)
            self._generation += 1
            self._judge(self._pending_uid, judgement)
            return None

    def close(self, end: EpisodeEnd, now: float) -> None:
        """Close the open episode on the harness's answer.

        The answer's outcome stands, except that a Discard pressed for the episode wins over its time limit. An answer
        with no outcome takes the judgement pressed for the episode, or waits for one.
        """
        with self._lock:
            pressed, self._ending_with = self._ending_with, None
            answered = outcome_of(end.terminal)
            episode = self._episodes[-1]
            episode.ended_at = now
            self._generation += 1
            if answered is not None and not (answered is Outcome.OUT_OF_TIME and pressed is Outcome.DISCARDED):
                episode.outcome = answered
            elif pressed is None:
                self._pending_uid = end.uid
            else:
                # rules-allow: swallowed-error — the episode keeps waiting for a judgement, and its row shows the error
                try:
                    self._judge(end.uid, pressed)
                except Refused as exc:
                    logger.error(f'Episode {episode.number} waits for a judgement again: {exc}')

    def _judge(self, uid: str | None, judgement: Judgement) -> None:
        """Close the open episode with ``judgement``, and write it into the recording ``uid``.

        When the write fails, the episode keeps waiting for a judgement, its row shows the error, and this raises
        ``Refused``.
        """
        episode = self._episodes[-1]
        if uid is not None:
            try:
                self._record(uid, judgement)
            except (OSError, ValueError) as exc:
                self._pending_uid, episode.error = uid, f'{type(exc).__name__}: {exc}'
                raise Refused(f'the judgement is not in the recording: {episode.error}') from exc
        episode.outcome, episode.error, self._pending_uid = judgement, None, None

    def fail(self, error: str, now: float) -> None:
        """Close the open episode, which the harness failed with ``error``."""
        with self._lock:
            episode = self._episodes[-1]
            episode.ended_at, episode.outcome, episode.error = now, Unrecorded.ERROR, error
            self._ending_with = None
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
        if self._awaits_judgement():
            return Phase.JUDGEMENT_PENDING
        if self._ending_with is not None:
            return Phase.ENDING
        return Phase.RUNNING if self._is_open() else Phase.READY

    def view(self, now: float) -> RunView:
        with self._lock:
            return RunView(
                phase=self._phase(),
                configured=self._trial.instruction,
                override=self._override,
                override_since=self._override_since,
                episodes=[episode.model_copy() for episode in self._episodes],
                generation=self._generation,
                now=now,
            )
