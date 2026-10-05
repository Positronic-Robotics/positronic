"""What a station console holds about its run: the trial the next Start sends, the operator's instruction
override, and each episode so far."""

import threading
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from positronic.eval import Task
from positronic.eval import keys as eval_keys


class Outcome(StrEnum):
    """How an episode ended. The operator gives the first two, and the harness gives the others."""

    PASS = 'pass'
    FAIL = 'fail'
    TIMEOUT = 'timeout'
    ERROR = 'error'


Verdict = Literal[Outcome.PASS, Outcome.FAIL]


def terminal_payload(verdict: Verdict) -> dict[str, Any]:
    """The ``done`` payload that ends an episode with ``verdict``. The harness records it in the episode's statics."""
    return {eval_keys.ENDED_BY: eval_keys.ENDED_BY_OPERATOR, eval_keys.SUCCESS: verdict is Outcome.PASS}


def outcome_of(result: dict[str, Any]) -> Outcome:
    """The outcome that the harness's answer to an episode carries."""
    if not result[eval_keys.TERMINATED]:
        return Outcome.TIMEOUT
    return Outcome.PASS if result[eval_keys.SUCCESS] else Outcome.FAIL


class Phase(StrEnum):
    READY = 'ready'
    RUNNING = 'running'
    # The operator gave a verdict, and the harness has not answered yet.
    ENDING = 'ending'


class Episode(BaseModel):
    """One attempt. The operator's Starts number the attempts from 1."""

    number: int
    instruction: str
    overridden: bool
    started_at: float
    # Both stay ``None`` while the episode is open.
    ended_at: float | None = None
    outcome: Outcome | None = None


class RunView(BaseModel):
    """The station at ``now``, on the clock that stamps ``Episode.started_at``."""

    phase: Phase
    configured: str
    override: str | None
    # The first episode that sent the override in force. ``None`` until an episode sends it.
    override_since: int | None
    episodes: list[Episode]
    now: float


class Refused(RuntimeError):
    """The station's state does not allow the action. The message says why."""


class Station:
    """The console's record of one run. Each method takes a lock, so the web server and the control loop share it.

    ``next_task`` makes a trial before it is needed, so the page shows the instruction that the next Start sends.
    A trial must carry that instruction as a string: one known only after the trial's reset has nothing to show.
    """

    def __init__(self, next_task: Callable[[], Task]):
        self._next_task = next_task
        self._trial = self._draw()
        self._override: str | None = None
        self._override_since: int | None = None
        self._episodes: list[Episode] = []
        self._ending = False
        self._lock = threading.Lock()

    def _draw(self) -> Task:
        trial = self._next_task()
        if not isinstance(trial.instruction_source, str):
            raise TypeError('the station console shows the instruction before Start, so it must be a string')
        return trial

    def _is_open(self) -> bool:
        return bool(self._episodes) and self._episodes[-1].outcome is None

    def set_override(self, text: str | None) -> None:
        """Send ``text`` in place of the trial's instruction from the next Start on. ``None`` or the trial's own
        instruction removes the override."""
        with self._lock:
            if self._is_open():
                raise Refused('the instruction is locked while an episode runs')
            override = None if text == self._trial.instruction else text
            if override != self._override:
                self._override, self._override_since = override, None

    def start(self, now: float) -> Task:
        """Open the next episode and return its trial, with the override and the episode's number applied."""
        with self._lock:
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
            meta = {**trial.meta, eval_keys.TRIAL_INDEX: number - 1, eval_keys.INSTRUCTION_OVERRIDDEN: overridden}
            return replace(trial, meta=meta)

    def end(self, verdict: Verdict) -> dict[str, Any]:
        """Mark the open episode as ending, and return the ``done`` payload that ends it with ``verdict``."""
        with self._lock:
            if not self._is_open() or self._ending:
                raise Refused('no episode is running')
            self._ending = True
            return terminal_payload(verdict)

    def close(self, outcome: Outcome, now: float) -> None:
        """Record how the open episode ended."""
        with self._lock:
            episode = self._episodes[-1]
            episode.ended_at, episode.outcome = now, outcome
            self._ending = False

    def view(self, now: float) -> RunView:
        with self._lock:
            phase = Phase.ENDING if self._ending else Phase.RUNNING if self._is_open() else Phase.READY
            return RunView(
                phase=phase,
                configured=self._trial.instruction,
                override=self._override,
                override_since=self._override_since,
                episodes=[episode.model_copy() for episode in self._episodes],
                now=now,
            )
