"""A run's `scores.json`, from the outcome that each recorded episode reports.

A scorer gives one episode its `Outcome`, and `tally` adds the outcomes of a run into `Scores`.
`PUBLIC_SCORERS` holds the scorers of the public evals, under the names that `EvalDefinition.scorer`
gives. `score` takes the table of scorers as an argument, so a caller adds its own scorers to these.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from eval_vocabulary.episode import STATIC_FILE, SUCCESS, TASK, TERMINATED
from platform_client.evals import MOLMO_SCORER, ScorerRef
from pydantic import BaseModel

log = logging.getLogger(__name__)

SCORES_FILENAME = 'scores.json'


@dataclass(frozen=True)
class Outcome:
    """One episode's task, its success, and its graded score where its scorer grades one."""

    task: str
    succeeded: bool
    graded: float | None = None

    @property
    def score(self) -> float:
        """What the episode adds to the primary score of the run. An ungraded episode scores its success."""
        return self.graded if self.graded is not None else float(self.succeeded)


# Gives one recorded episode its outcome, or None when the episode recorded no outcome.
Scorer = Callable[[Path], Outcome | None]


class TaskScore(BaseModel):
    """The trials of one task, with the count beside the share.

    `mean_subtask_score` is None when the scorer grades no episode of the task.
    """

    trials: int
    successes: int
    success_rate: float
    mean_subtask_score: float | None = None


class Scores(BaseModel):
    """The `scores.json` of a run.

    A board ranks on `primary`: the mean episode score. `unscored` counts the episodes that recorded no
    outcome. The rates do not include them, and do not count them as failures. `responses.Scores` publishes
    a subset of this record.
    """

    primary: float
    success_rate: float
    per_task: dict[str, TaskScore]
    episodes: int
    unscored: int


def read_static(episode: Path) -> dict[str, object] | None:
    """The statics that `episode` recorded, or None when it recorded none or they do not parse."""
    static = episode / STATIC_FILE
    if not static.is_file():
        return None
    try:
        loaded = json.loads(static.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        loaded = None
    if not isinstance(loaded, dict):
        # One damaged upload leaves the other episodes scorable, so this is logged, not raised.
        log.error('%s does not hold a JSON object; scoring this episode as unscored', static)
        return None
    return loaded


def recorded_task_and_success(static: Mapping[str, object], task_key: str) -> tuple[str, bool] | None:
    """The task under `task_key` and the recorded success, or None when either is missing or of the wrong type.

    A trial that ran out of time failed, as a real-robot episode that runs out of time does.
    """
    terminated = static.get(TERMINATED)
    if TERMINATED in static and not isinstance(terminated, bool):
        log.error('%s=%r is not a bool; reading it as absent', TERMINATED, terminated)
    ran_out_of_time = terminated is False
    success = False if ran_out_of_time else static.get(SUCCESS)
    task = static.get(task_key)
    checked = ((task_key, str),) if ran_out_of_time else ((SUCCESS, bool), (task_key, str))
    misrecorded = [
        f'{key}={static[key]!r} is not a {kind.__name__}'
        for key, kind in checked
        if key in static and not isinstance(static[key], kind)
    ]
    if misrecorded:
        # One damaged record leaves the other episodes scorable, so this is logged, not raised.
        log.error('%s; scoring this episode as unscored', '; '.join(misrecorded))
    if not isinstance(success, bool) or not isinstance(task, str):
        return None
    return task, success


def molmo_outcome(episode: Path) -> Outcome | None:
    """MolmoSpaces' scorer: an episode succeeds or fails, and a run ranks on its success rate."""
    static = read_static(episode)
    recorded = recorded_task_and_success(static, TASK) if static is not None else None
    return Outcome(*recorded) if recorded is not None else None


PUBLIC_SCORERS: Mapping[ScorerRef, Scorer] = MappingProxyType({MOLMO_SCORER: molmo_outcome})


def score(scorer: ScorerRef, episodes: Iterable[Path], scorers: Mapping[ScorerRef, Scorer] = PUBLIC_SCORERS) -> Scores:
    """The scores of `episodes` under `scorer`, or LookupError when `scorers` does not hold it."""
    outcome_of = scorers.get(scorer)
    if outcome_of is None:
        raise LookupError(f'no scorer {scorer!r}; the scorers are {", ".join(sorted(scorers))}')
    return tally(map(outcome_of, episodes))


def tally(outcomes: Iterable[Outcome | None]) -> Scores:
    """The scores of a run whose episodes gave `outcomes`. A None outcome counts as unscored."""
    trials: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    scored_sum: defaultdict[str, float] = defaultdict(float)
    graded: set[str] = set()
    total = unscored = 0
    for outcome in outcomes:
        total += 1
        if outcome is None:
            unscored += 1
            continue
        trials[outcome.task] += 1
        successes[outcome.task] += outcome.succeeded
        scored_sum[outcome.task] += outcome.score
        if outcome.graded is not None:
            graded.add(outcome.task)
    scored = total - unscored
    return Scores(
        primary=sum(scored_sum.values()) / scored if scored else 0.0,
        success_rate=successes.total() / scored if scored else 0.0,
        per_task={
            task: TaskScore(
                trials=trials[task],
                successes=successes[task],
                success_rate=successes[task] / trials[task],
                mean_subtask_score=scored_sum[task] / trials[task] if task in graded else None,
            )
            for task in sorted(trials)
        },
        episodes=total,
        unscored=unscored,
    )
