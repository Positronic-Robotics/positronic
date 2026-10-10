"""A run's `scores.json`, from the outcome that each recorded episode reports.

A scorer gives one episode its `Outcome`, and `tally` adds the outcomes of a run into `Scores`.
`PUBLIC_SCORERS` holds the scorers of the public evals, under the names that `EvalDefinition.scorer`
gives. `score` takes the table of scorers as an argument, so a caller adds its own scorers to these.
"""

from __future__ import annotations

import json
import logging
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


class Scores(BaseModel):
    """The `scores.json` of a run.

    A board ranks on `primary`: the mean score of the episodes that recorded an outcome. `episodes` counts every
    episode, and `unscored` counts the ones that recorded no outcome. `primary` does not include those, and does not
    count them as failures. `responses.Scores` publishes a subset of this record.
    """

    primary: float
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

    A trial that ran out of time failed, as a real-robot episode that runs out of time does. Its recorded
    success does not count, and a success of the wrong type is still logged.
    """
    terminated = static.get(TERMINATED)
    if TERMINATED in static and not isinstance(terminated, bool):
        log.error('%s=%r is not a bool; reading it as absent', TERMINATED, terminated)
    success = False if terminated is False else static.get(SUCCESS)
    task = static.get(task_key)
    recorded = (task, success) if isinstance(task, str) and isinstance(success, bool) else None
    misrecorded = [
        f'{key}={static[key]!r} is not a {kind.__name__}'
        for key, kind in ((SUCCESS, bool), (task_key, str))
        if key in static and not isinstance(static[key], kind)
    ]
    if misrecorded:
        # One damaged record leaves the other episodes scorable, so this is logged, not raised.
        verdict = 'unscored' if recorded is None else 'failed, because it ran out of time'
        log.error('%s; scoring this episode as %s', '; '.join(misrecorded), verdict)
    return recorded


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
    recorded = list(outcomes)
    scored = [outcome.score for outcome in recorded if outcome is not None]
    return Scores(
        primary=sum(scored) / len(scored) if scored else 0.0,
        episodes=len(recorded),
        unscored=len(recorded) - len(scored),
    )
