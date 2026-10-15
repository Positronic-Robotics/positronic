"""How an eval scores a run: its labelers in precedence order, and one metric over the labels they give.

`episode_score` gives what one label adds to a mean, from 0 to 1, and `score` gives a run's `Scores` from the
labels of its episodes. No file holds a score.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from positronic_evals.labels import Label, LabelerRef, Outcome, Stage
from positronic_evals.slug import Slugged

_SCORING_CONFIG = ConfigDict(extra='forbid', frozen=True)

# A share of the whole task, from 0 to 1.
Share = Annotated[float, Field(ge=0, le=1)]


class EpisodeScore(BaseModel):
    """What one label adds to a mean. Parameters only."""

    model_config = _SCORING_CONFIG

    success: Share = 1.0
    # The partial scores below apply when the episode did not succeed.
    stage: dict[Slugged[Stage], Share] = {}  # the score of the furthest stage
    use_subtask_score: bool = False  # the last subtask score
    use_item_share: bool = False  # delivered / total_items
    zero: frozenset[Slugged[Outcome]] = frozenset()  # outcomes that score 0, whatever progress the episode made


class MeanEpisodeScore(BaseModel):
    """The mean of a run's episode scores, or of its task means."""

    model_config = _SCORING_CONFIG

    kind: Literal['mean_episode_score'] = 'mean_episode_score'
    episode_score: EpisodeScore = EpisodeScore()
    time_limit_s: PositiveFloat | None = None
    per_task: bool = False  # True: the mean of the task means, so each task has the same weight


# A closed set: a new metric kind joins this union.
Metric = Annotated[MeanEpisodeScore, Field(discriminator='kind')]


class Scoring(BaseModel):
    """How an eval scores a run."""

    model_config = _SCORING_CONFIG

    labelers: tuple[LabelerRef, ...] = Field(min_length=1)  # in precedence order
    primary: Metric


class Scores(BaseModel):
    """The scores of a run.

    A board ranks on `primary`, the run's value under the eval's primary metric. `episodes` counts every episode,
    and `unscored` counts the episodes that no labeler labelled. `primary` does not include those, and does not
    count them as failures. `platform_client.responses.Scores` publishes a subset of this record.
    """

    primary: float
    episodes: int
    unscored: int


def episode_score(label: Label, rule: EpisodeScore, time_limit_s: float | None) -> float:
    """What `label` adds to a mean under `rule`, as the label stood at `time_limit_s`.

    An outcome in `rule.zero` scores 0, and a success scores `rule.success`. Any other label scores the largest
    partial score that it holds, or 0 when it holds none.
    """
    if time_limit_s is not None:
        label = label.until(time_limit_s)
    if label.outcome in rule.zero:
        return 0.0
    if label.outcome is Outcome.success:
        return rule.success
    partial = []
    if label.furthest_stage is not None:
        partial.append(rule.stage.get(label.furthest_stage, 0.0))
    if rule.use_subtask_score and label.subtask_score is not None:
        partial.append(label.subtask_score)
    if rule.use_item_share and label.total_items is not None:
        partial.append(label.delivered / label.total_items)
    return max(partial, default=0.0)


def score(scoring: Scoring, labels: Iterable[Label | None]) -> Scores:
    """The scores of a run whose episodes have `labels`. A None is an episode that no labeler labelled."""
    given = list(labels)
    metric = scoring.primary
    scored = [label for label in given if label is not None]
    values = [episode_score(label, metric.episode_score, metric.time_limit_s) for label in scored]
    if metric.per_task:
        by_task: dict[str, list[float]] = defaultdict(list)
        for label, value in zip(scored, values, strict=True):
            by_task[label.task].append(value)
        values = [fmean(task_values) for task_values in by_task.values()]
    return Scores(primary=fmean(values) if values else 0.0, episodes=len(given), unscored=len(given) - len(scored))
