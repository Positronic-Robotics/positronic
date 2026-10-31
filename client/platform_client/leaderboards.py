"""What the `leaderboards.*` routes answer: each leaderboard, its rows, and the parts that a page draws from it.

The descriptor of a leaderboard names its metrics and its outcome groups, and the other parts refer to them by key.
A field that a later leaderboard can give a new value is an open string, so this client reads that leaderboard too.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Annotated, Self

from platform_client.ids import LeaderboardVersionId, SubmissionId
from platform_client.responses import DataBackedReplayLink
from positronic_evals.evals import EvalRef
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    NonNegativeFloat,
    NonNegativeInt,
    model_validator,
)
from pydantic_core import core_schema

# A definition holds this placeholder in the place of the time limit, in seconds.
TIME_LIMIT_PLACEHOLDER = '{seconds}'


class _Wire(BaseModel):
    """A model of this module. JSON carries no NaN and no infinity, so a float field refuses them."""

    model_config = ConfigDict(allow_inf_nan=False)


# How far two computed floats can differ and still be one value, as a sum of shares or a stack of edges.
_FLOAT_SLACK = 1e-9


def _refuse_repeats(keys: Iterable[object], what: str) -> None:
    repeated = [key for key, count in Counter(keys).items() if count > 1]
    if repeated:
        raise ValueError(f'two {what} share a key: {repeated}')


def _refuse_other_lengths(vectors: dict[str, Sequence[object]], length: int, what: str) -> None:
    for name, vector in vectors.items():
        if len(vector) != length:
            raise ValueError(f'{what}: {name} holds {len(vector)} values, and the axis holds {length}')


class _Name(str):
    """A name the platform gives, as a validated `str`: not empty, and with no whitespace."""

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        if not value or any(c.isspace() for c in value):
            raise ValueError(f'not a {cls.__name__}: {value!r}')
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class LeaderboardRef(_Name):
    """The slug a leaderboard is read by."""

    __slots__ = ()


class LeaderboardTag(_Name):
    """A tag that groups leaderboards, such as the site that shows them."""

    __slots__ = ()


class ModelFamilyRef(_Name):
    """The id of a model family that `leaderboards.models` lists."""

    __slots__ = ()


class CheckpointRef(_Name):
    """The id of one checkpoint of a model family."""

    __slots__ = ()


class MetricKey(_Name):
    """The key of a metric in the descriptor of its leaderboard, as its `Metric.key`."""

    __slots__ = ()


class OutcomeKey(_Name):
    """The key of an outcome group in the descriptor of its leaderboard, as its `OutcomeGroup.key`."""

    __slots__ = ()


# The metric key of a leaderboard that ranks one score of each row, such as the primary score of an eval.
PRIMARY_METRIC = MetricKey('primary')

# A share, from 0 to 1.
Fraction = Annotated[float, Field(ge=0, le=1)]
# A share, from 0 to 100.
Percent = Annotated[float, Field(ge=0, le=100)]


class UserTag(_Name):
    """The tag of a user. It is the same on every leaderboard, so a user finds their own row by it."""

    __slots__ = ()


class RowKey(_Name):
    """The key of a row on its leaderboard. It is the same in every version, and the other parts name a row by it."""

    __slots__ = ()


class LeaderboardTaskId(_Name):
    """The id of a task that `LeaderboardResponse.tasks` lists, as its `LeaderboardTask.id`."""

    __slots__ = ()


class EpisodeId(_Name):
    """The id of a recorded episode. It is the same in every version of every leaderboard that lists the episode."""

    __slots__ = ()


class Estimate(_Wire):
    """A value and its range. A value with no range has a `low` and a `high` equal to it."""

    value: float
    low: float
    high: float

    @model_validator(mode='after')
    def _the_range_is_in_order(self) -> Self:
        if self.low > self.high:
            raise ValueError(f'a range from {self.low} down to {self.high}')
        return self


class RankRange(_Wire):
    """The places a row can take, from `best` to `worst`. Rows that the leaderboard cannot tell apart share places."""

    best: int = Field(ge=1)
    worst: int = Field(ge=1)

    @model_validator(mode='after')
    def _the_best_place_comes_first(self) -> Self:
        if self.best > self.worst:
            raise ValueError(f'places from {self.best} down to {self.worst}')
        return self


class Link(_Wire):
    text: str
    url: str


class Metric(_Wire):
    """One number of a row, in `unit`.

    `key` names the metric in the `metrics` of a row and in the series of a chart. `definition` states what the
    metric counts, with `TIME_LIMIT_PLACEHOLDER` for the time limit. `row_definition` states what the value of a row
    counts, where that value is not the value at the default time limit. A chart axis of the metric ends at `axis_max`.
    """

    key: MetricKey
    name: str
    unit: str | None = None
    decimals: int = Field(ge=0)
    definition: str | None = None
    row_definition: str | None = None
    axis_max: float | None = None


class OutcomeGroup(_Wire):
    """One group that the result of an episode falls in.

    `key` names the group in the outcome shares of a row, in an episode and in a progress band. `step_score` is the
    score of one step in the group, where the leaderboard scores steps.
    """

    key: OutcomeKey
    name: str
    step_score: Fraction | None = None


class TimeLimit(_Wire):
    """The time limits of a leaderboard, in seconds. A page opens at `default_s`, and `axis` names the time axis."""

    default_s: int = Field(ge=1)
    max_s: int = Field(ge=1)
    axis: str | None = None

    @model_validator(mode='after')
    def _the_default_is_within_the_max(self) -> Self:
        if self.default_s > self.max_s:
            raise ValueError(f'a default time limit of {self.default_s} s past the maximum of {self.max_s} s')
        return self


class LeaderboardDescriptor(_Wire):
    """The facts of a leaderboard, and the words that its pages use.

    A simulated leaderboard ranks `eval`. The rows rank on the metric that `ranked_by` names.
    `outcomes` lists the outcome groups in the order a page offers them. A `frozen` leaderboard takes no new episodes.
    """

    eval: EvalRef | None = None
    short_name: str | None = None
    robot: str | None = None
    release: str | None = None
    summary: str | None = None
    setup: Link | None = None
    task_noun: str = 'tasks'
    episode_noun: str = 'episodes'
    offers_task_charts: bool = True
    # What the test that tells two rows apart compares, as a verdict names it.
    compared_by: str | None = None
    metrics: list[Metric] = Field(min_length=1)
    ranked_by: MetricKey
    outcomes: list[OutcomeGroup] = Field(default_factory=list)
    time_limit: TimeLimit | None = None
    frozen: bool = False

    @model_validator(mode='after')
    def _no_two_metrics_share_a_key(self) -> Self:
        _refuse_repeats((metric.key for metric in self.metrics), 'metrics')
        return self

    @model_validator(mode='after')
    def _no_two_outcome_groups_share_a_key(self) -> Self:
        _refuse_repeats((outcome.key for outcome in self.outcomes), 'outcome groups')
        return self

    @model_validator(mode='after')
    def _the_rows_rank_on_one_of_the_metrics(self) -> Self:
        keys = [metric.key for metric in self.metrics]
        if self.ranked_by not in keys:
            raise ValueError(f'ranked_by names {self.ranked_by!r}, which is not one of the metrics {keys}')
        return self


class Caveat(_Wire):
    """A note that a reader needs to read the numbers correctly. `text` is for a human."""

    kind: str = Field(min_length=1)
    text: str


class TagPosition(_Wire):
    """The place of a leaderboard in one tag."""

    tag: LeaderboardTag
    position: int = Field(ge=0)


class LeaderboardSummary(_Wire):
    """One leaderboard of `leaderboards.list`. The other routes take `leaderboard` as the slug.

    `method` names the method that computes the rows from the episodes, such as the best submission of each team.
    """

    leaderboard: LeaderboardRef
    title: str
    method: str = Field(min_length=1)
    tags: list[TagPosition] = Field(default_factory=list)
    descriptor: LeaderboardDescriptor
    caveats: list[Caveat] = Field(default_factory=list)

    @model_validator(mode='after')
    def _one_place_in_each_tag(self) -> Self:
        _refuse_repeats((place.tag for place in self.tags), 'tag places')
        return self


class LeaderboardListResponse(_Wire):
    """`leaderboards.list` — the leaderboards of one tag, in the order of their positions in it."""

    tag: LeaderboardTag
    leaderboards: list[LeaderboardSummary] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_leaderboards_share_a_slug(self) -> Self:
        _refuse_repeats((summary.leaderboard for summary in self.leaderboards), 'leaderboards')
        return self

    @model_validator(mode='after')
    def _each_leaderboard_has_a_place_in_the_tag(self) -> Self:
        for summary in self.leaderboards:
            if all(place.tag != self.tag for place in summary.tags):
                raise ValueError(f'leaderboard {summary.leaderboard!r} has no place in tag {self.tag!r}')
        return self

    @model_validator(mode='after')
    def _the_leaderboards_follow_their_places_in_the_tag(self) -> Self:
        places = [place.position for summary in self.leaderboards for place in summary.tags if place.tag == self.tag]
        if any(later <= earlier for earlier, later in zip(places, places[1:], strict=False)):
            raise ValueError(f'the leaderboards of tag {self.tag!r} are not in the order of their places: {places}')
        return self


# rules-allow: earn-its-place — a reference row can name no publisher, and the rankings `BaselineMark` needs one
class Baseline(_Wire):
    """The mark of a reference row. `publisher` names the organization that released its model, where one did."""

    publisher: str | None = Field(default=None, min_length=1)


class TeamSubmission(_Wire):
    """The submission that the numbers of a team row come from. `replay` is absent until its page is built."""

    id: SubmissionId
    submitted_at: AwareDatetime
    replay: DataBackedReplayLink | None = None


class LeaderboardRow(_Wire):
    """One row of a leaderboard: a model family, or a team.

    `key` identifies the row. The platform gives each team a stable key that no other team shares. The row shows
    `display_name`, and two rows can show the same name. A team row stands for one user, and `user_tag` holds the
    tag of that user.

    A row that measures a model names its family, and the checkpoints whose episodes it counts. `submission` is
    absent on a row whose numbers come from more than one submission, such as a model row.

    `metrics` holds a value for each metric that the row has a value for. `episodes` counts the episodes that its
    numbers read, where the leaderboard counts them. `outcome_shares` gives the share of those episodes in each outcome
    group, by the key of the group, and the shares are not rounded. A `baseline` row is a reference on this
    leaderboard and has no `rank`, whatever the `baseline` of its model family is.
    """

    key: RowKey
    display_name: str
    user_tag: UserTag | None = None
    model: ModelFamilyRef | None = None
    checkpoints: list[CheckpointRef] = Field(default_factory=list)
    rank: RankRange | None
    baseline: Baseline | None = None
    metrics: dict[MetricKey, Estimate] = Field(default_factory=dict)
    episodes: int | None = Field(default=None, ge=0)
    outcome_shares: dict[OutcomeKey, Fraction] = Field(default_factory=dict)
    submission: TeamSubmission | None = None
    caveats: list[Caveat] = Field(default_factory=list)

    @model_validator(mode='after')
    def _ranked_exactly_when_not_a_baseline(self) -> Self:
        if (self.rank is None) != (self.baseline is not None):
            raise ValueError('a row has a rank when it is not a baseline, and only then')
        return self

    @model_validator(mode='after')
    def _each_checkpoint_once(self) -> Self:
        _refuse_repeats(self.checkpoints, 'checkpoints')
        return self

    @model_validator(mode='after')
    def _the_outcome_shares_add_up_to_at_most_one(self) -> Self:
        if sum(self.outcome_shares.values()) > 1 + _FLOAT_SLACK:
            raise ValueError(f'outcome shares that add up to more than 1: {self.outcome_shares}')
        return self


class PairVerdict(_Wire):
    """Whether the leaderboard tells two rows apart."""

    rows: tuple[RowKey, RowKey]
    separated: bool

    @model_validator(mode='after')
    def _a_pair_is_two_rows(self) -> Self:
        if self.rows[0] == self.rows[1]:
            raise ValueError(f'a pair of row {self.rows[0]!r} with itself')
        return self


class LeaderboardTask(_Wire):
    """One task that the episodes of a leaderboard ran. The numbers leave out a task that is not `counted`."""

    id: LeaderboardTaskId
    label: str
    counted: bool = True
    difficulty: str | None = None


class LeaderboardCounts(_Wire):
    """What the episode list holds for the rows that are not baselines: the tasks they ran, and their episodes."""

    tasks: int = Field(ge=0)
    episodes: int = Field(ge=0)


class EpisodeMedia(_Wire):
    """Where the recordings of one episode play back.

    A phone plays `video`, the 3D replay reads its data from `replay`, and `cameras` holds the video of each camera
    by its name. The links resolve when the route answers, and a link can stop working later, so keep
    the numbers of a version and read its links again.
    """

    video: str | None = None
    replay: str | None = None
    cameras: dict[str, str] = Field(default_factory=dict)


class Highlight(_Wire):
    """A clip of one episode that the top of a page plays, from `start_s` to `end_s`, or to the end of the episode."""

    episode: EpisodeId
    row: RowKey
    task: LeaderboardTaskId
    caption: str
    start_s: float = Field(default=0.0, ge=0)
    end_s: float | None = None
    media: EpisodeMedia | None = None

    @model_validator(mode='after')
    def _the_clip_ends_after_it_starts(self) -> Self:
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError(f'a clip that ends at {self.end_s} s, not after its start at {self.start_s} s')
        return self


class LeaderboardResponse(LeaderboardSummary):
    """`leaderboards.get` — one leaderboard and its rows, in the order that the leaderboard shows them.

    The routes of its parts take `version`. `version` is absent on a leaderboard that has no versions, and such a
    leaderboard has no parts. The platform computed the numbers at `computed_at`.
    """

    version: LeaderboardVersionId | None = None
    computed_at: AwareDatetime
    rows: list[LeaderboardRow] = Field(default_factory=list)
    pairs: list[PairVerdict] = Field(default_factory=list)
    tasks: list[LeaderboardTask] = Field(default_factory=list)
    counts: LeaderboardCounts | None = None
    highlights: list[Highlight] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_rows_share_a_key(self) -> Self:
        _refuse_repeats((row.key for row in self.rows), 'rows')
        return self

    @model_validator(mode='after')
    def _no_two_tasks_share_an_id(self) -> Self:
        _refuse_repeats((task.id for task in self.tasks), 'tasks')
        return self

    @model_validator(mode='after')
    def _a_row_names_only_metrics_of_the_descriptor(self) -> Self:
        metrics = {metric.key for metric in self.descriptor.metrics}
        for row in self.rows:
            if not set(row.metrics) <= metrics:
                raise ValueError(f'row {row.key!r} names a metric that the descriptor has not: {sorted(row.metrics)}')
        return self

    @model_validator(mode='after')
    def _a_row_names_only_outcome_groups_of_the_descriptor(self) -> Self:
        outcomes = {outcome.key for outcome in self.descriptor.outcomes}
        for row in self.rows:
            if not set(row.outcome_shares) <= outcomes:
                raise ValueError(f'row {row.key!r} names an outcome group that the descriptor has not')
        return self

    @model_validator(mode='after')
    def _the_pairs_name_rows_of_the_leaderboard(self) -> Self:
        rows = {row.key for row in self.rows}
        for pair in self.pairs:
            if not set(pair.rows) <= rows:
                raise ValueError(f'a pair names a row that the leaderboard has not: {pair.rows}')
        return self

    @model_validator(mode='after')
    def _each_highlight_names_a_row_of_the_leaderboard(self) -> Self:
        rows = {row.key for row in self.rows}
        for highlight in self.highlights:
            if highlight.row not in rows:
                raise ValueError(f'highlight {highlight.episode!r} names a row that the leaderboard has not')
        return self

    @model_validator(mode='after')
    def _each_highlight_names_a_task_of_the_leaderboard(self) -> Self:
        tasks = {task.id for task in self.tasks}
        for highlight in self.highlights:
            if highlight.task not in tasks:
                raise ValueError(f'highlight {highlight.episode!r} names a task that the leaderboard has not')
        return self

    @model_validator(mode='after')
    def _no_place_is_past_the_ranked_rows(self) -> Self:
        ranked = sum(row.rank is not None for row in self.rows)
        for row in self.rows:
            if row.rank is not None and row.rank.worst > ranked:
                raise ValueError(f'row {row.key!r} takes place {row.rank.worst} of {ranked} ranked rows')
        return self

    @model_validator(mode='after')
    def _the_counts_hold_no_more_tasks_than_the_list(self) -> Self:
        if self.counts is not None and self.counts.tasks > len(self.tasks):
            raise ValueError(f'counts name {self.counts.tasks} tasks, and the list holds {len(self.tasks)}')
        return self


class _LeaderboardPart(_Wire):
    """The leaderboard version that a part comes from."""

    leaderboard: LeaderboardRef
    version: LeaderboardVersionId


class EstimateSeries(_Wire):
    """One metric at each time limit of a chart, with its range. A value is None at a limit where it has none.

    Each value is rounded to the decimals of the metric. `margin` is half the range, rounded from the unrounded ends.
    """

    value: list[float | None]
    low: list[float | None]
    high: list[float | None]
    margin: list[NonNegativeFloat | None]

    @model_validator(mode='after')
    def _the_parts_of_the_series_have_one_length(self) -> Self:
        _refuse_other_lengths({'low': self.low, 'high': self.high, 'margin': self.margin}, len(self.value), 'a series')
        return self

    @model_validator(mode='after')
    def _each_limit_has_a_whole_estimate_or_none(self) -> Self:
        for at in zip(self.value, self.low, self.high, self.margin, strict=True):
            if None in at and at != (None, None, None, None):
                raise ValueError(f'a limit with part of an estimate: {at}')
        return self

    @model_validator(mode='after')
    def _each_range_is_in_order(self) -> Self:
        for low, high in zip(self.low, self.high, strict=True):
            if low is not None and high is not None and low > high:
                raise ValueError(f'a range from {low} down to {high}')
        return self


class RowCharts(_Wire):
    """The curves of one row.

    `series` holds the series of each metric by its key. `density` gives how fast the row gains on what the
    leaderboard counts at each time limit, per second. `rank` holds the places of the row at each time limit.
    """

    episodes: int = Field(ge=0)
    series: dict[MetricKey, EstimateSeries] = Field(default_factory=dict)
    density: list[float]
    rank: list[RankRange | None]


class LeaderboardChartsResponse(_LeaderboardPart):
    """`leaderboards.charts` — the charts over one task, or over every task where `task` is None.

    Each series has a value at each of the `windows`, which are time limits in seconds. The chart compares the rows
    in `pairs`.
    """

    task: LeaderboardTaskId | None = None
    windows: list[NonNegativeFloat]
    rows: dict[RowKey, RowCharts] = Field(default_factory=dict)
    pairs: list[PairVerdict] = Field(default_factory=list)

    @model_validator(mode='after')
    def _the_windows_rise(self) -> Self:
        if any(later <= earlier for earlier, later in zip(self.windows, self.windows[1:], strict=False)):
            raise ValueError(f'windows that do not rise: {self.windows}')
        return self

    @model_validator(mode='after')
    def _each_row_has_a_value_at_each_window(self) -> Self:
        for key, charts in self.rows.items():
            vectors: dict[str, Sequence[object]] = {'density': charts.density, 'rank': charts.rank}
            vectors.update({f'series {metric}': series.value for metric, series in charts.series.items()})
            _refuse_other_lengths(vectors, len(self.windows), f'row {key!r}')
        return self

    @model_validator(mode='after')
    def _the_pairs_name_rows_of_the_chart(self) -> Self:
        for pair in self.pairs:
            if not set(pair.rows) <= set(self.rows):
                raise ValueError(f'a pair names a row that the chart has not: {pair.rows}')
        return self


class EpisodeColumn(_Wire):
    """One column of the episode list: its name, the text that explains it, and the decimals of its values."""

    name: str
    description: str | None = None
    decimals: int | None = Field(default=None, ge=0)


class LeaderboardEpisode(_Wire):
    """One episode of the episode list.

    `outcome` names the outcome group of the episode by its key, and the list shows `result` for it. `steps` and
    `steps_done` count the steps of the task and the steps that the episode did. `furthest_stage` names how far the
    episode got in the step that it stopped in. The episode did `credit` percent of the task, and the pace column
    shows `pace`. `seconds_to_success` and `duration_s` count from the first sample of the try, with no cap. `media`
    is absent where no recording of the episode plays back.
    """

    id: EpisodeId
    row: RowKey
    checkpoint: CheckpointRef | None = None
    task: LeaderboardTaskId
    outcome: OutcomeKey
    result: str
    steps: int | None = Field(default=None, ge=1)
    steps_done: int | None = Field(default=None, ge=0)
    furthest_stage: str | None = None
    credit: Percent | None = None
    pace: float | None = Field(default=None, ge=0)
    seconds_to_success: float | None = Field(default=None, ge=0)
    duration_s: float | None = Field(default=None, ge=0)
    recorded_at: AwareDatetime | None = None
    media: EpisodeMedia | None = None

    @model_validator(mode='after')
    def _the_success_comes_within_the_episode(self) -> Self:
        if self.seconds_to_success is not None and self.duration_s is not None:
            if self.seconds_to_success > self.duration_s:
                raise ValueError(f'a success at {self.seconds_to_success} s in an episode of {self.duration_s} s')
        return self


class LeaderboardEpisodesResponse(_LeaderboardPart):
    """`leaderboards.episodes` — every episode that the leaderboard lists, in the order a page lists them.

    A list that shows no credit column or no difficulty column has None in its place.
    """

    result: EpisodeColumn
    pace: EpisodeColumn
    credit: EpisodeColumn | None = None
    difficulty: EpisodeColumn | None = None
    episodes: list[LeaderboardEpisode] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_episodes_share_an_id(self) -> Self:
        _refuse_repeats((episode.id for episode in self.episodes), 'episodes')
        return self


class ProgressBand(_Wire):
    """The count of episodes in one band at each second, before smoothing.

    A band holds the episodes of one outcome group, by the key of the group, or the episodes that did one count of
    steps. Exactly one of `outcome` and `steps_done` is set.
    """

    outcome: OutcomeKey | None = None
    steps_done: int | None = Field(default=None, ge=0)
    episodes: list[NonNegativeInt]

    @model_validator(mode='after')
    def _the_band_names_exactly_one_key(self) -> Self:
        if (self.outcome is None) == (self.steps_done is None):
            raise ValueError('a band names exactly one of an outcome group and a count of steps')
        return self


class ProgressPanel(_Wire):
    """How far the episodes of a row on one task had got at each second, or on every task where `task` is None.

    `edges` holds the smoothed count of episodes below each band, then of all episodes: band `i` runs from edge `i`
    to edge `i + 1`. `mean_credit` is the smoothed mean percent of the task done. Where the bands count steps,
    `mean_steps_done` is the mean count at each second and `mean_steps_done_at_end` is the mean at the episode end.
    """

    task: LeaderboardTaskId | None = None
    episodes: int = Field(ge=1)
    bands: list[ProgressBand] = Field(min_length=1)
    edges: list[list[NonNegativeFloat]]
    mean_credit: list[Percent] = Field(min_length=1)
    mean_steps_done: list[NonNegativeFloat] | None = None
    mean_steps_done_at_end: NonNegativeFloat | None = None

    @property
    def seconds(self) -> int:
        """The count of seconds that each series of the panel covers."""
        return len(self.mean_credit)

    @model_validator(mode='after')
    def _the_bands_name_one_kind_of_key(self) -> Self:
        if len({band.outcome is None for band in self.bands}) > 1:
            raise ValueError('a panel with bands of outcome groups and bands of step counts')
        return self

    @model_validator(mode='after')
    def _no_two_bands_share_a_key(self) -> Self:
        _refuse_repeats(((band.outcome, band.steps_done) for band in self.bands), 'bands')
        return self

    @model_validator(mode='after')
    def _one_more_edge_than_bands(self) -> Self:
        if len(self.edges) != len(self.bands) + 1:
            raise ValueError(f'{len(self.edges)} edges around {len(self.bands)} bands')
        return self

    @model_validator(mode='after')
    def _each_series_has_a_value_at_each_second(self) -> Self:
        vectors: dict[str, Sequence[object]] = {f'band {i}': band.episodes for i, band in enumerate(self.bands)}
        vectors.update({f'edge {i}': edge for i, edge in enumerate(self.edges)})
        if self.mean_steps_done is not None:
            vectors['mean_steps_done'] = self.mean_steps_done
        _refuse_other_lengths(vectors, self.seconds, 'a panel')
        return self

    @model_validator(mode='after')
    def _the_bands_hold_each_episode_once_at_each_second(self) -> Self:
        for second in range(self.seconds):
            held = sum(band.episodes[second] for band in self.bands)
            if held != self.episodes:
                raise ValueError(f'at second {second}, the bands hold {held} of {self.episodes} episodes')
        return self

    @model_validator(mode='after')
    def _the_top_edge_counts_every_episode(self) -> Self:
        for second, top in enumerate(self.edges[-1]):
            if abs(top - self.episodes) > _FLOAT_SLACK:
                raise ValueError(f'at second {second}, the top edge is {top} of {self.episodes} episodes')
        return self

    @model_validator(mode='after')
    def _the_stack_starts_at_zero(self) -> Self:
        for second, bottom in enumerate(self.edges[0]):
            if abs(bottom) > _FLOAT_SLACK:
                raise ValueError(f'at second {second}, a stack that starts at {bottom}')
        return self

    @model_validator(mode='after')
    def _each_edge_is_at_or_above_the_edge_under_it(self) -> Self:
        for second in range(self.seconds):
            stack = [edge[second] for edge in self.edges]
            if any(upper < lower - _FLOAT_SLACK for lower, upper in zip(stack, stack[1:], strict=False)):
                raise ValueError(f'at second {second}, an edge below the edge under it: {stack}')
        return self


class RowProgress(_Wire):
    """The progress panels of one row. All panels share one time axis, from 0 to `horizon_s`."""

    horizon_s: int = Field(ge=0)
    panels: list[ProgressPanel] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_panels_share_a_task(self) -> Self:
        _refuse_repeats((panel.task for panel in self.panels), 'panels')
        return self

    @model_validator(mode='after')
    def _no_panel_runs_past_the_horizon(self) -> Self:
        for panel in self.panels:
            if panel.seconds > self.horizon_s + 1:
                raise ValueError(f'a panel of {panel.seconds} seconds past a horizon at {self.horizon_s} s')
        return self


class LeaderboardProgressResponse(_LeaderboardPart):
    """`leaderboards.progress` — the progress panels of each row that has episodes."""

    rows: dict[RowKey, RowProgress] = Field(default_factory=dict)


class CardFact(_Wire):
    """One fact of a model card. `source` is the URL or path that the text is from, and a page links `href`.

    `checkpoint` names the checkpoint of the family that the fact is about. It is absent on a fact about the family.
    """

    text: str
    source: str
    href: str | None = None
    checkpoint: CheckpointRef | None = None


class CardLine(_Wire):
    """One line of a model card. `field` names the line.

    `facts` holds one fact about the family, or facts that each name a checkpoint of the family.
    """

    field: str = Field(min_length=1)
    facts: list[CardFact] = Field(min_length=1)

    @model_validator(mode='after')
    def _a_fact_about_the_family_stands_alone(self) -> Self:
        if len(self.facts) > 1 and any(fact.checkpoint is None for fact in self.facts):
            raise ValueError(f'line {self.field!r} holds a fact about the family beside other facts')
        return self

    @model_validator(mode='after')
    def _each_checkpoint_has_one_fact(self) -> Self:
        named = (fact.checkpoint for fact in self.facts if fact.checkpoint is not None)
        _refuse_repeats(named, f'facts of line {self.field!r}')
        return self


class RunStep(_Wire):
    """One command that a reader runs, under the `label` that names the step, with the `source` it is from."""

    label: str
    command: str
    source: str


class RunCommands(_Wire):
    """The commands that run a checkpoint, in the order a reader runs them.

    `checkpoint` names the checkpoint that the commands run. It is absent only on a family with one checkpoint.
    """

    steps: list[RunStep] = Field(min_length=1)
    checkpoint: CheckpointRef | None = None


class ModelCard(_Wire):
    """A card gives the commands that run the model, or a guide to them, or neither."""

    lines: list[CardLine] = Field(default_factory=list)
    run_commands: RunCommands | None = None
    run_guide: CardFact | None = None

    @model_validator(mode='after')
    def _at_most_one_way_to_run_the_model(self) -> Self:
        if self.run_commands is not None and self.run_guide is not None:
            raise ValueError('a card gives at most one of the commands and a guide to them')
        return self

    @model_validator(mode='after')
    def _no_two_lines_share_a_field(self) -> Self:
        _refuse_repeats((line.field for line in self.lines), 'card lines')
        return self


class ModelCheckpoint(_Wire):
    id: CheckpointRef
    display_name: str


class ModelFamily(_Wire):
    """One model family of `leaderboards.models`.

    A page shows `short_name` where the display name does not fit. A `baseline` family is a reference on every
    leaderboard, such as a person or teleoperation. On each leaderboard, the `baseline` of a row decides whether that
    row is a reference. A family that stands for a person or for teleoperation lists no checkpoint.
    `former_display_names` holds the names that the family had before.
    """

    id: ModelFamilyRef
    display_name: str
    short_name: str
    baseline: bool = False
    categories: list[str] = Field(default_factory=list)
    checkpoints: list[ModelCheckpoint] = Field(default_factory=list)
    card: ModelCard | None = None
    former_display_names: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_checkpoints_share_an_id(self) -> Self:
        _refuse_repeats((checkpoint.id for checkpoint in self.checkpoints), 'checkpoints')
        return self

    @model_validator(mode='after')
    def _the_commands_have_a_checkpoint_to_run(self) -> Self:
        commands = self.card.run_commands if self.card is not None else None
        if commands is not None and not self.checkpoints:
            raise ValueError(f'commands for family {self.id!r}, which lists no checkpoint to run')
        return self

    @model_validator(mode='after')
    def _the_commands_run_a_checkpoint_of_the_family(self) -> Self:
        commands = self.card.run_commands if self.card is not None else None
        if commands is not None and commands.checkpoint is not None:
            if commands.checkpoint not in {checkpoint.id for checkpoint in self.checkpoints}:
                raise ValueError(f'the commands of family {self.id!r} run another checkpoint: {commands.checkpoint!r}')
        return self

    @model_validator(mode='after')
    def _the_commands_name_their_checkpoint_in_a_family_of_several(self) -> Self:
        commands = self.card.run_commands if self.card is not None else None
        if commands is not None and commands.checkpoint is None and len(self.checkpoints) > 1:
            raise ValueError(f'the commands of family {self.id!r} name none of its {len(self.checkpoints)} checkpoints')
        return self

    @model_validator(mode='after')
    def _each_card_fact_names_a_checkpoint_of_the_family(self) -> Self:
        if self.card is None:
            return self
        facts = [fact for line in self.card.lines for fact in line.facts]
        if self.card.run_guide is not None:
            facts.append(self.card.run_guide)
        known = {checkpoint.id for checkpoint in self.checkpoints}
        for fact in facts:
            if fact.checkpoint is not None and fact.checkpoint not in known:
                raise ValueError(f'a fact of family {self.id!r} names another checkpoint: {fact.checkpoint!r}')
        return self


class LeaderboardModelsResponse(_Wire):
    """`leaderboards.models` — the model families that a leaderboard row can name, in a stable order."""

    models: list[ModelFamily] = Field(default_factory=list)

    @model_validator(mode='after')
    def _no_two_families_share_an_id(self) -> Self:
        _refuse_repeats((family.id for family in self.models), 'model families')
        return self
