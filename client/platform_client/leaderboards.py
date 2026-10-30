"""What the `leaderboards.*` routes answer: each leaderboard, its rows, and the parts that a page draws from it.

The descriptor of a leaderboard names its metrics and its outcome groups, and the other parts refer to them by key.
A field that a later leaderboard can give a new value is an open string, so this client reads that leaderboard too.
"""

from __future__ import annotations

from typing import Annotated, Self

from platform_client.ids import LeaderboardVersionId, SubmissionId
from platform_client.responses import DataBackedReplayLink
from positronic_evals.evals import EvalRef
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema

# What a definition holds in the place of the time limit, in seconds.
TIME_LIMIT_PLACEHOLDER = '{seconds}'


class _Wire(BaseModel):
    """A model of this module. JSON carries no NaN and no infinity, so a float field refuses them."""

    model_config = ConfigDict(allow_inf_nan=False)


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


class LeaderboardDescriptor(_Wire):
    """The facts of a leaderboard, and the words that its pages use.

    `eval` is the eval that a simulated leaderboard ranks. The rows rank on the metric that `ranked_by` names.
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
        keys = [metric.key for metric in self.metrics]
        if len(set(keys)) != len(keys):
            raise ValueError(f'two metrics share a key: {keys}')
        return self

    @model_validator(mode='after')
    def _no_two_outcome_groups_share_a_key(self) -> Self:
        keys = [outcome.key for outcome in self.outcomes]
        if len(set(keys)) != len(keys):
            raise ValueError(f'two outcome groups share a key: {keys}')
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
    """One leaderboard of `leaderboards.list`. `leaderboard` is the slug that the other routes take.

    `method` names the method that computes the rows from the episodes, such as the best submission of each team.
    """

    leaderboard: LeaderboardRef
    title: str
    method: str = Field(min_length=1)
    tags: list[TagPosition] = Field(default_factory=list)
    descriptor: LeaderboardDescriptor
    caveats: list[Caveat] = Field(default_factory=list)


class LeaderboardListResponse(_Wire):
    """`leaderboards.list` — the leaderboards of one tag, in the order of their positions in it."""

    tag: LeaderboardTag
    leaderboards: list[LeaderboardSummary] = Field(default_factory=list)


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

    `key` identifies the row. The platform gives each team a stable key that no other team shares. `display_name` is
    what the row shows, and two rows can show the same name. `tag` is the tag of the user that a team row stands for.
    It is the same on every leaderboard, so a user finds their own row by it.

    A row that measures a model names its family, and the checkpoints whose episodes it counts. `submission` is
    absent on a row whose numbers come from more than one submission, such as a model row.

    `metrics` holds a value for each metric that the row has a value for. `episodes` counts the episodes that its
    numbers read, where the leaderboard counts them. `outcome_shares` gives the share of those episodes in each outcome
    group, by the key of the group. A `baseline` row is a reference on this leaderboard and
    has no `rank`, whatever the `baseline` of its model family is.
    """

    key: RowKey
    display_name: str
    tag: str | None = None
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


class PairVerdict(_Wire):
    """Whether the leaderboard tells two rows apart."""

    rows: tuple[RowKey, RowKey]
    separated: bool


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

    `video` is the video that a phone plays, `replay` is the data of the 3D replay, and `cameras` holds the video of
    each camera by its name. The links resolve when the route answers, and a link can stop working later, so keep
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


class LeaderboardResponse(LeaderboardSummary):
    """`leaderboards.get` — one leaderboard and its rows, in the order that the leaderboard shows them.

    The routes of its parts take `version`. `version` is absent on a leaderboard that has no versions, and such a
    leaderboard has no parts. `computed_at` is when the numbers were computed.
    """

    version: LeaderboardVersionId | None = None
    computed_at: AwareDatetime
    rows: list[LeaderboardRow] = Field(default_factory=list)
    pairs: list[PairVerdict] = Field(default_factory=list)
    tasks: list[LeaderboardTask] = Field(default_factory=list)
    counts: LeaderboardCounts | None = None
    highlights: list[Highlight] = Field(default_factory=list)


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
    margin: list[float | None]


class RowCharts(_Wire):
    """The curves of one row.

    `series` holds the series of each metric by its key. `density` is how fast the ranked metric rises at each time
    limit, in the unit of that metric per second. `rank` holds the places of the row at each time limit.
    """

    episodes: int = Field(ge=0)
    series: dict[MetricKey, EstimateSeries] = Field(default_factory=dict)
    density: list[float] = Field(default_factory=list)
    rank: list[RankRange | None] = Field(default_factory=list)


class LeaderboardChartsResponse(_LeaderboardPart):
    """`leaderboards.charts` — the charts over one task, or over every task where `task` is None.

    Each series has a value at each of the `windows`, which are time limits in seconds. The chart compares the rows
    in `pairs`.
    """

    task: LeaderboardTaskId | None = None
    windows: list[float]
    rows: dict[RowKey, RowCharts] = Field(default_factory=dict)
    pairs: list[PairVerdict] = Field(default_factory=list)


class EpisodeColumn(_Wire):
    """One column of the episode list: its name, the text that explains it, and the decimals of its values."""

    name: str
    description: str | None = None
    decimals: int | None = Field(default=None, ge=0)


class LeaderboardEpisode(_Wire):
    """One episode of the episode list.

    `outcome` is the key of the outcome group of the episode, and `result` is the text that the list shows for it.
    `steps` and `steps_done` count the steps of the task and the steps that the episode did. `furthest_stage` names
    how far the episode got in the step that it stopped in. `credit` is the percent of the task that the episode did,
    and `pace` is its value in the pace column. `media` is absent where no recording of the episode plays back.
    """

    id: EpisodeId
    row: RowKey
    checkpoint: CheckpointRef | None = None
    task: LeaderboardTaskId
    outcome: OutcomeKey
    result: str
    steps: int | None = Field(default=None, ge=0)
    steps_done: int | None = Field(default=None, ge=0)
    furthest_stage: str | None = None
    credit: Percent | None = None
    pace: float | None = None
    seconds_to_success: float | None = Field(default=None, ge=0)
    duration_s: float | None = Field(default=None, ge=0)
    recorded_at: AwareDatetime | None = None
    media: EpisodeMedia | None = None


class LeaderboardEpisodesResponse(_LeaderboardPart):
    """`leaderboards.episodes` — every episode that the leaderboard lists, in the order a page lists them.

    A list that shows no credit column or no difficulty column has None in its place.
    """

    result: EpisodeColumn
    pace: EpisodeColumn
    credit: EpisodeColumn | None = None
    difficulty: EpisodeColumn | None = None
    episodes: list[LeaderboardEpisode] = Field(default_factory=list)


class ProgressBand(_Wire):
    """The count of episodes in one band at each second, before smoothing.

    A band holds the episodes of one outcome group, by the key of the group, or the episodes that did one count of
    steps. Exactly one of `outcome` and `steps_done` is set.
    """

    outcome: OutcomeKey | None = None
    steps_done: int | None = Field(default=None, ge=0)
    episodes: list[int]

    @model_validator(mode='after')
    def _one_band_key(self) -> Self:
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
    episodes: int = Field(ge=0)
    bands: list[ProgressBand]
    edges: list[list[float]]
    mean_credit: list[Percent]
    mean_steps_done: list[float] | None = None
    mean_steps_done_at_end: float | None = None


class RowProgress(_Wire):
    """The progress panels of one row. All panels share one time axis, from 0 to `horizon_s`."""

    horizon_s: int = Field(ge=0)
    panels: list[ProgressPanel] = Field(default_factory=list)


class LeaderboardProgressResponse(_LeaderboardPart):
    """`leaderboards.progress` — the progress panels of each row that has episodes."""

    rows: dict[RowKey, RowProgress] = Field(default_factory=dict)


class CardFact(_Wire):
    """One fact of a model card. `source` is the URL or path that the text is from, and a page links `href`."""

    text: str
    source: str
    href: str | None = None


class CardLine(_Wire):
    """One line of a model card. `field` names the line, and `facts` holds one fact, or one for each checkpoint."""

    field: str = Field(min_length=1)
    facts: list[CardFact] = Field(min_length=1)


class RunStep(_Wire):
    """One command that a reader runs, under the `label` that names the step, with the `source` it is from."""

    label: str
    command: str
    source: str


class RunCommands(_Wire):
    """The commands that run a checkpoint, in the order a reader runs them.

    `checkpoint` names the checkpoint that the commands run, where the card covers more than one.
    """

    steps: list[RunStep] = Field(min_length=1)
    checkpoint: CheckpointRef | None = None


class ModelCard(_Wire):
    """What a model family is. A card gives the commands that run the model, or a guide to them, or neither."""

    lines: list[CardLine] = Field(default_factory=list)
    run_commands: RunCommands | None = None
    run_guide: CardFact | None = None

    @model_validator(mode='after')
    def _one_way_to_run_the_model(self) -> Self:
        if self.run_commands is not None and self.run_guide is not None:
            raise ValueError('a card gives at most one of the commands and a guide to them')
        return self


class ModelCheckpoint(_Wire):
    id: CheckpointRef
    display_name: str


class ModelFamily(_Wire):
    """One model family of `leaderboards.models`.

    A page shows `short_name` where the display name does not fit. A `baseline` family is a reference on every
    leaderboard, such as a person or teleoperation. On each leaderboard, the `baseline` of a row decides whether that
    row is a reference. `former_display_names` holds the names that the family had before.
    """

    id: ModelFamilyRef
    display_name: str
    short_name: str
    baseline: bool = False
    categories: list[str] = Field(default_factory=list)
    checkpoints: list[ModelCheckpoint] = Field(default_factory=list)
    card: ModelCard | None = None
    former_display_names: list[str] = Field(default_factory=list)


class LeaderboardModelsResponse(_Wire):
    """`leaderboards.models` — the model families that a leaderboard row can name, in a stable order."""

    models: list[ModelFamily] = Field(default_factory=list)
