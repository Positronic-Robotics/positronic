"""What every gateway endpoint answers with — the typed shape both sides bind to.

Timestamps are aware UTC, ids are hex strings (`platform_client.ids`), closed sets are slugs
(`platform_client.slug`), and locations are opaque. `submissions.get` answers one variant per status,
discriminated on the status slug.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, Self

from platform_client.boards import BoardRef
from platform_client.enums import (
    BoardVisibility,
    CameraVantage,
    EndpointKind,
    KeyStatus,
    OnExhausted,
    Placement,
    QuotaSubject,
    ReasonCode,
    SubmissionStatus,
)
from platform_client.eval_plan import Clutter
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, SubmissionId, UserId
from platform_client.slug import Slugged, slug_of
from platform_client.tasks import TaskRef
from pydantic import AwareDatetime, BaseModel, Discriminator, Field, Tag, model_validator


class Scores(BaseModel):
    """A finished run's published score. `primary` is its value under the eval's primary metric."""

    primary: float | None = None


# The rule keys a caller matches on: a 429 names one in its `details`, and `quota_for` takes one.
# A key absent from this list is a plan declaring its own rule, not an error.
QUOTA_SUBMISSIONS_DAY = 'submissions.day'
QUOTA_SUBMISSIONS_CONCURRENT = 'submissions.concurrent'


class QuotaLimit(BaseModel):
    """One rule of the caller's plan, and what is left of it at the moment of the read."""

    key: str  # the rule's identity, one of the published keys above — what a 429 names
    meter: str  # open set: a plan may declare one. 'submissions' | 'credits'
    unit: str  # the display unit, open for the same reason
    scale: int = Field(gt=0)  # meter units per display unit (credits: 6, submissions: 1), always positive
    window: str  # a display label: 'day', '24 Jul – 23 Aug', 'concurrent'
    subject: Slugged[QuotaSubject]
    scope: list[str] = Field(default_factory=list)  # tags this rule counts; empty counts the whole meter
    limit: int  # meter units
    used: int  # meter units; may exceed limit under on_exhausted=meter
    resets_at: AwareDatetime | None
    on_exhausted: Slugged[OnExhausted]

    @property
    def remaining(self) -> int:
        """Meter units left, clamped at 0."""
        return max(self.limit - self.used, 0)


class ArtifactRefs(BaseModel):
    """Where one submission's outputs are readable.

    `diagnostics` is why a failed run's policy failed. It is absent on a run that succeeded, and on
    a failed run whose record was not written.

    `policy_log` is the text the submitter's own policy container printed, stdout and stderr
    together, from the attempt that decided the run. It is absent where the container printed
    nothing.
    """

    result: str
    diagnostics: str | None = None
    policy_log: str | None = None


class ArtifactEntry(BaseModel):
    """One object a finished run wrote.

    `key` names it under the submission's own prefix, so `episodes/0000/meta.json` reads the same
    whichever bucket holds the run. `url` is a signed URL, because the submitter holds no
    credential for the bucket, and it expires — mint a fresh page to fetch again.
    """

    key: str
    size: int = Field(ge=0)
    url: str


class ArtifactListResponse(BaseModel):
    """`submissions.artifacts` — one page of objects, in key order.

    `next` is the LAST key this page returned, which is what the next page passes as its `after`.
    It is absent on the last page. A first-unseen key here would skip one object per boundary,
    because `after` is exclusive.
    """

    artifacts: list[ArtifactEntry] = Field(default_factory=list)
    next: str | None = None


class EpisodeCounts(BaseModel):
    """What a run asked for and where it stands.

    `total` is fixed when the plan is filed; the other two move as episodes land.
    """

    total: int = Field(default=0, ge=0)
    done: int = Field(default=0, ge=0)
    outstanding: int = Field(default=0, ge=0)


class RunSummary(BaseModel):
    """One launch that served the plan.

    `started_at` is when the operator pressed Start; `ended_at` is unset while it runs.
    """

    run_tag: str
    started_at: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None


class ResolvedEndpoint(BaseModel):
    """One endpoint of a resolved task: where its policy comes from, and the episodes it takes there."""

    name: str
    kind: Slugged[EndpointKind]
    url: str | None = None
    provider: str | None = None
    spec: str | None = None
    episodes: int = Field(ge=1)


# The sides a resolution may not carry: a draw is made before the plan is answered.
_UNRESOLVED_SIDES = frozenset({Placement.random, Placement.INVALID})


def _require_a_side(side: Placement, what: str) -> None:
    if side in _UNRESOLVED_SIDES:
        raise ValueError(f'{what} resolves to {slug_of(side)}, which names no side')


class ResolvedTask(BaseModel):
    """One task of a plan as the rig runs it: every property with its final value.

    The nearest level that states a value gives it; where none does, the task's catalogue entry gives
    it, and where that gives none, the platform draws it (`README.md` §Eval plans).
    `ResolvedPlan.tasks` lists the tasks in the plan's order.
    """

    task_id: TaskRef
    endpoints: list[ResolvedEndpoint] = Field(min_length=1)
    cap_per_episode_sec: int = Field(ge=1)
    policy_preset: str = Field(min_length=1)
    # `none` where the task has no tote.
    tote_placement: Slugged[Placement]
    # None where no external camera watches the task.
    camera_vantage: Slugged[CameraVantage] | None = None
    # Per external camera, keyed by its mount name.
    external_cameras: dict[str, Slugged[Placement]] = Field(default_factory=dict)
    # The bounds of the clutter draw, or None where the task is laid out with nothing else.
    clutter: Clutter | None = None
    # The kit objects the draw puts on the table beside the task's own.
    clutter_objects: list[str] = Field(default_factory=list)
    # The endpoint name each episode runs against, in the order the episodes run.
    episode_order: list[str] = Field(default_factory=list)

    @property
    def episodes(self) -> int:
        """Every episode this task takes, over all of its endpoints."""
        return sum(endpoint.episodes for endpoint in self.endpoints)

    @model_validator(mode='after')
    def _every_side_is_drawn(self) -> Self:
        _require_a_side(self.tote_placement, f'task {self.task_id!r} tote_placement')
        for mount, side in self.external_cameras.items():
            _require_a_side(side, f'task {self.task_id!r} external camera {mount!r}')
        return self

    @model_validator(mode='after')
    def _the_order_serves_each_count(self) -> Self:
        declared = {endpoint.name: endpoint.episodes for endpoint in self.endpoints}
        if Counter(self.episode_order) != Counter(declared):
            raise ValueError(f'task {self.task_id!r} orders {len(self.episode_order)} episodes against {declared}')
        return self


class ResolvedPlan(BaseModel):
    """A rig plan as it runs: the episode total, and each task with every value resolved."""

    episodes_total: int = Field(ge=1)
    tasks: list[ResolvedTask] = Field(min_length=1)

    @model_validator(mode='after')
    def _the_total_is_the_sum(self) -> Self:
        if self.episodes_total != sum(task.episodes for task in self.tasks):
            raise ValueError(f'episodes_total states {self.episodes_total}, and the tasks sum to another count')
        return self


# The outcomes that mint a key. `existing` is the one that does not.
_MINTING_OUTCOMES = frozenset({KeyStatus.created, KeyStatus.rotated})


class RegisterResponse(BaseModel):
    """`users.register`. `api_key` is present exactly when a key was minted: `created` or `rotated`."""

    user_id: UserId
    artifact_location: str
    api_key: ApiKey | None = None
    key_status: Slugged[KeyStatus]

    @model_validator(mode='after')
    def _the_key_and_the_outcome_agree(self) -> Self:
        # The outcome and the key are one fact, held together here rather than inferred apart. A
        # blank key is no key: caught here it reads as a malformed response, and caught at the
        # record it is a traceback out of a command that has already spent its one mint.
        minted = self.key_status in _MINTING_OUTCOMES
        if minted and not (self.api_key or '').strip():
            raise ValueError(f'key_status is {self.key_status.name} but no api_key came with it')
        if not minted and self.api_key is not None:
            raise ValueError(f'key_status is {self.key_status.name}, which mints no key, yet an api_key is present')
        return self


class MeResponse(BaseModel):
    """`users.me`."""

    user_id: UserId
    alias: str | None = None
    tenant: str
    plan: str
    quota: list[QuotaLimit]

    def quota_for(self, key: str) -> QuotaLimit | None:
        """The limit under a rule key (`QUOTA_SUBMISSIONS_DAY` and friends), or None where the plan
        carries no such rule."""
        return next((limit for limit in self.quota if limit.key == key), None)


class _ReasonBearing(BaseModel):
    """A flat submission row whose `reason_code` is absent unless `status` is `errored`."""

    status: Slugged[SubmissionStatus]
    reason_code: Slugged[ReasonCode] | None = None

    @model_validator(mode='after')
    def _a_reason_means_it_errored(self) -> Self:
        if self.reason_code is not None and self.status is not SubmissionStatus.errored:
            raise ValueError(f'reason_code {self.reason_code.name} on a {self.status.name} submission')
        return self


class SubmissionCreateResponse(_ReasonBearing):
    """`submissions.create` — covers a fresh create, an idempotent replay, and an unpullable image.

    An unpullable image is a caller fault: the submission is terminal and charged, so it comes
    back `errored` with a `reason_code` rather than as an error envelope.
    """

    submission_id: SubmissionId
    policy_image_digest: str | None = None
    # A rig plan as it will run. None on a plan the simulator runs, which names an eval.
    resolved: ResolvedPlan | None = None


class SubmissionListRow(_ReasonBearing):
    """One row of `submissions.list`. `user_id` attributes it — an admin listing spans users.

    `eval` is the name the catalogue expanded, and is absent on a run that stated its own tasks.
    """

    id: SubmissionId
    user_id: UserId
    alias: str | None = None
    eval: EvalRef | None = None
    episodes: EpisodeCounts = Field(default_factory=EpisodeCounts)
    received_at: AwareDatetime


class SubmissionListResponse(BaseModel):
    """`submissions.list` — one page, oldest first.

    `next` is the cursor for the page after it, and is absent on the last page.
    """

    submissions: list[SubmissionListRow] = Field(default_factory=list)
    next: SubmissionId | None = None


# The field the view union discriminates on, named so a rename moves the discriminator with it.
STATUS_FIELD = 'status'

# The field every view identifies a submission by, named for the same reason: a renderer that
# excludes it by a stale literal prints it twice.
ID_FIELD = 'id'


class _TaggedView(BaseModel):
    """One `submissions.get` variant. The union selects it by its `status` default, so a payload
    carrying any other status belongs to a different variant and is refused rather than validated
    into this one.
    """

    status: Slugged[SubmissionStatus]
    # What the run asked for and what it has landed, and the launches that served it. A run the
    # platform executes itself reports one launch; a plan the lab rig serves reports one per start.
    episodes: EpisodeCounts = Field(default_factory=EpisodeCounts)
    runs: list[RunSummary] = Field(default_factory=list)
    # A rig plan as it runs; `submissions.create` answered with the same. None on a simulator run.
    resolved: ResolvedPlan | None = None

    @model_validator(mode='after')
    def _the_status_is_this_variants_tag(self) -> Self:
        tag = type(self).model_fields[STATUS_FIELD].default
        if self.status is not tag:
            raise ValueError(f'{type(self).__name__} carries status {self.status.name}, not {tag.name}')
        return self


class PendingSubmissionView(_TaggedView):
    """Queued, not yet running. `queue_position` is 1-based by arrival and computed on read."""

    id: SubmissionId
    alias: str | None = None
    received_at: AwareDatetime
    queued_at: AwareDatetime
    queue_position: int = Field(gt=0)
    status: Slugged[SubmissionStatus] = SubmissionStatus.pending


class RunningSubmissionView(_TaggedView):
    """Executing. `stage` names the orchestrator stage; `stage_detail` decorates it for display."""

    id: SubmissionId
    running_since: AwareDatetime
    stage: str | None = None
    stage_detail: str | None = None
    status: Slugged[SubmissionStatus] = SubmissionStatus.running


class ErroredSubmissionView(_TaggedView):
    """Terminal failure. `reason_code` is the machine-readable taxonomy; `reason` is for humans.

    `artifacts` reads the run's own records. It is absent where the run wrote none.
    """

    id: SubmissionId
    reason_code: Slugged[ReasonCode] | None = None
    reason: str | None = None
    artifacts: ArtifactRefs | None = None
    status: Slugged[SubmissionStatus] = SubmissionStatus.errored


class FinishedSubmissionView(_TaggedView):
    """Terminal success."""

    id: SubmissionId
    scores: Scores = Field(default_factory=Scores)
    artifacts: ArtifactRefs
    status: Slugged[SubmissionStatus] = SubmissionStatus.finished


class CancelledSubmissionView(_TaggedView):
    """Terminal, cancelled by the caller."""

    id: SubmissionId
    cancelled_at: AwareDatetime | None = None
    status: Slugged[SubmissionStatus] = SubmissionStatus.cancelled


class BlockedSubmissionView(_TaggedView):
    """A blocked run, and the `reason` it waits on."""

    id: SubmissionId
    reason: str | None = None
    status: Slugged[SubmissionStatus] = SubmissionStatus.blocked


def _status_tag(value: Any) -> str | None:
    """The status slug a `submissions.get` payload selects its variant by.

    Reads the raw input, so it runs before validation and must handle both a decoded JSON mapping
    and an already-built model.
    """
    raw = value.get(STATUS_FIELD) if isinstance(value, dict) else getattr(value, STATUS_FIELD, None)
    if isinstance(raw, SubmissionStatus):
        return slug_of(raw)
    return raw if isinstance(raw, str) else None


# Each tag is the slug of the status its variant declares, taken from the enum rather than spelled:
# the wire vocabulary is the enum's, so a member renamed there renames the discriminator with it.
SubmissionView = Annotated[
    Annotated[PendingSubmissionView, Tag(slug_of(SubmissionStatus.pending))]
    | Annotated[RunningSubmissionView, Tag(slug_of(SubmissionStatus.running))]
    | Annotated[ErroredSubmissionView, Tag(slug_of(SubmissionStatus.errored))]
    | Annotated[FinishedSubmissionView, Tag(slug_of(SubmissionStatus.finished))]
    | Annotated[CancelledSubmissionView, Tag(slug_of(SubmissionStatus.cancelled))]
    | Annotated[BlockedSubmissionView, Tag(slug_of(SubmissionStatus.blocked))],
    Discriminator(_status_tag),
]


class CancelResponse(BaseModel):
    """`submissions.cancel`. `refunded` is false once the run started — started work is charged."""

    status: Slugged[SubmissionStatus]
    refunded: bool


class RankingRow(BaseModel):
    """One row of a board: a user's best submission on it.

    `display_name` is what the board shows — the user's alias, or a placeholder where a board is
    anonymous or the user set none. It is NOT an identifier: aliases are not unique, so two rows may
    carry the same one. `tag` is what tells them apart, and is what lets a user find their own row;
    it is stable for a user across boards. Render them together (`ateam#0ddba7`).

    The value the board ranks on is `scores.primary`.
    """

    rank: int
    display_name: str
    tag: str
    scores: Scores = Field(default_factory=Scores)
    submission_id: SubmissionId
    submitted_at: AwareDatetime


class BoardSummary(BaseModel):
    """One board of `rankings.list`. `board` is the slug `rankings.get` takes."""

    board: BoardRef
    title: str
    eval: EvalRef
    primary_metric: str
    visibility: Slugged[BoardVisibility]


class BoardListResponse(BaseModel):
    """`rankings.list` — the boards the caller can see. A board they cannot see is absent, not refused."""

    boards: list[BoardSummary] = Field(default_factory=list)


class RankingsResponse(BaseModel):
    """`rankings.get` for one board.

    `primary_metric` NAMES the metric the board sorts on — `scores.primary`, whatever the eval calls
    it. It is a label for a reader, never a key: nothing looks a value up by it, and it is not
    guaranteed to name a field of `Scores`.
    """

    board: BoardRef
    eval: EvalRef
    primary_metric: str
    rankings: list[RankingRow] = Field(default_factory=list)
