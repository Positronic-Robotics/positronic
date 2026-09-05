"""What every gateway endpoint answers with — the typed shape both sides bind to.

Timestamps are aware UTC, ids are hex strings (`platform_client.ids`), closed sets are slugs
(`platform_client.slug`), and locations are opaque. `submissions.get` answers one of five variants,
discriminated on the status slug.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from platform_client.boards import BoardRef
from platform_client.enums import (
    REQUEST_STOPPED_STATUSES,
    BoardVisibility,
    KeyStatus,
    OnExhausted,
    QuotaSubject,
    ReasonCode,
    RequestStatus,
    SubmissionStatus,
)
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, RequestId, SubmissionId, UserId
from platform_client.slug import Slugged, slug_of
from pydantic import AfterValidator, AwareDatetime, BaseModel, Discriminator, Field, Tag, model_validator


def _public(status: SubmissionStatus) -> SubmissionStatus:
    """A status as a CALLER may see it: `submitting` is an internal claim state, reported as `pending`."""
    if status is SubmissionStatus.submitting:
        raise ValueError(f'{status.name} is an internal state and never reaches a caller')
    return status


# Every status a caller-facing model may carry. The five `submissions.get` variants pin their own
# tag instead (`_TaggedView`), which is the same rule stated per variant.
PublicStatus = Annotated[Slugged[SubmissionStatus], AfterValidator(_public)]


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
    """Where a finished submission's outputs are readable."""

    result: str


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
        # The outcome and the key are one fact, held together here rather than inferred apart.
        minted = self.key_status in _MINTING_OUTCOMES
        if minted and self.api_key is None:
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

    status: PublicStatus
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


class SubmissionListRow(_ReasonBearing):
    """One row of `submissions.list`. `user_id` attributes it — an admin listing spans users."""

    id: SubmissionId
    user_id: UserId
    alias: str | None = None
    eval: EvalRef
    received_at: AwareDatetime


class SubmissionListResponse(BaseModel):
    submissions: list[SubmissionListRow] = Field(default_factory=list)


# The field the view union discriminates on, named so a rename moves the discriminator with it.
STATUS_FIELD = 'status'

# The field every view identifies a submission by, named for the same reason: a renderer that
# excludes it by a stale literal prints it twice.
ID_FIELD = 'id'


class _TaggedView(BaseModel):
    """One `submissions.get` variant. Its `status` default IS the tag the union selects it by, so a
    payload carrying any other status belongs to a different variant and is refused rather than
    validated into this one — which is what stops an internal state the union has no variant for,
    `submitting`, from arriving dressed as a pending view.
    """

    status: Slugged[SubmissionStatus]

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
    """Terminal failure. `reason_code` is the machine-readable taxonomy; `reason` is for humans."""

    id: SubmissionId
    reason_code: Slugged[ReasonCode] | None = None
    reason: str | None = None
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
    | Annotated[CancelledSubmissionView, Tag(slug_of(SubmissionStatus.cancelled))],
    Discriminator(_status_tag),
]


class CancelResponse(BaseModel):
    """`submissions.cancel`. `refunded` is false once the run started — started work is charged."""

    status: PublicStatus
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


class RequestCreated(BaseModel):
    """`requests.create` — a fresh request, or the one an earlier create under the same key made."""

    request_id: RequestId
    status: Slugged[RequestStatus]


class EpisodeCounts(BaseModel):
    """What a request asked for and where it stands: `total` is fixed at create; the other two move as episodes land."""

    total: int = Field(ge=0)
    done: int = Field(ge=0)
    outstanding: int = Field(ge=0)


class RunSummary(BaseModel):
    """One launch that served the request.

    `started_at` is when the operator pressed Start; `ended_at` is unset while it runs.
    """

    run_tag: str
    started_at: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None


class RequestView(BaseModel):
    """`requests.get`, and one row of `requests.list`.

    `slug` names the request once the coordinator files it. `artifacts` is the prefix the episodes
    land under, once one exists. `error` says why a `blocked` request waits, or why an `errored` one
    stopped.
    """

    request_id: RequestId
    status: Slugged[RequestStatus]
    slug: str | None = None
    episodes: EpisodeCounts
    runs: list[RunSummary] = Field(default_factory=list)
    artifacts: str | None = None
    error: str | None = None

    @model_validator(mode='after')
    def _an_error_travels_with_a_stopped_status(self) -> Self:
        if self.error is not None and self.status not in REQUEST_STOPPED_STATUSES:
            raise ValueError(f'an error on a {self.status.name} request')
        return self


class RequestListResponse(BaseModel):
    """`requests.list` — one page, oldest first.

    `next` is the cursor for the page after it, and is absent on the last page.
    """

    requests: list[RequestView] = Field(default_factory=list)
    next: RequestId | None = None
