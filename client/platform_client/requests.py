"""What a caller sends: the POST bodies, and the query models the GET endpoints take.

Unknown fields are rejected, so a typo'd field is a 422 rather than a silently dropped input that
would change what the submission means. `submissions.create` takes an `EvalPlan`, which lives in
`eval_plan` with the cascade it carries.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated

from platform_client.boards import BoardRef
from platform_client.ids import LeaderboardVersionId, OrgSlug, PackageId, PurchaseId, SubmissionId, TransactionKey
from platform_client.leaderboards import LeaderboardRef, LeaderboardTag, LeaderboardTaskId
from platform_client.model_config import INPUT_MODEL_CONFIG
from pydantic import AfterValidator, BaseModel, Field

TEAM_NAME_MAX_CHARACTERS = 64
TEAM_NAME_RULE = (
    f'a team name has 1 to {TEAM_NAME_MAX_CHARACTERS} characters after the spaces at its ends are removed, '
    'and no control or text-direction characters'
)
# The Unicode Bidi_Control property. Other format characters (Cf) stay: emoji and some scripts need the
# zero-width joiners.
_BIDI_CONTROLS = frozenset('\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')


def team_name(alias: str) -> str:
    """The alias without the spaces at its ends. Raise `ValueError` on one that breaks `TEAM_NAME_RULE`."""
    name = alias.strip()
    if not 1 <= len(name) <= TEAM_NAME_MAX_CHARACTERS or any(
        unicodedata.category(character) == 'Cc' or character in _BIDI_CONTROLS for character in name
    ):
        raise ValueError(TEAM_NAME_RULE)
    return name


class RegisterRequest(BaseModel):
    """`users.register` — create-or-return, keyed on the external identity behind `credential`."""

    model_config = INPUT_MODEL_CONFIG

    credential: str
    alias: Annotated[str, AfterValidator(team_name)] | None = None
    rotate: bool = False


class CancelRequest(BaseModel):
    """`submissions.cancel`."""

    model_config = INPUT_MODEL_CONFIG

    id: SubmissionId


class SubmissionGetQuery(BaseModel):
    """`submissions.get` — the id travels in the query string, in its hex wire form."""

    model_config = INPUT_MODEL_CONFIG

    id: SubmissionId


class SubmissionArtifactsQuery(BaseModel):
    """`submissions.artifacts` — one page of the objects a finished submission wrote.

    `prefix` keeps the page to the keys under it, and is read relative to the submission's own
    prefix: `episodes/` lists the episodes alone. `after` is the key the page before it ended on.
    A `limit` above the gateway's cap is clamped to it.
    """

    model_config = INPUT_MODEL_CONFIG

    id: SubmissionId
    prefix: str | None = None
    after: str | None = None
    limit: int | None = Field(default=None, gt=0)


class RankingsQuery(BaseModel):
    """`rankings.get` — one board by slug."""

    model_config = INPUT_MODEL_CONFIG

    board: BoardRef


class LeaderboardListQuery(BaseModel):
    """`leaderboards.list` — the leaderboards of one tag."""

    model_config = INPUT_MODEL_CONFIG

    tag: LeaderboardTag


class LeaderboardQuery(BaseModel):
    """`leaderboards.get` — one leaderboard by slug."""

    model_config = INPUT_MODEL_CONFIG

    leaderboard: LeaderboardRef


class LeaderboardPartQuery(BaseModel):
    """`leaderboards.episodes` and `leaderboards.progress` — one part of a leaderboard at `version`.

    With no `version`, the platform answers from the published version. A leaderboard with no versions has no parts.
    """

    model_config = INPUT_MODEL_CONFIG

    leaderboard: LeaderboardRef
    version: LeaderboardVersionId | None = None


class LeaderboardChartsQuery(LeaderboardPartQuery):
    """`leaderboards.charts` — the charts over one `task`, or over every task where it is None."""

    task: LeaderboardTaskId | None = None


class SubmissionListQuery(BaseModel):
    """`submissions.list` — the page after the last id seen. A `limit` above the gateway's cap is clamped to it."""

    model_config = INPUT_MODEL_CONFIG

    after: SubmissionId | None = None
    limit: int | None = Field(default=None, gt=0)


class BillingOrgQuery(BaseModel):
    model_config = INPUT_MODEL_CONFIG

    org: OrgSlug = Field(min_length=1)


DEFAULT_PURCHASE_PAGE_SIZE = 50
PurchasePageLimit = Annotated[int, Field(strict=True, gt=0)]


class BillingPurchaseListQuery(BaseModel):
    model_config = INPUT_MODEL_CONFIG

    org: OrgSlug = Field(min_length=1)
    after: PurchaseId | None = Field(default=None, min_length=1)
    limit: PurchasePageLimit = DEFAULT_PURCHASE_PAGE_SIZE


class BillingPurchaseGetQuery(BaseModel):
    model_config = INPUT_MODEL_CONFIG

    id: PurchaseId = Field(min_length=1)


class BillingPurchaseCreateRequest(BaseModel):
    model_config = INPUT_MODEL_CONFIG

    org: OrgSlug = Field(min_length=1)
    package_id: PackageId = Field(min_length=1)
    transaction_key: TransactionKey = Field(min_length=1)
