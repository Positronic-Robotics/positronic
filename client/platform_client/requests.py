"""What a caller sends: the POST bodies, and the query models the GET endpoints take.

Unknown fields are rejected, so a typo'd field is a 422 rather than a silently dropped input that
would change what the submission means. `submissions.create` takes an `EvalPlan`, which lives in
`eval_plan` with the cascade it carries.
"""

from __future__ import annotations

from platform_client.boards import BoardRef
from platform_client.ids import SubmissionId
from pydantic import BaseModel, ConfigDict, Field

_FORBID_EXTRA = ConfigDict(extra='forbid')


class RegisterRequest(BaseModel):
    """`users.register` — create-or-return, keyed on the external identity behind `credential`."""

    model_config = _FORBID_EXTRA

    credential: str
    alias: str | None = None
    rotate: bool = False


class CancelRequest(BaseModel):
    """`submissions.cancel`."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class SubmissionGetQuery(BaseModel):
    """`submissions.get` — the id travels in the query string, in its hex wire form."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class RankingsQuery(BaseModel):
    """`rankings.get` — one board by slug."""

    model_config = _FORBID_EXTRA

    board: BoardRef


class SubmissionListQuery(BaseModel):
    """`submissions.list` — the page after the last id seen. A `limit` above the gateway's cap is clamped to it."""

    model_config = _FORBID_EXTRA

    after: SubmissionId | None = None
    limit: int | None = Field(default=None, gt=0)
