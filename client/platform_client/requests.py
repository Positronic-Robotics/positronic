"""What a caller sends: the POST bodies, and the query models the GET endpoints take.

Unknown fields are rejected, so a typo'd field is a 422 rather than a silently dropped input that
would change what the submission means. `submissions.create` takes an `EvalPlan`, which lives in
`eval_plan` with the cascade it carries.
"""

from __future__ import annotations

from platform_client.boards import BoardRef
from platform_client.ids import SubmissionId
from platform_client.model_config import INPUT_MODEL_CONFIG
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """`users.register` — create-or-return, keyed on the external identity behind `credential`."""

    model_config = INPUT_MODEL_CONFIG

    credential: str
    alias: str | None = None
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


class SubmissionListQuery(BaseModel):
    """`submissions.list` — the page after the last id seen. A `limit` above the gateway's cap is clamped to it."""

    model_config = INPUT_MODEL_CONFIG

    after: SubmissionId | None = None
    limit: int | None = Field(default=None, gt=0)
