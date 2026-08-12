"""What a caller sends: the POST bodies, and the query models the GET endpoints take.

Both halves are models so that pydantic owns the field names on both sides of the wire, and a GET's
parameters are built by dumping one rather than by assembling a dict beside the path. A parameter
left unset is dumped away (`exclude_none`) and is absent from the URL, never an empty value.

Every model rejects unknown fields: a typo'd field is a 422, not a silently dropped input that would
change what the submission means without the caller knowing.
"""

from __future__ import annotations

from platform_client.evals import EvalRef
from platform_client.ids import SubmissionId, TransactionKey
from platform_client.images import ImageRef
from pydantic import BaseModel, ConfigDict, Field

_FORBID_EXTRA = ConfigDict(extra='forbid')


class RegisterRequest(BaseModel):
    """`users.register` — create-or-return, keyed on the external identity behind `credential`."""

    model_config = _FORBID_EXTRA

    credential: str
    alias: str | None = None
    rotate: bool = False


class SubmissionCreateRequest(BaseModel):
    """`submissions.create` — the run-defining fields, exactly as sent."""

    model_config = _FORBID_EXTRA

    # An `ImageRef`, so a reference the registry could never resolve is refused in the caller's own
    # process instead of spending a round trip to learn it.
    policy_image: ImageRef
    # The whole of what the caller chooses: the eval names its own embodiment, and the platform
    # answers an unknown name with the ones it offers.
    eval: EvalRef
    alias: str | None = None
    # A present key must be non-empty: an empty string is a client bug, not "no key", and accepting
    # it would silently drop the caller's dedup guarantee.
    transaction_key: TransactionKey | None = Field(default=None, min_length=1)


class CancelRequest(BaseModel):
    """`submissions.cancel`."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class SubmissionGetQuery(BaseModel):
    """`submissions.get` — the id travels in the query string, in its hex wire form."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class RankingsQuery(BaseModel):
    """`rankings.get` — one board by slug. An `eval_version` pins a past board; omitting it asks for
    the board as it stands."""

    model_config = _FORBID_EXTRA

    board: str
    eval_version: str | None = None
