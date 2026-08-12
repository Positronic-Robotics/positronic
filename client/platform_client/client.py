"""`PlatformClient` — one method per gateway endpoint, request model in, response model out.

A response is validated from its raw bytes rather than from a decoded object: an id is hex text on
the wire, and only the JSON schema enforces that — decoding first would take a number as well.

A non-2xx response raises `PlatformError` carrying the parsed error envelope. `api_key` is a plain
attribute: `users.register` returns one, and every authenticated call sends whatever is set.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from types import TracebackType
from typing import Any, Self, TypeVar

import httpx
from platform_client import routes
from platform_client.errors import PlatformError
from platform_client.ids import ApiKey, SubmissionId
from platform_client.requests import (
    CancelRequest,
    RankingsQuery,
    RegisterRequest,
    SubmissionCreateRequest,
    SubmissionGetQuery,
)
from platform_client.responses import (
    BoardListResponse,
    CancelResponse,
    MeResponse,
    RankingsResponse,
    RegisterResponse,
    SubmissionCreateResponse,
    SubmissionListResponse,
    SubmissionView,
)
from pydantic import BaseModel, TypeAdapter

M = TypeVar('M', bound=BaseModel)

_SUBMISSION_VIEW: TypeAdapter[SubmissionView] = TypeAdapter(SubmissionView)

DEFAULT_TIMEOUT_S = 30.0

# The platform a caller reaches with no configuration at all. A user should never have to know a
# URL; the environment variable below is for a developer pointing at a gateway of their own.
DEFAULT_PLATFORM_URL = 'https://platform.positronic.ro'

# How a caller is configured. Both secrets are read from the environment and never taken as an
# argument: a command line is readable by every process on the box and lands in shell history.
# Named because three places agree on it: what an authenticated call sends, what an unauthenticated
# one must not, and what a caller's own client may therefore not default.
AUTH_HEADER = 'Authorization'

API_URL_ENV = 'POSITRONIC_PLATFORM_URL'
API_KEY_ENV = 'POSITRONIC_PLATFORM_API_KEY'
CREDENTIAL_ENV = 'POSITRONIC_PLATFORM_CREDENTIAL'


def resolve_base_url(base_url: str | None = None) -> str:
    """The platform to talk to: what the caller passed, else the environment, else the default.

    One owner for the precedence, so a command and a script that skip different steps of it cannot
    end up talking to different platforms.
    """
    return base_url or os.environ.get(API_URL_ENV) or DEFAULT_PLATFORM_URL


class Auth(Enum):
    """What an endpoint does with the client's API key."""

    REQUIRED = auto()
    # Sent when one is set: a public board answers either way, a tenant's board only with a key.
    OPTIONAL = auto()
    NONE = auto()


class PlatformClient:
    """A caller's handle on one platform.

    With no `base_url` it reads `API_URL_ENV`, and falls back to `DEFAULT_PLATFORM_URL` — so a user
    configures nothing, a developer exports one variable, and a caller with a URL of its own passes
    it. Pass an `httpx.Client` instead when the transport matters (an in-process ASGI transport, a
    custom retry or proxy configuration); it carries its own base URL.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: ApiKey | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        if client is None:
            client = httpx.Client(base_url=resolve_base_url(base_url), timeout=timeout)
        elif base_url is not None:
            raise ValueError('a client of your own already carries its base URL; pass one or the other')
        elif AUTH_HEADER in client.headers:
            # httpx MERGES client-level headers into every request, so a default here would reach
            # `users.register`, which this module declares unauthenticated and the gateway reads as
            # a registration by whoever that credential names.
            raise ValueError(f'the supplied client carries a default {AUTH_HEADER} header; pass the key as api_key')
        self._client = client
        self.api_key = api_key

    # --- endpoints ---------------------------------------------------------------------------

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Create-or-return a user. A key comes back only on a first registration or a `rotate`."""
        return self._post(routes.USERS_REGISTER, request, RegisterResponse, auth=Auth.NONE)

    def me(self) -> MeResponse:
        return self._get(routes.USERS_ME, MeResponse)

    def create_submission(self, request: SubmissionCreateRequest) -> SubmissionCreateResponse:
        return self._post(routes.SUBMISSIONS_CREATE, request, SubmissionCreateResponse)

    def list_submissions(self) -> SubmissionListResponse:
        """The caller's own submissions — or every user's, for an admin or service principal."""
        return self._get(routes.SUBMISSIONS_LIST, SubmissionListResponse)

    def get_submission(self, submission_id: SubmissionId) -> SubmissionView:
        query = SubmissionGetQuery(id=submission_id)
        return _SUBMISSION_VIEW.validate_json(self._send('GET', routes.SUBMISSIONS_GET, query=query).content)

    def cancel_submission(self, request: CancelRequest) -> CancelResponse:
        return self._post(routes.SUBMISSIONS_CANCEL, request, CancelResponse)

    def rankings(self, *, board: str, eval_version: str | None = None) -> RankingsResponse:
        """One board by slug. An explicit `eval_version` pins a past board; otherwise the current one."""
        query = RankingsQuery(board=board, eval_version=eval_version)
        return self._get(routes.RANKINGS_GET, RankingsResponse, query=query, auth=Auth.OPTIONAL)

    def list_boards(self) -> BoardListResponse:
        """The boards this caller can read: the public ones, plus their tenant's once a key is set."""
        return self._get(routes.RANKINGS_LIST, BoardListResponse, auth=Auth.OPTIONAL)

    # --- plumbing ----------------------------------------------------------------------------

    def _get(self, path: str, model: type[M], *, query: BaseModel | None = None, auth: Auth = Auth.REQUIRED) -> M:
        return model.model_validate_json(self._send('GET', path, query=query, auth=auth).content)

    def _post(self, path: str, request: BaseModel, model: type[M], *, auth: Auth = Auth.REQUIRED) -> M:
        body = request.model_dump(mode='json')
        return model.model_validate_json(self._send('POST', path, json=body, auth=auth).content)

    def _send(
        self, method: str, path: str, *, query: BaseModel | None = None, auth: Auth = Auth.REQUIRED, **kwargs: Any
    ) -> httpx.Response:
        # `exclude_none` is what makes an unset parameter absent from the URL rather than an empty
        # value, which the gateway would have to tell apart from a caller that meant the empty string.
        if query is not None:
            kwargs['params'] = query.model_dump(mode='json', exclude_none=True)
        headers: dict[str, str] = {}
        if auth is Auth.REQUIRED and self.api_key is None:
            raise ValueError(f'{path} needs an API key; set .api_key from a users.register response')
        if auth is not Auth.NONE and self.api_key is not None:
            headers[AUTH_HEADER] = f'Bearer {self.api_key}'
        response = self._client.request(method, path, headers=headers, **kwargs)
        # Not `>= 400`: httpx follows no redirect by default, so a 3xx arrives here with a body that
        # is not an envelope and would surface as a parse error instead of the promised PlatformError.
        if not 200 <= response.status_code < 300:
            raise PlatformError.from_payload(response.status_code, _decode(response))
        return response

    def close(self) -> None:
        """Close the underlying transport, unless the caller supplied its own client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def _decode(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text
