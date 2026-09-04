"""`PlatformClient` — one method per gateway endpoint, request model in, response model out.

    with PlatformClient() as client:
        client.register(RegisterRequest(credential=..., alias=...))  # keeps the key it returns
        client.create_submission(SubmissionCreateRequest(policy_image=..., eval=...))

The platform is `base_url`, else `POSITRONIC_PLATFORM_URL`, else production; the key is `api_key`,
else `POSITRONIC_PLATFORM_API_KEY`, else whatever `register` came back with. A non-2xx raises
`PlatformError` carrying the error envelope.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from types import TracebackType
from typing import Any, ClassVar, Self, TypeVar

import httpx
from platform_client import routes
from platform_client.boards import BoardRef
from platform_client.errors import PlatformError
from platform_client.ids import ApiKey, RequestId, SubmissionId
from platform_client.requests import (
    CancelRequest,
    RankingsQuery,
    RegisterRequest,
    RequestCreate,
    RequestGetQuery,
    RequestListQuery,
    SubmissionCreateRequest,
    SubmissionGetQuery,
)
from platform_client.responses import (
    BoardListResponse,
    CancelResponse,
    MeResponse,
    RankingsResponse,
    RegisterResponse,
    RequestCreated,
    RequestListResponse,
    RequestView,
    SubmissionCreateResponse,
    SubmissionListResponse,
    SubmissionView,
)
from pydantic import BaseModel, TypeAdapter

M = TypeVar('M', bound=BaseModel)

DEFAULT_TIMEOUT_S = 30.0

# The platform a caller reaches with no configuration at all.
DEFAULT_PLATFORM_URL = 'https://platform.positronic.ro'

# What an authenticated call sends, what `users.register` must not, and what a caller's own client
# may therefore not default — three places that have to agree.
AUTH_HEADER = 'Authorization'

API_URL_ENV = 'POSITRONIC_PLATFORM_URL'
API_KEY_ENV = 'POSITRONIC_PLATFORM_API_KEY'
CREDENTIAL_ENV = 'POSITRONIC_PLATFORM_CREDENTIAL'


def resolve_api_key(api_key: ApiKey | None = None) -> ApiKey | None:
    """The key to authenticate with: what the caller passed, else the environment, else none."""
    if api_key is not None:
        return api_key
    from_env = os.environ.get(API_KEY_ENV)
    return ApiKey(from_env) if from_env else None


def require_absolute_url(url: str | httpx.URL, source: str) -> None:
    """Refuse a base URL that names no host, naming the value and what it has to be.

    Every endpoint sends a path relative to the client's base URL, so one that names no host — empty,
    or a bare path — resolves against nothing and the request never leaves for a platform.
    """
    if httpx.URL(url).is_absolute_url:
        return
    raise ValueError(
        f'{source} is {str(url)!r}, which names no host: give an absolute URL, scheme and host, naming the platform'
    )


def resolve_base_url(base_url: str | None = None) -> str:
    """The platform to talk to: what the caller passed, else the environment, else the default."""
    for value, source in ((base_url, 'base_url'), (os.environ.get(API_URL_ENV), API_URL_ENV)):
        if value is None:
            continue
        # Empty is a misconfiguration, not a request for the default: `--platform-url=` reads as a
        # platform the caller named, and falling through would send that run to the production one.
        if not value.strip():
            raise ValueError(f'{source} is empty; name a platform, or leave it unset for {DEFAULT_PLATFORM_URL}')
        require_absolute_url(value, source)
        return value
    return DEFAULT_PLATFORM_URL


class Auth(Enum):
    """What an endpoint does with the client's API key."""

    REQUIRED = auto()
    # Sent when one is set: a public board answers either way, a tenant's board only with a key.
    OPTIONAL = auto()
    NONE = auto()


class PlatformClient:
    """A caller's handle on one platform.

    Pass an `httpx.Client` instead of a `base_url` when the transport matters (an in-process ASGI
    transport, a proxy, a retry policy); it carries its own base URL, so the two are exclusive.
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
        else:
            if base_url is not None:
                raise ValueError('a client of your own already carries its base URL; pass one or the other')
            require_absolute_url(client.base_url, "the supplied client's base_url")
            if AUTH_HEADER in client.headers:
                # httpx MERGES client-level headers into every request, so a default here would reach
                # `users.register`, which this module declares unauthenticated and the gateway reads
                # as a registration by whoever that credential names.
                raise ValueError(f'the supplied client carries a default {AUTH_HEADER} header; pass the key as api_key')
            if client.auth is not None:
                # An auth flow reaches the same requests by another route: httpx runs it on every
                # one, so it would sign `users.register` too.
                raise ValueError(
                    f'the supplied client carries an auth flow, which sets {AUTH_HEADER} on every '
                    f'request including `users.register`; pass the key as api_key'
                )
        self._client = client
        self.api_key = resolve_api_key(api_key)

    # --- endpoints ---------------------------------------------------------------------------

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Create-or-return a user, KEEPING any key it returns — the next call is authenticated.

        A key comes back only on a first registration or a `rotate`; a repeat registration answers
        `existing` with none, and the client goes on holding whatever it already had.
        """
        response = self._post(routes.USERS_REGISTER, request, RegisterResponse, auth=Auth.NONE)
        if response.api_key is not None:
            self.api_key = response.api_key
        return response

    def me(self) -> MeResponse:
        return self._get(routes.USERS_ME, MeResponse)

    def create_submission(self, request: SubmissionCreateRequest) -> SubmissionCreateResponse:
        return self._post(routes.SUBMISSIONS_CREATE, request, SubmissionCreateResponse)

    def list_submissions(self) -> SubmissionListResponse:
        """The caller's own submissions — or every user's, for an admin or service principal."""
        return self._get(routes.SUBMISSIONS_LIST, SubmissionListResponse)

    # The view is a discriminated union rather than a model, so it validates through an adapter.
    # Built once, because building one costs more than the validation it then does.
    _SUBMISSION_VIEW: ClassVar[TypeAdapter[SubmissionView]] = TypeAdapter(SubmissionView)

    def get_submission(self, submission_id: SubmissionId) -> SubmissionView:
        query = SubmissionGetQuery(id=submission_id)
        return self._SUBMISSION_VIEW.validate_json(self._send('GET', routes.SUBMISSIONS_GET, query=query).content)

    def cancel_submission(self, request: CancelRequest) -> CancelResponse:
        return self._post(routes.SUBMISSIONS_CANCEL, request, CancelResponse)

    def rankings(self, *, board: BoardRef) -> RankingsResponse:
        """One board by slug."""
        query = RankingsQuery(board=board)
        return self._get(routes.RANKINGS_GET, RankingsResponse, query=query, auth=Auth.OPTIONAL)

    def list_boards(self) -> BoardListResponse:
        """The boards this caller can read: the public ones, plus their tenant's once a key is set."""
        return self._get(routes.RANKINGS_LIST, BoardListResponse, auth=Auth.OPTIONAL)

    def requests_create(self, request: RequestCreate) -> RequestCreated:
        """File one rollout request. Needs a customer grant: a key without one is refused `forbidden`."""
        return self._post(routes.REQUESTS_CREATE, request, RequestCreated)

    def requests_get(self, request_id: RequestId) -> RequestView:
        return self._get(routes.REQUESTS_GET, RequestView, query=RequestGetQuery(id=request_id))

    def requests_list(self, *, after: RequestId | None = None, limit: int | None = None) -> RequestListResponse:
        """One page of the caller's requests, oldest first. Pass a page's `next` as `after` for the page after it."""
        query = RequestListQuery(after=after, limit=limit)
        return self._get(routes.REQUESTS_LIST, RequestListResponse, query=query)

    # --- plumbing ----------------------------------------------------------------------------

    # Validated from the raw bytes, never from a decoded object: an id is hex text on the wire and
    # only the JSON schema enforces that, so decoding first would take a number as well.
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
            raise ValueError(f'{path} needs an API key: pass api_key=, set {API_KEY_ENV}, or call register() first')
        if auth is not Auth.NONE and self.api_key is not None:
            headers[AUTH_HEADER] = f'Bearer {self.api_key}'
        response = self._client.request(method, path, headers=headers, **kwargs)
        # Not `>= 400`: httpx follows no redirect by default, so a 3xx arrives here with a body that
        # is not an envelope and would surface as a parse error instead of the promised PlatformError.
        if not 200 <= response.status_code < 300:
            try:
                payload: object = response.json()
            except ValueError:
                # A failure need not carry an envelope at all: a proxy's HTML 502 arrives here too.
                payload = response.text
            raise PlatformError.from_payload(response.status_code, payload)
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
