"""The device flow of RFC 8628, and the `platform-register` command that runs it.

The platform's `auth.config` names the identity provider: WorkOS or GitHub. The flow asks that provider for a
device code, shows the user the short code, polls until the provider answers, and hands the access token to
`users.register` as its `credential`. It keeps no other token: WorkOS also sends a refresh token, and the flow
drops it.

Usage
  platform-register --alias='<display name>'
  platform-register --platform-url=http://127.0.0.1:8080
  platform-register --platform-url=http://staging.internal:8080 --plaintext-http  # a plain http link
  platform-register --rotate  # mint a new key for an account that is already registered
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto

import httpx
from platform_client import routes
from platform_client.client import API_KEY_ENV, API_URL_ENV, PlatformClient, resolve_base_url
from platform_client.errors import PlatformError
from platform_client.identity_providers import (
    GITHUB,
    GITHUB_REGISTER_TIMEOUT,
    REQUEST_TIMEOUT,
    WORKOS,
    IdentityProvider,
    workos_register_timeout,
)
from platform_client.requests import RegisterRequest
from platform_client.responses import RegisterResponse
from pydantic import ValidationError

# The grant the token request names (RFC 8628 §3.4).
DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'

# The platform's OAuth app `positronic-platform-device-flow`. A device flow carries no client
# secret, so its client id is public (RFC 8628 section 3.1).
DEFAULT_GITHUB_CLIENT_ID = 'Ov23liw7qPBHHzUfjfZA'

GITHUB_CLIENT_ID_ENV = 'POSITRONIC_PLATFORM_GITHUB_CLIENT_ID'
WORKOS_CLIENT_ID_ENV = 'POSITRONIC_PLATFORM_WORKOS_CLIENT_ID'

# The OAuth spelling of each field that more than one step writes or reads. Those steps have to agree,
# so each is named once.
_CLIENT_ID_FIELD = 'client_id'
_DEVICE_CODE_FIELD = 'device_code'
_ERROR_FIELD = 'error'
_INTERVAL_FIELD = 'interval'

DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_EXPIRES_IN_S = 900.0


class DeviceFlowError(Exception):
    """The flow ended without a token: the user or the provider refused it, or the code ran out."""


class ProviderFailed(DeviceFlowError):
    """The identity provider is unreachable, unavailable, or answered something this code cannot read."""


class _TransientFailure(ProviderFailed):
    """One request failed in transport or timed out, or the provider answered 429 or 5xx."""

    def __init__(self, message: str, *, reached_the_provider: bool, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        # A request that reached the provider may have spent the device code before its answer was lost.
        self.reached_the_provider = reached_the_provider
        self.retry_after_s = retry_after_s


class NoSignIn(DeviceFlowError):
    """The platform takes no sign-in that this client can run."""


@dataclass(frozen=True)
class DeviceAuthorization:
    """What step one returns: the code the user types, and the code this flow polls with."""

    device_code: str
    user_code: str
    verification_uri: str
    interval: float
    expires_in: float


class PollOutcome(Enum):
    """What one poll of the token endpoint means for the flow (RFC 8628 §3.5)."""

    GRANTED = auto()
    PENDING = auto()
    SLOW_DOWN = auto()
    DENIED = auto()
    EXPIRED = auto()
    UNREADABLE = auto()


class DeviceFlow:
    """The device of RFC 8628: it asks the provider for a code, and it polls until the provider answers."""

    def __init__(
        self,
        provider: IdentityProvider,
        client_id: str,
        http: httpx.Client,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.client_id = client_id
        self._http = http
        self._monotonic = monotonic
        self._sleep = sleep

    def _require_str(self, payload: Mapping[str, object], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ProviderFailed(f'{self.provider.name} answered without {field}')
        return value

    def _optional_duration_s(self, payload: Mapping[str, object], field: str, default: float) -> float:
        """Seconds the provider may omit. Absent takes the default; present must be above zero and sleepable.

        A negative or NaN value raises inside `time.sleep`, a zero one polls until the code expires, and a
        value past a day overflows `time.sleep` or `float` itself.
        """
        longest_s = 24 * 60 * 60  # a device code lives minutes (RFC 8628 gives 15 min as the example)
        if field not in payload:
            return default
        value = payload[field]
        # `bool` is an `int`, so `True` would pass as an interval of one second.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProviderFailed(f'{self.provider.name} answered with an unreadable {field}')
        try:
            seconds = float(value)
        except OverflowError:  # an integer past what a float holds is as unsleepable as infinity
            seconds = math.inf
        if not math.isfinite(seconds) or seconds <= 0 or seconds > longest_s:
            raise ProviderFailed(f'{self.provider.name} answered with {field}={seconds}, which is no length of time')
        return seconds

    def start_device_authorization(self) -> DeviceAuthorization:
        """Step one: ask the provider for a device code, and for the short code the user types."""
        form = {_CLIENT_ID_FIELD: self.client_id}
        if self.provider.scope is not None:
            form['scope'] = self.provider.scope
        payload = self._post_form(self.provider.device_authorization_url, form)
        error = payload.get(_ERROR_FIELD)
        if error is not None:
            raise DeviceFlowError(f'{self.provider.name} refused the device-code request: {error}')
        return DeviceAuthorization(
            device_code=self._require_str(payload, _DEVICE_CODE_FIELD),
            user_code=self._require_str(payload, 'user_code'),
            verification_uri=self._require_str(payload, 'verification_uri'),
            interval=self._optional_duration_s(payload, _INTERVAL_FIELD, DEFAULT_POLL_INTERVAL_S),
            expires_in=self._optional_duration_s(payload, 'expires_in', DEFAULT_EXPIRES_IN_S),
        )

    @staticmethod
    def _outcome_of(error: object) -> PollOutcome:
        """Read the `error` field. An absent one means the access token is in the answer."""
        if error is None:
            return PollOutcome.GRANTED
        # The RFC 8628 spelling of each outcome. The set of errors a provider may send is open, so
        # anything else is a request it could not read.
        outcomes = {
            'authorization_pending': PollOutcome.PENDING,
            'slow_down': PollOutcome.SLOW_DOWN,
            'access_denied': PollOutcome.DENIED,
            'expired_token': PollOutcome.EXPIRED,
        }
        return outcomes.get(str(error), PollOutcome.UNREADABLE)

    def _terminal_failure(
        self, outcome: PollOutcome, error: object, reached_before: _TransientFailure | None
    ) -> DeviceFlowError:
        """The failure a terminal poll answer ends the flow with. UNREADABLE carries the provider's own word."""
        terminal_messages = {
            PollOutcome.DENIED: 'the user refused the authorization',
            PollOutcome.EXPIRED: 'the device code expired before it was authorized',
        }
        message = terminal_messages.get(outcome, f'{self.provider.name} refused the device code: {error}')
        # A failed poll that reached the provider may have spent the code, and its token was lost in transit.
        if outcome in (PollOutcome.UNREADABLE, PollOutcome.EXPIRED) and reached_before is not None:
            failure = ProviderFailed(f'{message}, after a poll that failed: {reached_before}')
            failure.__cause__ = reached_before
            return failure
        return DeviceFlowError(message)

    def _poll(self, authorization: DeviceAuthorization) -> Mapping[str, object]:
        """One request to the token endpoint."""
        return self._post_form(
            self.provider.token_url,
            {
                _CLIENT_ID_FIELD: self.client_id,
                _DEVICE_CODE_FIELD: authorization.device_code,
                'grant_type': DEVICE_GRANT_TYPE,
            },
        )

    def poll_for_token(self, authorization: DeviceAuthorization) -> str:
        """Step two: poll the device code until the provider mints the access token, or until it expires.

        The access token is all this returns. Any other token in the answer is dropped here.
        """
        # How much a `slow_down` answer or a transient failure raises the poll interval (RFC 8628 §3.5).
        slow_down_step_s = 5.0
        interval = authorization.interval
        # The next wait: the interval, or once the longer wait a 429 asked for.
        wait = interval
        deadline = self._monotonic() + authorization.expires_in
        last_failure: _TransientFailure | None = None
        # The first failed poll since the last answer that reached the provider. A later failure keeps it.
        reached_before: _TransientFailure | None = None
        while True:
            # RFC 8628 §3.3: the device waits the interval before each poll, the first one included.
            # A poll that starts at once is answered `slow_down`, which spends the code's lifetime.
            if self._monotonic() + wait > deadline:
                if last_failure is not None:
                    # The wait the provider asked for outlives the code, which the user may have approved.
                    raise ProviderFailed(f'{last_failure}, and the device code expires first') from last_failure
                raise DeviceFlowError('the device code was not authorized in time')
            self._sleep(wait)
            try:
                payload = self._poll(authorization)
            except _TransientFailure as exc:
                last_failure = exc
                if reached_before is None and exc.reached_the_provider:
                    reached_before = exc
                # RFC 8628 §3.5: a poll that timed out slows the rate and tries again, and WorkOS names a
                # 429 and a 5xx transient as well. The code may already be authorized, and the deadline
                # above ends the flow.
                interval += slow_down_step_s
                wait = max(interval, exc.retry_after_s or 0.0)
                continue
            last_failure = None
            error = payload.get(_ERROR_FIELD)
            outcome = self._outcome_of(error)
            if outcome is PollOutcome.GRANTED:
                return self._require_str(payload, 'access_token')
            if outcome is PollOutcome.SLOW_DOWN:
                # GitHub repeats the original interval in this answer, so the step raises the rate;
                # taking the value alone would poll on at the rate the provider just refused.
                interval = max(
                    interval + slow_down_step_s, self._optional_duration_s(payload, _INTERVAL_FIELD, interval)
                )
            elif outcome is not PollOutcome.PENDING:
                raise self._terminal_failure(outcome, error, reached_before)
            reached_before = None
            wait = interval

    @staticmethod
    def _retry_after_s(response: httpx.Response) -> float | None:
        """The wait a 429 names in whole seconds, or None. The HTTP-date form is left to the default step."""
        value = response.headers.get('retry-after', '')
        return float(value) if value.isascii() and value.isdigit() else None

    def _post_form(self, url: str, data: Mapping[str, str]) -> Mapping[str, object]:
        """POST a form and read the JSON answer, an OAuth error included.

        GitHub reports a device-flow error as HTTP 200 with an `error` field, and WorkOS as HTTP 400
        (RFC 6749 §5.2 also allows 401), so the body says whether the flow is still running.
        """
        name = self.provider.name
        try:
            response = self._http.request(
                'POST', url, data=dict(data), headers={'accept': 'application/json'}, timeout=REQUEST_TIMEOUT
            )
        except httpx.TimeoutException as exc:
            reached = not isinstance(exc, httpx.ConnectTimeout | httpx.PoolTimeout)
            raise _TransientFailure(f'{name} did not answer in time', reached_the_provider=reached) from exc
        except httpx.TransportError as exc:
            reached = not isinstance(exc, httpx.ConnectError)
            raise _TransientFailure(f'{name} is unreachable', reached_the_provider=reached) from exc
        except httpx.HTTPError as exc:
            raise ProviderFailed(f'{name} did not answer: {exc}') from exc
        status = response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS:
            retry_after_s = self._retry_after_s(response)
            wait = f'; it asks for {retry_after_s:g} s' if retry_after_s is not None else ''
            raise _TransientFailure(
                f'{name} refused the request rate{wait}', reached_the_provider=False, retry_after_s=retry_after_s
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise _TransientFailure(f'{name} is unavailable: HTTP {status}', reached_the_provider=True)
        carries_oauth_error = status in (httpx.codes.BAD_REQUEST, httpx.codes.UNAUTHORIZED)
        if status >= 400 and not carries_oauth_error:
            raise ProviderFailed(f'{name} is unavailable: HTTP {status}')
        try:
            payload: object = response.json()
        except ValueError as exc:
            if carries_oauth_error:
                raise ProviderFailed(f'{name} is unavailable: HTTP {status}') from exc
            raise ProviderFailed(f'{name} answered with no JSON') from exc
        if carries_oauth_error and not (isinstance(payload, dict) and _ERROR_FIELD in payload):
            raise ProviderFailed(f'{name} is unavailable: HTTP {status}')
        if not isinstance(payload, dict):
            raise ProviderFailed(f'{name} answered with no JSON object')
        return payload


@dataclass(frozen=True)
class ClientIds:
    """The clients a sign-in runs through. `workos` replaces the WorkOS client the platform names; None takes it."""

    github: str = DEFAULT_GITHUB_CLIENT_ID
    workos: str | None = None

    def __post_init__(self) -> None:
        # A provider answers an empty id with an error that names nothing, so an empty override stops here.
        if not self.github:
            raise ValueError(
                f'the GitHub client id is empty ({GITHUB_CLIENT_ID_ENV} or its flag): unset it for the default'
            )
        if self.workos == '':
            raise ValueError(
                f'the WorkOS client id is empty ({WORKOS_CLIENT_ID_ENV} or its flag): '
                'unset it for the one the platform names'
            )

    @classmethod
    def resolve(cls, environ: Mapping[str, str], *, github: str | None = None, workos: str | None = None) -> ClientIds:
        """Each id from its flag, else from its environment variable, else the default."""
        return cls(
            github=environ.get(GITHUB_CLIENT_ID_ENV, DEFAULT_GITHUB_CLIENT_ID) if github is None else github,
            workos=environ.get(WORKOS_CLIENT_ID_ENV) if workos is None else workos,
        )


@dataclass(frozen=True)
class SignIn:
    """A device flow, and the bound on the `users.register` that takes its token."""

    flow: DeviceFlow
    register_timeout: httpx.Timeout


def sign_in_flow(platform: PlatformClient, http: httpx.Client, client_ids: ClientIds) -> SignIn:
    """The sign-in the platform's `auth.config` names."""
    config = platform.auth_config()
    if config.workos is not None:
        client_id = config.workos.client_id if client_ids.workos is None else client_ids.workos
        return SignIn(DeviceFlow(WORKOS, client_id, http), workos_register_timeout(config.workos))
    if config.github_registration:
        return SignIn(DeviceFlow(GITHUB, client_ids.github, http), GITHUB_REGISTER_TIMEOUT)
    raise NoSignIn(f'the platform at {platform.base_url} takes no sign-in that this client can run')


def may_have_minted_a_key(exc: httpx.HTTPError) -> bool:
    """Whether a failed request reached `users.register`, which may have minted a key whose answer is lost."""
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout | httpx.PoolTimeout):
        return False
    return exc.request.url.path.endswith(routes.USERS_REGISTER)


def register_with_device_flow(
    platform: PlatformClient, sign_in: SignIn, *, alias: str | None = None, rotate: bool = False
) -> RegisterResponse:
    """Run the whole flow: show the code, poll for the token, register with it as the credential."""
    flow = sign_in.flow
    authorization = flow.start_device_authorization()
    # To stderr: it is a prompt for the person at the terminal, and stdout carries the command's answer.
    print(f'open {authorization.verification_uri} and enter {authorization.user_code}', file=sys.stderr, flush=True)
    token = flow.poll_for_token(authorization)
    request = RegisterRequest(credential=token, alias=alias, rotate=rotate)
    return platform.register(request, timeout=sign_in.register_timeout)


def platform_url_is_allowed(base_url: str, *, plaintext_http: bool) -> bool:
    """https anywhere, http to a loopback address, and any other http only under `plaintext_http`.

    An http platform that is not loopback shows the sign-in token to every host on the path, so the
    caller names the link it trusts.
    """
    url = httpx.URL(base_url)
    if url.scheme == 'https':
        return True
    if url.scheme != 'http':
        return False
    if plaintext_http:
        return True
    if url.host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(url.host).is_loopback
    except ValueError:
        return False


def require_allowed_platform_url(base_url: str, *, plaintext_http: bool, flag: str) -> None:
    """Refuse, with a `ValueError` naming `flag`, a platform URL that would carry the token in the clear."""
    if not platform_url_is_allowed(base_url, plaintext_http=plaintext_http):
        raise ValueError(
            f'{base_url} would carry the sign-in token in the clear: '
            f'use https, a loopback address, or pass {flag} on a link you trust'
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags a registration takes."""
    parser.add_argument('--alias', default=None, help='the display name for this account')
    parser.add_argument(
        '--rotate', action='store_true', help='mint a new key for an account that is already registered'
    )
    parser.add_argument('--platform-url', default=None, help=f'the platform to register with, else {API_URL_ENV}')
    parser.add_argument(
        '--plaintext-http', action='store_true', help='reach an http platform that is not loopback, on a link you trust'
    )
    parser.add_argument(
        '--client-id',
        default=None,
        help=f'the GitHub OAuth app to register through, else {GITHUB_CLIENT_ID_ENV}, else the platform app',
    )
    parser.add_argument(
        '--workos-client-id',
        default=None,
        help=f'the WorkOS client to register through, else {WORKOS_CLIENT_ID_ENV}, else the one the platform names',
    )


def allowed_platform_url(platform_url: str | None, *, plaintext_http: bool) -> str:
    """The platform to register with, or a `SystemExit` naming why the URL cannot carry the token."""
    try:
        base_url = resolve_base_url(platform_url)
        require_allowed_platform_url(base_url, plaintext_http=plaintext_http, flag='--plaintext-http')
    except (ValueError, httpx.InvalidURL) as exc:
        # An empty, relative, malformed or cleartext platform URL is an argument fault, so it reads like the line above.
        raise SystemExit(str(exc)) from exc
    return base_url


def run_registration(
    platform_url: str | None, *, client_ids: ClientIds, alias: str | None, rotate: bool, plaintext_http: bool = False
) -> RegisterResponse:
    """Register through the device flow the platform names, or a `SystemExit` with one line saying what stopped it.

    This resolves the platform itself, so no caller can reach the transport with a URL the gate
    would refuse: the token goes to the link `plaintext_http` names, or to none.
    """
    base_url = allowed_platform_url(platform_url, plaintext_http=plaintext_http)
    with httpx.Client(timeout=REQUEST_TIMEOUT) as provider_http, PlatformClient(base_url) as platform:
        try:
            sign_in = sign_in_flow(platform, provider_http, client_ids)
            return register_with_device_flow(platform, sign_in, alias=alias, rotate=rotate)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # The provider half wraps its own transport failures, so one arriving here dialled the
            # gateway. httpx names the cause and not the host, which is the half a reader needs.
            raise SystemExit(f'the platform at {base_url} is unreachable: {exc}') from exc
        except httpx.HTTPError as exc:
            if not may_have_minted_a_key(exc):
                raise SystemExit(f'the platform at {base_url} did not answer: {exc}') from exc
            # The request may have reached the gateway and minted the key before the answer was lost.
            raise SystemExit(
                f'the platform at {base_url} did not answer the registration: {exc}. '
                'A key may have been minted, and a WorkOS token registers once: run again with --rotate, '
                'which signs in again.'
            ) from exc
        except (DeviceFlowError, PlatformError) as exc:
            raise SystemExit(str(exc)) from exc
        except ValidationError as exc:
            # A 2xx whose body is not what the route answers: a proxy page, or a gateway of another version.
            raise SystemExit(
                f'the platform at {base_url} answered with a body this client cannot read: '
                f'{exc.error_count()} field(s) off'
            ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        client_ids = ClientIds.resolve(os.environ, github=args.client_id, workos=args.workos_client_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    response = run_registration(
        args.platform_url,
        client_ids=client_ids,
        alias=args.alias,
        rotate=args.rotate,
        plaintext_http=args.plaintext_http,
    )
    print(f'user {response.user_id} ({response.key_status.name})')
    print(f'artifacts at {response.artifact_location}')
    if response.api_key is None:
        print('no key issued: one is minted on a first registration, or by --rotate')
    else:
        # The key is opaque, so it may hold whitespace or shell metacharacters; an unquoted export
        # line would either mangle it or run the rest of it.
        print(f'export {API_KEY_ENV}={shlex.quote(response.api_key)}')


if __name__ == '__main__':
    main()
