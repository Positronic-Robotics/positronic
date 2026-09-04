"""GitHub's device flow, and the `platform-register` command that runs it.

It asks GitHub for a device code, shows the user the short code, polls until GitHub answers, and
hands the access token to `users.register` as its `credential`.

Usage
  platform-register --alias='<display name>'
  platform-register --client-id=<id> --platform-url=http://127.0.0.1:8080
  platform-register --platform-url=http://staging.internal:8080 --plaintext-http  # a plain http link
  platform-register --rotate  # mint a new key for an account that is already registered
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import os
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto

import httpx
from platform_client.client import API_KEY_ENV, API_URL_ENV, PlatformClient, resolve_base_url
from platform_client.errors import PlatformError
from platform_client.requests import RegisterRequest
from platform_client.responses import RegisterResponse
from pydantic import ValidationError

DEVICE_CODE_URL = 'https://github.com/login/device/code'
ACCESS_TOKEN_URL = 'https://github.com/login/oauth/access_token'

# The grant the token request names (RFC 8628 §3.4).
DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'

# The scope the minted token carries: the profile and the verified email. It reads no repository and
# writes nothing, so the platform holds no power over the account.
IDENTITY_SCOPE = 'read:user user:email'

# The platform's OAuth app `positronic-platform-device-flow`. A device flow carries no client
# secret, so its client id is public (RFC 8628 section 3.1).
DEFAULT_GITHUB_CLIENT_ID = 'Ov23liw7qPBHHzUfjfZA'

GITHUB_CLIENT_ID_ENV = 'POSITRONIC_PLATFORM_GITHUB_CLIENT_ID'

# GitHub's spelling of every field this flow reads or writes. The step that writes one and the step
# that reads it have to agree, so each is named once.
_CLIENT_ID_FIELD = 'client_id'
_DEVICE_CODE_FIELD = 'device_code'
_ERROR_FIELD = 'error'
_INTERVAL_FIELD = 'interval'

DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_EXPIRES_IN_S = 900.0


class DeviceFlowError(Exception):
    """GitHub refused the flow, answered something this code cannot read, or is unreachable."""


class _RequestTimedOut(DeviceFlowError):
    """One request to GitHub timed out. `poll_for_token` retries it; every other caller sees a DeviceFlowError."""


# One request's phases. httpx bounds inactivity inside each one separately, never the whole request.
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 10.0
WRITE_TIMEOUT_S = 5.0
POOL_TIMEOUT_S = 5.0

REQUEST_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S, write=WRITE_TIMEOUT_S, pool=POOL_TIMEOUT_S
)


def max_stall_s(timeout: httpx.Timeout) -> float:
    """The four phase timeouts added up: the longest a request can go with no byte in any one phase.

    Each phase timeout bounds inactivity inside that phase, so a slow but steady response outlives
    this sum. A phase left unset bounds no stall at all, so it refuses.
    """
    connect, read, write, pool = timeout.connect, timeout.read, timeout.write, timeout.pool
    if connect is None or read is None or write is None or pool is None:
        raise ValueError(f'{timeout} leaves a phase unbounded, so a request stalled in it never fails')
    return connect + read + write + pool


# The longest one GitHub request can stall, read off the phases so the two cannot drift.
MAX_REQUEST_STALL_S = max_stall_s(REQUEST_TIMEOUT)

# The gateway's own GitHub reads inside `users.register`: the account, plus the addresses of an
# account whose address is private. Neither distribution imports the other, so this value is a copy.
GATEWAY_GITHUB_READS = 2

# The gateway verifies inside the registration call, so the read phase covers its GitHub reads plus
# one request for its own work. A caller that gives up first leaves behind a user and a key it can
# never read.
REGISTER_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S,
    read=GATEWAY_GITHUB_READS * MAX_REQUEST_STALL_S + MAX_REQUEST_STALL_S,
    write=WRITE_TIMEOUT_S,
    pool=POOL_TIMEOUT_S,
)


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


class GitHubDeviceFlow:
    """The device of RFC 8628: it asks for a code, and it polls until GitHub answers."""

    def __init__(
        self,
        client_id: str,
        http: httpx.Client,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client_id = client_id
        self._http = http
        self._monotonic = monotonic
        self._sleep = sleep

    @staticmethod
    def _require_str(payload: Mapping[str, object], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise DeviceFlowError(f'GitHub answered without {field}')
        return value

    @staticmethod
    def _optional_duration_s(payload: Mapping[str, object], field: str, default: float) -> float:
        """A number of seconds GitHub may omit. Absent takes the default; present must be above zero and sleepable.

        A negative or NaN value raises inside `time.sleep`, a zero one polls until the code expires, and a
        value past a day overflows `time.sleep` or `float` itself.
        """
        longest_s = 24 * 60 * 60  # a device code lives minutes (RFC 8628 gives 15 min as the example)
        if field not in payload:
            return default
        value = payload[field]
        # `bool` is an `int`, so `True` would pass as an interval of one second.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DeviceFlowError(f'GitHub answered with an unreadable {field}')
        try:
            seconds = float(value)
        except OverflowError:  # an integer past what a float holds is as unsleepable as infinity
            seconds = math.inf
        if not math.isfinite(seconds) or seconds <= 0 or seconds > longest_s:
            raise DeviceFlowError(f'GitHub answered with {field}={seconds}, which is no length of time')
        return seconds

    def start_device_authorization(self) -> DeviceAuthorization:
        """Step one: ask GitHub for a device code, and for the short code the user types."""
        payload = self._post_form(DEVICE_CODE_URL, {_CLIENT_ID_FIELD: self._client_id, 'scope': IDENTITY_SCOPE})
        error = payload.get(_ERROR_FIELD)
        if error is not None:
            raise DeviceFlowError(f'GitHub refused the device-code request: {error}')
        return DeviceAuthorization(
            device_code=self._require_str(payload, _DEVICE_CODE_FIELD),
            user_code=self._require_str(payload, 'user_code'),
            verification_uri=self._require_str(payload, 'verification_uri'),
            interval=self._optional_duration_s(payload, _INTERVAL_FIELD, DEFAULT_POLL_INTERVAL_S),
            expires_in=self._optional_duration_s(payload, 'expires_in', DEFAULT_EXPIRES_IN_S),
        )

    @staticmethod
    def _outcome_of(error: object) -> PollOutcome:
        """Read GitHub's `error` field. An absent one means the access token is in the answer."""
        if error is None:
            return PollOutcome.GRANTED
        # GitHub's spelling of each outcome. The set of errors it may send is open, so anything else
        # is a request it could not read.
        outcomes = {
            'authorization_pending': PollOutcome.PENDING,
            'slow_down': PollOutcome.SLOW_DOWN,
            'access_denied': PollOutcome.DENIED,
            'expired_token': PollOutcome.EXPIRED,
        }
        return outcomes.get(str(error), PollOutcome.UNREADABLE)

    def poll_for_token(self, authorization: DeviceAuthorization) -> str:
        """Step two: poll the device code until GitHub mints the access token, or until it expires."""
        # How much a `slow_down` answer raises the poll interval (RFC 8628 §3.5).
        slow_down_step_s = 5.0
        # What each terminal outcome leaves the caller with. UNREADABLE carries GitHub's own word.
        terminal_messages = {
            PollOutcome.DENIED: 'the user refused the authorization',
            PollOutcome.EXPIRED: 'the device code expired before it was authorized',
        }
        interval = authorization.interval
        deadline = self._monotonic() + authorization.expires_in
        while True:
            # RFC 8628 §3.3: the device waits the interval before each poll, the first one included.
            # A poll that starts at once is answered `slow_down`, which spends the code's lifetime.
            if self._monotonic() + interval > deadline:
                raise DeviceFlowError('the device code was not authorized in time')
            self._sleep(interval)
            try:
                payload = self._post_form(
                    ACCESS_TOKEN_URL,
                    {
                        _CLIENT_ID_FIELD: self._client_id,
                        _DEVICE_CODE_FIELD: authorization.device_code,
                        'grant_type': DEVICE_GRANT_TYPE,
                    },
                )
            except _RequestTimedOut:
                # RFC 8628 §3.5: a poll that timed out slows the rate and tries again; the code may
                # already be authorized, and the deadline above ends the flow.
                interval += slow_down_step_s
                continue
            error = payload.get(_ERROR_FIELD)
            outcome = self._outcome_of(error)
            if outcome is PollOutcome.GRANTED:
                return self._require_str(payload, 'access_token')
            if outcome is PollOutcome.SLOW_DOWN:
                # GitHub repeats the original interval in this answer, so the step raises the rate;
                # taking the value alone would poll on at the rate GitHub just refused.
                interval = max(
                    interval + slow_down_step_s, self._optional_duration_s(payload, _INTERVAL_FIELD, interval)
                )
            elif outcome is not PollOutcome.PENDING:
                raise DeviceFlowError(terminal_messages.get(outcome, f'GitHub refused the device code: {error}'))

    def _post_form(self, url: str, data: Mapping[str, str]) -> Mapping[str, object]:
        """POST a form and read the JSON answer.

        GitHub reports a device-flow error as HTTP 200 with an `error` field, so the body says
        whether the flow is still running.
        """
        try:
            response = self._http.request(
                'POST', url, data=dict(data), headers={'accept': 'application/json'}, timeout=REQUEST_TIMEOUT
            )
        except httpx.TimeoutException as exc:
            raise _RequestTimedOut('GitHub did not answer in time') from exc
        except httpx.HTTPError as exc:
            raise DeviceFlowError('GitHub is unreachable') from exc
        if response.status_code >= 400:
            raise DeviceFlowError(f'GitHub is unavailable: HTTP {response.status_code}')
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise DeviceFlowError('GitHub answered with no JSON') from exc
        if not isinstance(payload, dict):
            raise DeviceFlowError('GitHub answered with no JSON object')
        return payload


def register_with_github(
    platform: PlatformClient, flow: GitHubDeviceFlow, *, alias: str | None = None, rotate: bool = False
) -> RegisterResponse:
    """Run the whole flow: show the code, poll for the token, register with it as the credential."""
    authorization = flow.start_device_authorization()
    print(f'open {authorization.verification_uri} and enter {authorization.user_code}', flush=True)
    token = flow.poll_for_token(authorization)
    return platform.register(RegisterRequest(credential=token, alias=alias, rotate=rotate))


def platform_url_is_allowed(base_url: str, *, plaintext_http: bool) -> bool:
    """https anywhere, http to a loopback address, and any other http only under `plaintext_http`.

    An http platform that is not loopback shows the GitHub token to every host on the path, so the
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        default=os.environ.get(GITHUB_CLIENT_ID_ENV, DEFAULT_GITHUB_CLIENT_ID),
        help=f'the OAuth app to register through, else {GITHUB_CLIENT_ID_ENV}, else the platform app',
    )
    args = parser.parse_args()
    if not args.client_id:
        # GitHub answers an empty id with an error that names nothing, so an empty override stops here.
        raise SystemExit(f'--client-id or {GITHUB_CLIENT_ID_ENV} is empty. Unset it to use the default.')
    try:
        base_url = resolve_base_url(args.platform_url)
        allowed = platform_url_is_allowed(base_url, plaintext_http=args.plaintext_http)
    except (ValueError, httpx.InvalidURL) as exc:
        # An empty, relative or malformed platform URL is an argument fault, so it reads like the line above.
        raise SystemExit(str(exc)) from exc
    if not allowed:
        raise SystemExit(
            f'{base_url} would carry the GitHub token in the clear: '
            'use https, a loopback address, or pass --plaintext-http on a link you trust'
        )

    github = httpx.Client(timeout=REQUEST_TIMEOUT)
    # An environment proxy would carry a plain-http token off the machine, past the URL gate above.
    gateway = httpx.Client(base_url=base_url, timeout=REGISTER_TIMEOUT, trust_env=httpx.URL(base_url).scheme == 'https')
    with github, gateway, PlatformClient(client=gateway) as platform:
        try:
            response = register_with_github(
                platform, GitHubDeviceFlow(args.client_id, github), alias=args.alias, rotate=args.rotate
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # The GitHub half wraps its own transport failures, so one arriving here dialled the
            # gateway. httpx names the cause and not the host, which is the half a reader needs.
            raise SystemExit(f'the platform at {base_url} is unreachable: {exc}') from exc
        except httpx.HTTPError as exc:
            # The request may have reached the gateway and minted the key before the answer was lost.
            raise SystemExit(
                f'the platform at {base_url} did not answer the registration: {exc}. '
                'A key may have been minted; run again with --rotate.'
            ) from exc
        except (DeviceFlowError, PlatformError) as exc:
            raise SystemExit(str(exc)) from exc
        except ValidationError as exc:
            # A 2xx whose body is not a registration: a proxy page, or a gateway of another version.
            raise SystemExit(
                f'the platform at {base_url} answered with no registration: {exc.error_count()} field(s) off'
            ) from exc
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
