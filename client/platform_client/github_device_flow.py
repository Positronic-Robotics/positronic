"""GitHub's device flow, and the `platform-register` command that runs it.

It asks GitHub for a device code, shows the user the short code, polls until GitHub answers, and
hands the access token to `users.register` as its `credential`.

Usage
  export POSITRONIC_PLATFORM_GITHUB_CLIENT_ID=<the OAuth app's public client id>
  platform-register --alias='<display name>'
  platform-register --client-id=<id> --platform-url=http://127.0.0.1:8080
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

DEVICE_CODE_URL = 'https://github.com/login/device/code'
ACCESS_TOKEN_URL = 'https://github.com/login/oauth/access_token'

# The grant the token request names (RFC 8628 §3.4).
DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'

# The scope the minted token carries: the profile and the verified email. It reads no repository and
# writes nothing, so the platform holds no power over the account.
IDENTITY_SCOPE = 'read:user user:email'

# The platform's OAuth app id. The device flow carries no client secret, so the id is public.
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
        """A number of seconds GitHub may omit. Absent takes the default; present must be finite and above zero.

        A negative or NaN value raises inside `time.sleep`, and a zero one polls until the code expires.
        """
        if field not in payload:
            return default
        value = payload[field]
        # `bool` is an `int`, so `True` would pass as an interval of one second.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DeviceFlowError(f'GitHub answered with an unreadable {field}')
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
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
            payload = self._post_form(
                ACCESS_TOKEN_URL,
                {
                    _CLIENT_ID_FIELD: self._client_id,
                    _DEVICE_CODE_FIELD: authorization.device_code,
                    'grant_type': DEVICE_GRANT_TYPE,
                },
            )
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


# A tailnet address is as private as loopback: the WireGuard link encrypts what `http://` sends
# over it, and staging is reached that way with no TLS of its own (services/ops/staging).
_TAILNET = ipaddress.ip_network('100.64.0.0/10')


def token_travels_encrypted(base_url: str) -> bool:
    """https anywhere, or http to loopback or a tailnet address. Any other http shows the token."""
    url = httpx.URL(base_url)
    if url.scheme == 'https':
        return True
    if url.scheme != 'http':
        return False
    try:
        address = ipaddress.ip_address(url.host)
    except ValueError:
        return url.host == 'localhost'
    return address.is_loopback or (address.version == 4 and address in _TAILNET)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--alias', default=None, help='the display name for this account')
    parser.add_argument(
        '--rotate', action='store_true', help='mint a new key for an account that is already registered'
    )
    parser.add_argument('--platform-url', default=None, help=f'the platform to register with, else {API_URL_ENV}')
    parser.add_argument('--client-id', default=os.environ.get(GITHUB_CLIENT_ID_ENV), help=GITHUB_CLIENT_ID_ENV)
    args = parser.parse_args()
    if not args.client_id:
        raise SystemExit(f'pass --client-id, or set {GITHUB_CLIENT_ID_ENV} to the public OAuth client id')
    try:
        base_url = resolve_base_url(args.platform_url)
    except ValueError as exc:
        # An empty or relative platform URL is an argument fault, so it reads like the line above.
        raise SystemExit(str(exc)) from exc
    if not token_travels_encrypted(base_url):
        raise SystemExit(
            f'{base_url} would carry the GitHub token in the clear: use https, or a loopback or tailnet address'
        )

    github = httpx.Client(timeout=REQUEST_TIMEOUT)
    gateway = httpx.Client(base_url=base_url, timeout=REGISTER_TIMEOUT)
    with github, gateway, PlatformClient(client=gateway) as platform:
        try:
            response = register_with_github(
                platform, GitHubDeviceFlow(args.client_id, github), alias=args.alias, rotate=args.rotate
            )
        except httpx.HTTPError as exc:
            # The GitHub half wraps its own transport failures, so one arriving here dialled the
            # gateway. httpx names the cause and not the host, which is the half a reader needs.
            raise SystemExit(f'the platform at {base_url} is unreachable: {exc}') from exc
        except (DeviceFlowError, PlatformError) as exc:
            raise SystemExit(str(exc)) from exc
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
