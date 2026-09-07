"""The user's half of the GitHub device flow, driven against a scripted transport.

No test reaches GitHub. An injected sleep records the intervals and advances a fake monotonic clock,
so the poll runs at full speed and the suite waits on nothing.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import github_device_flow, routes
from platform_client.client import PlatformClient
from platform_client.github_device_flow import (
    ACCESS_TOKEN_URL,
    CONNECT_TIMEOUT_S,
    DEFAULT_GITHUB_CLIENT_ID,
    DEVICE_CODE_URL,
    DEVICE_GRANT_TYPE,
    GATEWAY_GITHUB_READS,
    GITHUB_CLIENT_ID_ENV,
    IDENTITY_SCOPE,
    MAX_REQUEST_STALL_S,
    POOL_TIMEOUT_S,
    REGISTER_TIMEOUT,
    REQUEST_TIMEOUT,
    WRITE_TIMEOUT_S,
    DeviceAuthorization,
    DeviceFlowError,
    GitHubDeviceFlow,
    max_stall_s,
    register_with_github,
)
from platform_client.responses import RegisterResponse

CLIENT_ID = 'Iv1.testclientid'
DEVICE_CODE = 'dev-code-1'
TOKEN = 'gho_token'
DEVICE_ANSWER: dict[str, object] = {
    'device_code': DEVICE_CODE,
    'user_code': 'WDJB-MJHT',
    'verification_uri': 'https://github.com/login/device',
    'interval': 5,
    'expires_in': 900,
}
GRANTED = {'access_token': TOKEN, 'token_type': 'bearer', 'scope': IDENTITY_SCOPE}

# rules-allow: hardcoded-keys — `positronic/cli/conftest.py` holds the same value, and this package
# may not import from `positronic`: the dependency runs the other way, and the client installs alone.
MINTED_KEY = 'pk_live_secret'


@dataclass
class ScriptedGitHub:
    """One scripted answer per token poll, taken in order, plus the device-code endpoint."""

    polls: list[dict[str, object]] = field(default_factory=list)
    device: dict[str, object] = field(default_factory=lambda: dict(DEVICE_ANSWER))
    status: int = 200  # every endpoint's HTTP status; the flow reports its own errors in the body
    requests: list[httpx.Request] = field(default_factory=list)
    # Every wait and every poll, in the order they happened: the order is what RFC 8628 §3.3 fixes.
    events: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.events.append('poll')
        url = str(request.url)
        if url == DEVICE_CODE_URL:
            return httpx.Response(self.status, json=self.device)
        if url == ACCESS_TOKEN_URL:
            return httpx.Response(self.status, json=self.polls.pop(0))
        raise AssertionError(f'the flow reached an endpoint it has no business at: {url}')

    def form(self, url: str) -> dict[str, str]:
        """The last form this endpoint received, read back off the recorded request."""
        sent = [r for r in self.requests if str(r.url) == url]
        return dict(httpx.QueryParams(sent[-1].content.decode()))


@dataclass
class RecordedSleep:
    """A sleep the test reads back, advancing the clock the poll measures its deadline against."""

    events: list[str]
    seconds: list[float] = field(default_factory=list)
    now: float = 0.0

    def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)
        self.events.append(f'wait {seconds}')
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _flow(github: ScriptedGitHub) -> tuple[GitHubDeviceFlow, RecordedSleep]:
    sleep = RecordedSleep(github.events)
    http = httpx.Client(transport=httpx.MockTransport(github.handle))
    return GitHubDeviceFlow(CLIENT_ID, http, monotonic=sleep.monotonic, sleep=sleep), sleep


def _authorization(*, interval: float = 5.0, expires_in: float = 900.0) -> DeviceAuthorization:
    return DeviceAuthorization(
        device_code=DEVICE_CODE,
        user_code=str(DEVICE_ANSWER['user_code']),
        verification_uri=str(DEVICE_ANSWER['verification_uri']),
        interval=interval,
        expires_in=expires_in,
    )


# --- step one: the device code ------------------------------------------------


def test_the_device_code_request_asks_for_identity_scope_only():
    github = ScriptedGitHub()
    flow, _ = _flow(github)

    authorization = flow.start_device_authorization()

    assert github.form(DEVICE_CODE_URL) == {'client_id': CLIENT_ID, 'scope': IDENTITY_SCOPE}
    assert authorization.device_code == DEVICE_CODE
    assert authorization.user_code == DEVICE_ANSWER['user_code']
    assert authorization.verification_uri == DEVICE_ANSWER['verification_uri']
    assert authorization.interval == 5.0 and authorization.expires_in == 900.0


def test_a_device_code_answer_without_an_interval_polls_at_the_default_rate():
    github = ScriptedGitHub(device={k: v for k, v in DEVICE_ANSWER.items() if k not in ('interval', 'expires_in')})
    flow, _ = _flow(github)

    authorization = flow.start_device_authorization()

    assert authorization.interval == 5.0 and authorization.expires_in == 900.0


def test_an_unreadable_interval_is_a_bad_answer_rather_than_the_default():
    """An absent field takes the default; a present one GitHub cannot have meant is a bad answer."""
    flow, _ = _flow(ScriptedGitHub(device=dict(DEVICE_ANSWER) | {'interval': 'five'}))

    with pytest.raises(DeviceFlowError, match='unreadable interval'):
        flow.start_device_authorization()


def test_a_boolean_interval_is_a_bad_answer():
    """`bool` is an `int`, so the plain numeric test would take `True` as a one-second interval."""
    flow, _ = _flow(ScriptedGitHub(device=dict(DEVICE_ANSWER) | {'interval': True}))

    with pytest.raises(DeviceFlowError, match='unreadable interval'):
        flow.start_device_authorization()


@pytest.mark.parametrize('field', ['interval', 'expires_in'])
@pytest.mark.parametrize('value', [-1, 0, float('nan'), float('inf'), 1e100, 10**1000])
def test_a_timing_field_that_is_no_length_of_time_is_a_bad_answer(field: str, value: float):
    """A negative or NaN value raises inside `time.sleep`, zero polls in a tight loop, and a value past a day
    overflows `time.sleep` or `float` itself."""
    # `json` writes NaN and Infinity, which httpx's own encoder refuses, so the body is written here.
    body = json.dumps(dict(DEVICE_ANSWER) | {field: value})

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={'content-type': 'application/json'})

    http = httpx.Client(transport=httpx.MockTransport(answer))
    flow = GitHubDeviceFlow(CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(DeviceFlowError, match=f'{field}=.*no length of time'):
        flow.start_device_authorization()


@pytest.mark.parametrize('value', [5, 5.0, 0.5, 3600, 24 * 60 * 60])
def test_a_positive_interval_is_taken_as_github_sent_it(value: float):
    """The guard above refuses nothing GitHub can have meant, whole seconds or a fraction of one."""
    flow, _ = _flow(ScriptedGitHub(device=dict(DEVICE_ANSWER) | {'interval': value}))

    assert flow.start_device_authorization().interval == float(value)


def test_a_refused_device_code_request_names_what_github_answered():
    github = ScriptedGitHub(device={'error': 'incorrect_client_credentials'})
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='incorrect_client_credentials'):
        flow.start_device_authorization()


def test_a_device_code_answer_missing_a_field_is_unreadable():
    github = ScriptedGitHub(device={k: v for k, v in DEVICE_ANSWER.items() if k != 'user_code'})
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='user_code'):
        flow.start_device_authorization()


# --- step two: the poll -------------------------------------------------------


def test_an_authorized_device_code_yields_the_access_token():
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, sleep = _flow(github)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert sleep.seconds == [5.0]  # the wait GitHub asks for, then the poll that answers


def test_the_first_poll_waits_the_interval_github_asked_for():
    """RFC 8628 §3.3. A poll that starts at once is answered `slow_down`, which costs the code."""
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    flow.poll_for_token(_authorization(interval=7.0))

    assert github.events == ['wait 7.0', 'poll']


def test_the_token_request_names_the_device_grant_and_the_client():
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    flow.poll_for_token(_authorization())

    assert github.form(ACCESS_TOKEN_URL) == {
        'client_id': CLIENT_ID,
        'device_code': DEVICE_CODE,
        'grant_type': DEVICE_GRANT_TYPE,
    }


def test_a_pending_authorization_is_polled_until_the_user_finishes():
    pending = {'error': 'authorization_pending'}
    github = ScriptedGitHub(polls=[dict(pending), dict(pending), dict(GRANTED)])
    flow, sleep = _flow(github)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert sleep.seconds == [5.0, 5.0, 5.0]  # GitHub's own interval, held while it asks for no change


def test_slow_down_raises_the_interval_every_time_it_arrives():
    slow = {'error': 'slow_down', 'interval': 5}
    github = ScriptedGitHub(polls=[slow, slow, dict(GRANTED)])
    flow, sleep = _flow(github)

    flow.poll_for_token(_authorization())

    assert sleep.seconds == [5.0, 10.0, 15.0]  # the asked-for wait, then each raise it answers with


def test_an_expired_device_code_leaves_the_user_unauthorized():
    github = ScriptedGitHub(polls=[{'error': 'expired_token'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='expired'):
        flow.poll_for_token(_authorization())


def test_a_refused_authorization_leaves_the_user_unauthorized():
    github = ScriptedGitHub(polls=[{'error': 'authorization_pending'}, {'error': 'access_denied'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='refused the authorization'):
        flow.poll_for_token(_authorization())


def test_the_poll_stops_when_the_device_code_expires():
    """GitHub sets the lifetime, and a user who never authorizes runs it out."""
    github = ScriptedGitHub(polls=[{'error': 'authorization_pending'} for _ in range(100)])
    flow, sleep = _flow(github)

    with pytest.raises(DeviceFlowError, match='not authorized in time'):
        flow.poll_for_token(_authorization(expires_in=12.0))

    assert sleep.seconds == [5.0, 5.0]  # a third wait would pass 12 seconds, so it never starts


def test_an_unreadable_device_code_names_what_github_answered():
    """An error that is neither a refusal nor an expiry says GitHub could not read the request."""
    github = ScriptedGitHub(polls=[{'error': 'unsupported_grant_type'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='unsupported_grant_type'):
        flow.poll_for_token(_authorization())


def test_a_github_outage_is_reported_as_unavailable():
    github = ScriptedGitHub(polls=[dict(GRANTED)], status=503)
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='unavailable'):
        flow.poll_for_token(_authorization())


def test_a_poll_that_times_out_is_retried_at_a_slower_rate():
    """RFC 8628 §3.5: a timed-out poll slows down and tries again; the code may already be authorized."""
    answers = iter([None, dict(GRANTED)])
    slept: list[float] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = next(answers)
        if body is None:
            raise httpx.ReadTimeout('no answer in time', request=request)
        return httpx.Response(200, json=body)

    http = httpx.Client(transport=httpx.MockTransport(answer))
    flow = GitHubDeviceFlow(CLIENT_ID, http, monotonic=lambda: 0.0, sleep=slept.append)

    assert flow.poll_for_token(_authorization()) == GRANTED['access_token']
    assert slept == [_authorization().interval, _authorization().interval + 5.0]


def test_an_unreachable_github_is_reported_as_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('connection refused', request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    flow = GitHubDeviceFlow(CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(DeviceFlowError, match='unreachable'):
        flow.poll_for_token(_authorization())


def test_an_answer_that_is_not_a_json_object_is_unreadable():
    def answer_a_list(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(answer_a_list))
    flow = GitHubDeviceFlow(CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(DeviceFlowError, match='no JSON object'):
        flow.start_device_authorization()


# --- step three: the registration --------------------------------------------


@dataclass
class ScriptedGateway:
    """`users.register`, answering one scripted body and recording what reached it."""

    body: dict[str, object] = field(
        default_factory=lambda: {
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'api_key': MINTED_KEY,
            'key_status': 'created',
        }
    )
    requests: list[httpx.Request] = field(default_factory=list)

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url='https://gateway.example', transport=httpx.MockTransport(self.handle), timeout=REGISTER_TIMEOUT
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.body)


def test_the_registration_carries_the_access_token_as_the_credential():
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)
    gateway = ScriptedGateway()

    with PlatformClient(client=gateway.client()) as platform:
        response = register_with_github(platform, flow, alias='team-rocket')

    sent = gateway.requests[0]
    assert str(sent.url) == f'https://gateway.example{routes.USERS_REGISTER}'
    assert json.loads(sent.content) == {'credential': TOKEN, 'alias': 'team-rocket', 'rotate': False}
    assert response.api_key == MINTED_KEY


def test_the_registration_asks_the_gateway_to_rotate_when_told_to():
    """The account exists and this machine no longer holds its key, so a fresh one is minted."""
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)
    gateway = ScriptedGateway()

    with PlatformClient(client=gateway.client()) as platform:
        register_with_github(platform, flow, rotate=True)

    assert json.loads(gateway.requests[0].content)['rotate'] is True


@pytest.mark.parametrize(('flag', 'expected'), [(['--rotate'], True), ([], False)])
def test_the_command_asks_for_rotation_only_when_the_flag_is_given(flag: list[str], expected: bool):
    """Without the flag the command sends `rotate=False`, and the account keeps the key it holds."""
    asked: list[bool] = []

    def record(platform: object, flow: object, *, alias: str | None = None, rotate: bool = False) -> RegisterResponse:
        asked.append(rotate)
        return RegisterResponse.model_validate(ScriptedGateway().body)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', record)
        argv = ['platform-register', '--client-id=x', '--platform-url=https://gateway.example', *flag]
        patch.setattr(sys, 'argv', argv)

        github_device_flow.main()

    assert asked == [expected]


ENVIRONMENT_ID = 'Iv1.fromtheenvironment'
FLAG_ID = 'Iv1.fromtheflag'


@pytest.mark.parametrize(
    ('flag', 'environment', 'expected'),
    [
        ([], None, DEFAULT_GITHUB_CLIENT_ID),
        ([], ENVIRONMENT_ID, ENVIRONMENT_ID),
        ([f'--client-id={FLAG_ID}'], ENVIRONMENT_ID, FLAG_ID),
        ([f'--client-id={FLAG_ID}'], None, FLAG_ID),
    ],
    ids=['neither', 'the environment', 'both', 'the flag'],
)
def test_the_flag_beats_the_environment_and_both_beat_the_platform_app(
    flag: list[str], environment: str | None, expected: str
):
    """A user who sets nothing registers through the platform's own OAuth app."""
    asked: list[str] = []

    def register(platform: object, flow: object, *, alias: str | None = None, rotate: bool = False) -> RegisterResponse:
        return RegisterResponse.model_validate(ScriptedGateway().body)

    def capture(client_id: str, http: httpx.Client) -> GitHubDeviceFlow:
        asked.append(client_id)
        return GitHubDeviceFlow(client_id, http)

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(GITHUB_CLIENT_ID_ENV, raising=False)
        if environment is not None:
            patch.setenv(GITHUB_CLIENT_ID_ENV, environment)
        patch.setattr(github_device_flow, 'register_with_github', register)
        patch.setattr(github_device_flow, 'GitHubDeviceFlow', capture)
        patch.setattr(sys, 'argv', ['platform-register', '--platform-url=https://gateway.example', *flag])

        github_device_flow.main()

    assert asked == [expected]


@pytest.mark.parametrize('flag', [['--client-id='], []], ids=['the flag', 'the environment'])
def test_an_override_emptied_rather_than_unset_stops_before_github(flag: list[str]):
    """An empty value overrides the default with nothing, and GitHub reads an empty id as a bad request."""
    reached: list[str] = []

    def capture(client_id: str, http: httpx.Client) -> GitHubDeviceFlow:
        reached.append(client_id)
        return GitHubDeviceFlow(client_id, http)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(GITHUB_CLIENT_ID_ENV, '')
        patch.setattr(github_device_flow, 'GitHubDeviceFlow', capture)
        patch.setattr(sys, 'argv', ['platform-register', '--platform-url=https://gateway.example', *flag])

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    assert GITHUB_CLIENT_ID_ENV in str(raised.value) and reached == []


@pytest.mark.parametrize(
    ('base_url', 'allowed'),
    [
        ('https://gateway.example', True),
        ('http://127.0.0.1:8080', True),
        ('http://127.1.2.3:8080', True),
        ('http://[::1]:8080', True),
        ('http://localhost:8080', True),
        ('http://100.64.7.9:8080', False),
        ('http://203.0.113.5:8080', False),
        ('http://gateway.example', False),
        ('http://10.0.0.5:8080', False),
        ('ftp://gateway.example', False),
    ],
)
def test_only_https_and_a_loopback_address_pass_without_the_flag(base_url: str, allowed: bool):
    assert github_device_flow.platform_url_is_allowed(base_url, plaintext_http=False) is allowed


@pytest.mark.parametrize(
    ('base_url', 'allowed'),
    [
        ('http://100.64.7.9:8080', True),
        ('http://203.0.113.5:8080', True),
        ('http://gateway.example', True),
        ('https://gateway.example', True),
        ('ftp://gateway.example', False),
    ],
)
def test_the_flag_admits_any_http_and_no_other_scheme(base_url: str, allowed: bool):
    assert github_device_flow.platform_url_is_allowed(base_url, plaintext_http=True) is allowed


def test_a_malformed_platform_url_exits_with_one_line():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example:notaport']
        )

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    assert 'notaport' in str(raised.value)


@pytest.mark.parametrize(
    ('platform_url', 'trusts_env'), [('http://127.0.0.1:8080', False), ('https://gateway.example', True)]
)
def test_a_plain_http_gateway_client_ignores_an_environment_proxy(platform_url: str, trusts_env: bool):
    """`HTTP_PROXY` with no matching `NO_PROXY` would carry a plain-http token off the machine."""
    clients: list[dict[str, object]] = []
    real_client = httpx.Client

    def record(platform: object, flow: object, *, alias: str | None = None, rotate: bool = False) -> RegisterResponse:
        return RegisterResponse.model_validate(ScriptedGateway().body)

    def capture(**kwargs: Any) -> httpx.Client:
        clients.append(kwargs)
        return real_client(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', record)
        patch.setattr(httpx, 'Client', capture)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', f'--platform-url={platform_url}'])

        github_device_flow.main()

    gateway = next(c for c in clients if c.get('base_url') == platform_url)
    assert gateway['trust_env'] is trusts_env


def test_the_command_refuses_a_platform_that_would_show_the_token():
    """A shared-address http platform never sees the GitHub token: the command stops before the code."""
    reached: list[object] = []

    def record(platform: object, flow: object, *, alias: str | None = None, rotate: bool = False) -> RegisterResponse:
        reached.append(flow)
        return RegisterResponse.model_validate(ScriptedGateway().body)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', record)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=http://203.0.113.5:8080'])

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    assert '--plaintext-http' in str(raised.value) and reached == []


def test_the_flag_admits_a_platform_the_command_would_otherwise_refuse():
    """A staging user reaches an http platform off this machine, so the flow runs."""
    reached: list[object] = []

    def record(platform: object, flow: object, *, alias: str | None = None, rotate: bool = False) -> RegisterResponse:
        reached.append(flow)
        return RegisterResponse.model_validate(ScriptedGateway().body)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', record)
        argv = ['platform-register', '--client-id=x', '--platform-url=http://203.0.113.5:8080', '--plaintext-http']
        patch.setattr(sys, 'argv', argv)

        github_device_flow.main()

    assert len(reached) == 1


def test_a_success_that_is_no_registration_exits_with_one_line():
    """A 2xx from a proxy page or another gateway version ends the command without a traceback."""
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    gateway = ScriptedGateway(body={'unexpected': 'page'})
    flow, _ = _flow(github)  # built before the patch below, so GitHub keeps its own transport

    def flow_for(client_id: str, http: httpx.Client) -> GitHubDeviceFlow:
        return flow

    real_client = httpx.Client
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'GitHubDeviceFlow', flow_for)
        # Every client main opens answers as the gateway; the GitHub one is replaced above.
        patch.setattr(
            httpx,
            'Client',
            lambda **kwargs: real_client(**{**kwargs, 'transport': httpx.MockTransport(gateway.handle)}),
        )
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    assert 'no registration' in str(raised.value)


def test_the_user_is_shown_the_code_before_the_poll_starts(capsys):
    """Only the short code has to reach the user while the flow runs."""
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_github(platform, flow)

    shown = capsys.readouterr().out
    assert str(DEVICE_ANSWER['user_code']) in shown and str(DEVICE_ANSWER['verification_uri']) in shown


def test_the_registration_keeps_the_key_the_gateway_minted():
    """The flow leaves the client authenticated, so a caller registering in Python reads on."""
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_github(platform, flow)
        assert platform.api_key == MINTED_KEY


def test_the_registration_waits_out_the_gateways_own_github_budget():
    gateway = ScriptedGateway()
    github = ScriptedGitHub(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    with PlatformClient(client=gateway.client()) as platform:
        register_with_github(platform, flow)

    assert gateway.requests[0].extensions['timeout'] == {
        'connect': CONNECT_TIMEOUT_S,
        'read': GATEWAY_GITHUB_READS * MAX_REQUEST_STALL_S + MAX_REQUEST_STALL_S,
        'write': WRITE_TIMEOUT_S,
        'pool': POOL_TIMEOUT_S,
    }


def test_an_answer_lost_after_the_request_names_rotate_as_the_recovery():
    """The gateway may have minted the key before the connection dropped, so a plain retry finds no key."""

    def drop(*_args: object, **_kwargs: object) -> RegisterResponse:
        raise httpx.ReadError('connection reset')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', drop)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    assert isinstance(raised.value.code, str) and '--rotate' in raised.value.code


def test_a_refused_gateway_ends_the_command_rather_than_raising():
    """A dial the gateway refuses is a transport error, which no other layer here converts."""

    def refuse(*_args: object, **_kwargs: object) -> RegisterResponse:
        raise httpx.ConnectError('connection refused')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(github_device_flow, 'register_with_github', refuse)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            github_device_flow.main()

    # A string exit code prints and exits 1, which is how the argument faults report too.
    assert isinstance(raised.value.code, str) and 'gateway.example' in raised.value.code


# --- the timeout policy -------------------------------------------------------


def test_every_phase_of_a_github_request_carries_a_bound():
    assert max_stall_s(REQUEST_TIMEOUT) == MAX_REQUEST_STALL_S


def test_every_phase_of_a_registration_carries_a_bound():
    assert max_stall_s(REGISTER_TIMEOUT) > MAX_REQUEST_STALL_S


def test_a_timeout_that_leaves_a_phase_open_bounds_no_stall():
    open_pool = httpx.Timeout(connect=1.0, read=1.0, write=1.0, pool=None)

    with pytest.raises(ValueError, match='unbounded'):
        max_stall_s(open_pool)


def test_the_flow_sets_no_bare_number_as_a_timeout():
    """The mechanical half of the rule above: a bare number reads as a deadline and is not one."""
    source = Path(github_device_flow.__file__).read_text()
    # `timeout=` followed by a digit: the bare-number form, which httpx applies to every phase.
    offending = [line.strip() for line in source.splitlines() if re.search(r'timeout\s*=\s*[\d.]', line)]

    assert offending == []
