"""The user's half of the device flow, against GitHub and WorkOS, driven against a scripted transport.

No test reaches GitHub or WorkOS. An injected sleep records the intervals and advances a fake monotonic clock,
so the poll runs at full speed and the suite waits on nothing.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import device_flow, routes
from platform_client.client import CLIENT_VERSION, PLATFORM_VERSION_HEADER, PlatformClient
from platform_client.device_flow import (
    DEFAULT_GITHUB_CLIENT_ID,
    DEVICE_GRANT_TYPE,
    GITHUB_CLIENT_ID_ENV,
    WORKOS_CLIENT_ID_ENV,
    ClientIds,
    DeviceAuthorization,
    DeviceFlow,
    DeviceFlowError,
    NoSignIn,
    ProviderFailed,
    SignIn,
    register_with_device_flow,
    sign_in_flow,
)
from platform_client.errors import PlatformError
from platform_client.identity_providers import (
    CONNECT_TIMEOUT_S,
    GITHUB,
    GITHUB_REGISTER_TIMEOUT,
    IDENTITY_SCOPE,
    MAX_REQUEST_STALL_S,
    POOL_TIMEOUT_S,
    REQUEST_TIMEOUT,
    WORKOS,
    WRITE_TIMEOUT_S,
    IdentityProvider,
    max_stall_s,
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
COMPLETE_URI = 'https://a.authkit.app/device?user_code=WDJB-MJHT'


@dataclass
class ScriptedProvider:
    """One scripted answer per token poll, taken in order, plus the device-code endpoint.

    An answer that is an `httpx.Response` goes out as it is. A body with an `error` field goes out with
    the provider's error status: GitHub sends 200, WorkOS 400.
    """

    polls: list[dict[str, object] | httpx.Response] = field(default_factory=list)
    device: dict[str, object] | httpx.Response = field(default_factory=lambda: dict(DEVICE_ANSWER))
    status: int | None = None  # every endpoint's HTTP status when set
    provider: IdentityProvider = GITHUB
    error_status: int = 200
    requests: list[httpx.Request] = field(default_factory=list)
    # Every wait and every poll, in the order they happened: the order is what RFC 8628 §3.3 fixes.
    events: list[str] = field(default_factory=list)

    def _answer(self, body: dict[str, object] | httpx.Response) -> httpx.Response:
        if isinstance(body, httpx.Response):
            return body
        if self.status is not None:
            return httpx.Response(self.status, json=body)
        return httpx.Response(self.error_status if 'error' in body else 200, json=body)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.events.append('poll')
        url = str(request.url)
        if url == self.provider.device_authorization_url:
            return self._answer(self.device)
        if url == self.provider.token_url:
            return self._answer(self.polls.pop(0))
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


def _flow(scripted: ScriptedProvider) -> tuple[DeviceFlow, RecordedSleep]:
    sleep = RecordedSleep(scripted.events)
    http = httpx.Client(transport=httpx.MockTransport(scripted.handle))
    return DeviceFlow(scripted.provider, CLIENT_ID, http, monotonic=sleep.monotonic, sleep=sleep), sleep


def _authorization(*, interval: float = 5.0, expires_in: float = 900.0) -> DeviceAuthorization:
    return DeviceAuthorization(
        device_code=DEVICE_CODE,
        user_code=str(DEVICE_ANSWER['user_code']),
        verification_uri=str(DEVICE_ANSWER['verification_uri']),
        verification_uri_complete=None,
        interval=interval,
        expires_in=expires_in,
    )


# --- step one: the device code ------------------------------------------------


def test_the_device_code_request_asks_for_identity_scope_only():
    github = ScriptedProvider()
    flow, _ = _flow(github)

    authorization = flow.start_device_authorization()

    assert github.form(GITHUB.device_authorization_url) == {'client_id': CLIENT_ID, 'scope': IDENTITY_SCOPE}
    assert authorization.device_code == DEVICE_CODE
    assert authorization.user_code == DEVICE_ANSWER['user_code']
    assert authorization.verification_uri == DEVICE_ANSWER['verification_uri']
    assert authorization.verification_uri_complete is None
    assert authorization.interval == 5.0 and authorization.expires_in == 900.0


def test_a_device_code_answer_without_an_interval_polls_at_the_default_rate():
    github = ScriptedProvider(device={k: v for k, v in DEVICE_ANSWER.items() if k != 'interval'})
    flow, _ = _flow(github)

    authorization = flow.start_device_authorization()

    assert authorization.interval == 5.0 and authorization.expires_in == 900.0


def test_a_device_code_answer_without_expires_in_is_a_bad_answer():
    """RFC 8628 §3.2 requires the field, and the prompt tells the user the time that the provider gave."""
    flow, _ = _flow(ScriptedProvider(device={k: v for k, v in DEVICE_ANSWER.items() if k != 'expires_in'}))

    with pytest.raises(ProviderFailed, match='without expires_in'):
        flow.start_device_authorization()


def test_the_device_code_answer_keeps_the_link_that_carries_the_user_code():
    flow, _ = _flow(_workos(device=dict(DEVICE_ANSWER) | {'verification_uri_complete': COMPLETE_URI}))

    authorization = flow.start_device_authorization()

    assert authorization.verification_uri_complete == COMPLETE_URI


@pytest.mark.parametrize('value', ['', 7, None])
def test_an_unreadable_link_with_the_user_code_is_a_bad_answer(value: object):
    flow, _ = _flow(_workos(device=dict(DEVICE_ANSWER) | {'verification_uri_complete': value}))

    with pytest.raises(ProviderFailed, match='without verification_uri_complete'):
        flow.start_device_authorization()


def test_the_prompt_gives_the_link_with_the_user_code_and_its_lifetime():
    authorization = replace(_authorization(expires_in=300.0), verification_uri_complete=COMPLETE_URI)

    assert authorization.sign_in_prompt() == (
        f'open {COMPLETE_URI} and check that the page shows {DEVICE_ANSWER["user_code"]}; the link expires in 5 minutes'
    )


def test_the_prompt_without_a_complete_link_gives_the_link_the_code_and_its_lifetime():
    assert _authorization(expires_in=900.0).sign_in_prompt() == (
        f'open {DEVICE_ANSWER["verification_uri"]} and enter {DEVICE_ANSWER["user_code"]}; '
        'the code expires in 15 minutes'
    )


@pytest.mark.parametrize(
    ('expires_in', 'lifetime'),
    [(300.0, '5 minutes'), (299.0, '4 minutes'), (60.0, '1 minute'), (59.5, '59 seconds'), (1.0, '1 second')],
)
def test_the_prompt_rounds_the_lifetime_down(expires_in: float, lifetime: str):
    """A user told too much time comes back to a code that has expired."""
    assert _authorization(expires_in=expires_in).sign_in_prompt().endswith(f'expires in {lifetime}')


def test_an_unreadable_interval_is_a_bad_answer_rather_than_the_default():
    """An absent field takes the default; a present one GitHub cannot have meant is a bad answer."""
    flow, _ = _flow(ScriptedProvider(device=dict(DEVICE_ANSWER) | {'interval': 'five'}))

    with pytest.raises(DeviceFlowError, match='unreadable interval'):
        flow.start_device_authorization()


def test_a_boolean_interval_is_a_bad_answer():
    """`bool` is an `int`, so the plain numeric test would take `True` as a one-second interval."""
    flow, _ = _flow(ScriptedProvider(device=dict(DEVICE_ANSWER) | {'interval': True}))

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
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(DeviceFlowError, match=f'{field}=.*no length of time'):
        flow.start_device_authorization()


@pytest.mark.parametrize('value', [5, 5.0, 0.5, 3600, 24 * 60 * 60])
def test_a_positive_interval_is_taken_as_github_sent_it(value: float):
    """The guard above refuses nothing GitHub can have meant, whole seconds or a fraction of one."""
    flow, _ = _flow(ScriptedProvider(device=dict(DEVICE_ANSWER) | {'interval': value}))

    assert flow.start_device_authorization().interval == float(value)


def test_a_refused_device_code_request_names_what_github_answered():
    github = ScriptedProvider(device={'error': 'incorrect_client_credentials'})
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='incorrect_client_credentials'):
        flow.start_device_authorization()


def test_a_device_code_answer_missing_a_field_is_unreadable():
    github = ScriptedProvider(device={k: v for k, v in DEVICE_ANSWER.items() if k != 'user_code'})
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='user_code'):
        flow.start_device_authorization()


# --- step two: the poll -------------------------------------------------------


def test_an_authorized_device_code_yields_the_access_token():
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, sleep = _flow(github)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert sleep.seconds == [5.0]  # the wait GitHub asks for, then the poll that answers


def test_the_first_poll_waits_the_interval_github_asked_for():
    """RFC 8628 §3.3. A poll that starts at once is answered `slow_down`, which costs the code."""
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    flow.poll_for_token(_authorization(interval=7.0))

    assert github.events == ['wait 7.0', 'poll']


def test_the_token_request_names_the_device_grant_and_the_client():
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    flow.poll_for_token(_authorization())

    assert github.form(GITHUB.token_url) == {
        'client_id': CLIENT_ID,
        'device_code': DEVICE_CODE,
        'grant_type': DEVICE_GRANT_TYPE,
    }


def test_a_pending_authorization_is_polled_until_the_user_finishes():
    pending = {'error': 'authorization_pending'}
    github = ScriptedProvider(polls=[dict(pending), dict(pending), dict(GRANTED)])
    flow, sleep = _flow(github)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert sleep.seconds == [5.0, 5.0, 5.0]  # GitHub's own interval, held while it asks for no change


def test_slow_down_raises_the_interval_every_time_it_arrives():
    slow = {'error': 'slow_down', 'interval': 5}
    github = ScriptedProvider(polls=[slow, slow, dict(GRANTED)])
    flow, sleep = _flow(github)

    flow.poll_for_token(_authorization())

    assert sleep.seconds == [5.0, 10.0, 15.0]  # the asked-for wait, then each raise it answers with


def test_an_expired_device_code_leaves_the_user_unauthorized():
    github = ScriptedProvider(polls=[{'error': 'expired_token'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='expired'):
        flow.poll_for_token(_authorization())


def test_a_refused_authorization_leaves_the_user_unauthorized():
    github = ScriptedProvider(polls=[{'error': 'authorization_pending'}, {'error': 'access_denied'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='refused the authorization'):
        flow.poll_for_token(_authorization())


def test_the_poll_stops_when_the_device_code_expires():
    """GitHub sets the lifetime, and a user who never authorizes runs it out."""
    github = ScriptedProvider(polls=[{'error': 'authorization_pending'} for _ in range(100)])
    flow, sleep = _flow(github)

    with pytest.raises(DeviceFlowError, match='not authorized in time'):
        flow.poll_for_token(_authorization(expires_in=12.0))

    assert sleep.seconds == [5.0, 5.0]  # a third wait would pass 12 seconds, so it never starts


def test_an_unreadable_device_code_names_what_github_answered():
    """An error that is neither a refusal nor an expiry says GitHub could not read the request."""
    github = ScriptedProvider(polls=[{'error': 'unsupported_grant_type'}])
    flow, _ = _flow(github)

    with pytest.raises(DeviceFlowError, match='unsupported_grant_type'):
        flow.poll_for_token(_authorization())


def test_a_github_outage_is_reported_as_unavailable():
    github = ScriptedProvider(status=503)
    flow, _ = _flow(github)

    with pytest.raises(ProviderFailed, match='GitHub is unavailable: HTTP 503'):
        flow.start_device_authorization()


def test_a_poll_that_meets_a_server_error_is_retried_at_a_slower_rate():
    """The user may already have approved the code, so one failed poll does not cost a new sign-in."""
    github = ScriptedProvider(polls=[httpx.Response(502, text='bad gateway'), dict(GRANTED)])
    flow, sleep = _flow(github)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert sleep.seconds == [5.0, 10.0]


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
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=slept.append)

    assert flow.poll_for_token(_authorization()) == GRANTED['access_token']
    assert slept == [_authorization().interval, _authorization().interval + 5.0]


def test_an_unreachable_github_is_reported_as_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('connection refused', request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(ProviderFailed, match='GitHub is unreachable'):
        flow.start_device_authorization()


@pytest.mark.parametrize('error', [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError])
def test_a_poll_that_loses_its_connection_is_retried_at_a_slower_rate(error: type[httpx.TransportError]):
    """A short network drop costs the user no new sign-in: the code may already be approved."""
    answers = iter([None, dict(GRANTED)])
    slept: list[float] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = next(answers)
        if body is None:
            raise error('dropped', request=request)
        return httpx.Response(200, json=body)

    http = httpx.Client(transport=httpx.MockTransport(answer))
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=slept.append)

    assert flow.poll_for_token(_authorization()) == TOKEN
    assert slept == [5.0, 10.0]


def test_a_provider_that_stays_unreachable_ends_the_poll_when_the_code_expires():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('connection refused', request=request)

    sleep = RecordedSleep(events=[])
    http = httpx.Client(transport=httpx.MockTransport(refuse))
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=sleep.monotonic, sleep=sleep)

    with pytest.raises(ProviderFailed, match='GitHub is unreachable, and the device code expires first'):
        flow.poll_for_token(_authorization(expires_in=20.0))


def test_an_answer_that_is_not_a_json_object_is_unreadable():
    def answer_a_list(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(answer_a_list))
    flow = DeviceFlow(GITHUB, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=lambda _: None)

    with pytest.raises(DeviceFlowError, match='no JSON object'):
        flow.start_device_authorization()


# --- WorkOS: the same flow, its errors as HTTP 400 ----------------------------

WORKOS_TOKEN = 'eyJ.workos.access'
REFRESH_TOKEN = 'refresh-never-kept'
WORKOS_GRANTED: dict[str, object] = {
    'user': {'object': 'user', 'id': 'user_01', 'email': 'a@example.com', 'email_verified': True},
    'access_token': WORKOS_TOKEN,
    'refresh_token': REFRESH_TOKEN,
    'authentication_method': 'GoogleOAuth',
}


def _workos(**kwargs: Any) -> ScriptedProvider:
    return ScriptedProvider(provider=WORKOS, error_status=400, **kwargs)


def test_the_workos_device_code_request_names_the_client_and_no_scope():
    workos = _workos()
    flow, _ = _flow(workos)

    flow.start_device_authorization()

    assert workos.form(WORKOS.device_authorization_url) == {'client_id': CLIENT_ID}


def test_the_workos_token_request_names_the_device_grant_and_the_client():
    workos = _workos(polls=[dict(WORKOS_GRANTED)])
    flow, _ = _flow(workos)

    flow.poll_for_token(_authorization())

    assert workos.form(WORKOS.token_url) == {
        'client_id': CLIENT_ID,
        'device_code': DEVICE_CODE,
        'grant_type': DEVICE_GRANT_TYPE,
    }


def test_a_pending_workos_authorization_answered_with_400_is_polled_on():
    workos = _workos(polls=[{'error': 'authorization_pending'}, {'error': 'authorization_pending'}, WORKOS_GRANTED])
    flow, sleep = _flow(workos)

    assert flow.poll_for_token(_authorization()) == WORKOS_TOKEN
    assert sleep.seconds == [5.0, 5.0, 5.0]


def test_workos_slow_down_raises_the_interval():
    workos = _workos(polls=[{'error': 'slow_down'}, WORKOS_GRANTED])
    flow, sleep = _flow(workos)

    flow.poll_for_token(_authorization())

    assert sleep.seconds == [5.0, 10.0]


@pytest.mark.parametrize(
    ('error', 'message'),
    [
        ('access_denied', 'refused the authorization'),
        ('expired_token', 'expired'),
        ('invalid_grant', 'WorkOS refused the device code: invalid_grant'),
    ],
)
def test_a_terminal_workos_answer_ends_the_flow(error: str, message: str):
    workos = _workos(polls=[{'error': error, 'error_description': 'from WorkOS'}])
    flow, _ = _flow(workos)

    with pytest.raises(DeviceFlowError, match=message) as raised:
        flow.poll_for_token(_authorization())

    assert not isinstance(raised.value, ProviderFailed)


def test_a_refused_workos_device_request_is_the_callers_fault_not_an_outage():
    """WorkOS answers a client id it does not know with 400 and an `error`, which names the cause."""
    workos = _workos(device={'error': 'invalid_client', 'error_description': 'Invalid client ID.'})
    flow, _ = _flow(workos)

    with pytest.raises(DeviceFlowError, match='WorkOS refused the device-code request: invalid_client') as raised:
        flow.start_device_authorization()

    assert not isinstance(raised.value, ProviderFailed)


@pytest.mark.parametrize(
    'answer',
    [httpx.Response(400, text='<html>bad request</html>'), httpx.Response(400, json={'message': 'no error field'})],
    ids=['no JSON', 'no error field'],
)
def test_a_400_that_is_no_oauth_error_is_an_outage(answer: httpx.Response):
    workos = _workos(device=answer)
    flow, _ = _flow(workos)

    with pytest.raises(ProviderFailed, match='WorkOS is unavailable: HTTP 400'):
        flow.start_device_authorization()


def test_a_rate_limited_poll_waits_at_least_what_workos_asks():
    limited = httpx.Response(429, headers={'retry-after': '30'}, text='slow down')
    workos = _workos(polls=[limited, WORKOS_GRANTED])
    flow, sleep = _flow(workos)

    assert flow.poll_for_token(_authorization()) == WORKOS_TOKEN
    assert sleep.seconds == [5.0, 30.0]


def test_a_rate_limit_that_outlives_the_code_is_the_providers_failure():
    """The user may have approved the code; what ended the flow is the wait WorkOS asked for."""
    limited = httpx.Response(429, headers={'retry-after': '3600'})
    workos = _workos(polls=[limited])
    flow, sleep = _flow(workos)

    with pytest.raises(ProviderFailed, match='asks for 3600 s, and the device code expires first'):
        flow.poll_for_token(_authorization(expires_in=300.0))

    assert sleep.seconds == [5.0]


def test_a_refusal_after_a_failed_poll_is_the_providers_failure():
    """The failed poll may have spent the code, so the refusal that follows is no proof the user refused."""
    workos = _workos(polls=[httpx.Response(502), {'error': 'invalid_grant'}])
    flow, _ = _flow(workos)

    with pytest.raises(ProviderFailed, match='refused the device code: invalid_grant, after a poll that failed'):
        flow.poll_for_token(_authorization())


@pytest.mark.parametrize(
    ('failed', 'answer'),
    [(httpx.Response(429), {'error': 'invalid_grant'}), (httpx.Response(502), {'error': 'access_denied'})],
    ids=['a rate limit spends no code', 'a refusal is the user'],
)
def test_a_refusal_after_a_failure_that_spent_no_code_is_still_a_refusal(failed: httpx.Response, answer: dict):
    workos = _workos(polls=[failed, answer])
    flow, _ = _flow(workos)

    with pytest.raises(DeviceFlowError) as raised:
        flow.poll_for_token(_authorization())

    assert not isinstance(raised.value, ProviderFailed)


def test_an_expiry_after_a_poll_that_reached_the_provider_is_the_providers_failure():
    workos = _workos(polls=[httpx.Response(502), {'error': 'expired_token'}])
    flow, _ = _flow(workos)

    with pytest.raises(ProviderFailed, match='expired before it was authorized, after a poll that failed'):
        flow.poll_for_token(_authorization())


def test_a_rate_limit_after_a_lost_poll_keeps_the_lost_poll_in_mind():
    """The timed-out poll may have spent the code; the 429 after it spent nothing and does not erase that."""
    answers = iter(['timeout', httpx.Response(429), {'error': 'invalid_grant'}])
    slept: list[float] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = next(answers)
        if body == 'timeout':
            raise httpx.ReadTimeout('no answer in time', request=request)
        return body if isinstance(body, httpx.Response) else httpx.Response(400, json=body)

    http = httpx.Client(transport=httpx.MockTransport(answer))
    flow = DeviceFlow(WORKOS, CLIENT_ID, http, monotonic=lambda: 0.0, sleep=slept.append)

    with pytest.raises(ProviderFailed, match='invalid_grant, after a poll that failed: WorkOS did not answer in time'):
        flow.poll_for_token(_authorization())


def test_a_rate_limit_followed_by_a_pending_answer_expires_as_unauthorized():
    workos = _workos(polls=[httpx.Response(429), *[{'error': 'authorization_pending'}] * 10])
    flow, _ = _flow(workos)

    with pytest.raises(DeviceFlowError, match='not authorized in time') as raised:
        flow.poll_for_token(_authorization(expires_in=30.0))

    assert not isinstance(raised.value, ProviderFailed)


def test_a_rate_limit_slows_the_next_poll_only_as_long_as_workos_asked():
    """`Retry-After` sets one wait; the polls after it run at the stepped interval, not at the long wait."""
    limited = httpx.Response(429, headers={'retry-after': '30'})
    workos = _workos(polls=[limited, {'error': 'authorization_pending'}, WORKOS_GRANTED])
    flow, sleep = _flow(workos)

    assert flow.poll_for_token(_authorization()) == WORKOS_TOKEN
    assert sleep.seconds == [5.0, 30.0, 10.0]


@pytest.mark.parametrize('retry_after', [None, 'Wed, 21 Oct 2026 07:28:00 GMT', '-1', 'nan', '\u00b2', '\u0663'])
def test_a_rate_limited_poll_without_a_whole_number_of_seconds_takes_the_step(retry_after: str | None):
    # A digit outside ASCII passes `str.isdigit`, and `float` refuses it.
    headers: dict[bytes, bytes] = {} if retry_after is None else {b'retry-after': retry_after.encode()}
    workos = _workos(polls=[httpx.Response(429, headers=headers), WORKOS_GRANTED])
    flow, sleep = _flow(workos)

    flow.poll_for_token(_authorization())

    assert sleep.seconds == [5.0, 10.0]


def test_a_rate_limited_device_request_names_the_wait():
    workos = _workos(device=httpx.Response(429, headers={'retry-after': '42'}))
    flow, _ = _flow(workos)

    with pytest.raises(ProviderFailed, match='WorkOS refused the request rate; it asks for 42 s'):
        flow.start_device_authorization()


def test_the_refresh_token_leaves_the_flow_nowhere(capsys):
    """The access token reaches `users.register`; the refresh token reaches no return value and no output."""
    workos = _workos(polls=[dict(WORKOS_GRANTED)])
    flow, _ = _flow(workos)
    gateway = ScriptedGateway()

    with PlatformClient(client=gateway.client()) as platform:
        response = register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))

    assert json.loads(gateway.requests[-1].content)['credential'] == WORKOS_TOKEN
    captured = capsys.readouterr()
    assert REFRESH_TOKEN not in captured.out + captured.err + response.model_dump_json()
    assert all(REFRESH_TOKEN.encode() not in request.content for request in gateway.requests)


# --- step three: the registration --------------------------------------------


# What a platform built against this client names in every answer.
SERVES_THIS_CLIENT = {PLATFORM_VERSION_HEADER: str(CLIENT_VERSION)}


@dataclass
class ScriptedGateway:
    """`auth.config` and `users.register`, answering scripted bodies and recording what reached them."""

    body: dict[str, object] = field(
        default_factory=lambda: {
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'api_key': MINTED_KEY,
            'key_status': 'created',
        }
    )
    config: dict[str, object] | httpx.Response = field(
        default_factory=lambda: {'github_registration': False, 'workos': WORKOS_SETTINGS}
    )
    requests: list[httpx.Request] = field(default_factory=list)

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url='https://gateway.example', transport=httpx.MockTransport(self.handle), timeout=REQUEST_TIMEOUT
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == routes.AUTH_CONFIG:
            if isinstance(self.config, httpx.Response):
                return self.config
            return httpx.Response(200, json=self.config, headers=SERVES_THIS_CLIENT)
        return httpx.Response(200, json=self.body, headers=SERVES_THIS_CLIENT)


WORKOS_SETTINGS = {
    'client_id': 'client_from_the_gateway',
    'authkit_domain': 'auth.example.authkit.app',
    'register_deadline_s': 75.0,
}


def test_the_registration_carries_the_access_token_as_the_credential():
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)
    gateway = ScriptedGateway()

    with PlatformClient(client=gateway.client()) as platform:
        response = register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT), alias='team-rocket')

    sent = gateway.requests[-1]
    assert str(sent.url) == f'https://gateway.example{routes.USERS_REGISTER}'
    assert json.loads(sent.content) == {'credential': TOKEN, 'alias': 'team-rocket', 'rotate': False}
    assert response.api_key == MINTED_KEY


def test_the_registration_asks_the_gateway_to_rotate_when_told_to():
    """The account exists and this machine no longer holds its key, so a fresh one is minted."""
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)
    gateway = ScriptedGateway()

    with PlatformClient(client=gateway.client()) as platform:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT), rotate=True)

    assert json.loads(gateway.requests[-1].content)['rotate'] is True


# --- the pick: which provider `auth.config` names ------------------------------


def _picked(config: dict[str, object] | httpx.Response, *, workos_client_id: str | None = None) -> SignIn:
    gateway = ScriptedGateway(config=config)
    with PlatformClient(client=gateway.client()) as platform, httpx.Client() as http:
        sign_in = sign_in_flow(platform, http, ClientIds(github=CLIENT_ID, workos=workos_client_id))
    assert gateway.requests[0].url.path == routes.AUTH_CONFIG
    assert 'authorization' not in gateway.requests[0].headers
    return sign_in


def test_a_platform_that_names_a_workos_client_signs_in_through_workos_with_that_client():
    flow = _picked({'github_registration': False, 'workos': WORKOS_SETTINGS}).flow

    assert flow.provider is WORKOS
    assert flow.client_id == WORKOS_SETTINGS['client_id']


def test_a_platform_that_takes_both_signs_in_through_workos():
    assert _picked({'github_registration': True, 'workos': WORKOS_SETTINGS}).flow.provider is WORKOS


def test_the_workos_override_replaces_the_client_the_platform_names():
    flow = _picked({'github_registration': False, 'workos': WORKOS_SETTINGS}, workos_client_id='client_override').flow

    assert flow.client_id == 'client_override'


def test_a_platform_that_takes_github_only_signs_in_through_github():
    flow = _picked({'github_registration': True, 'workos': None}).flow

    assert flow.provider is GITHUB
    assert flow.client_id == CLIENT_ID


def test_the_workos_override_does_not_turn_a_github_platform_into_a_workos_one():
    flow = _picked({'github_registration': True, 'workos': None}, workos_client_id='client_override').flow

    assert flow.provider is GITHUB


def test_a_platform_that_takes_no_sign_in_this_client_runs_is_refused():
    with pytest.raises(NoSignIn, match='gateway.example'):
        _picked({'github_registration': False, 'workos': None})


@pytest.mark.parametrize('status', [404, 500])
def test_a_failing_auth_config_is_no_reason_to_pick_github(status: int):
    with pytest.raises(PlatformError):
        _picked(httpx.Response(status, json={'detail': 'boom'}))


# --- the command --------------------------------------------------------------


@dataclass
class Picked:
    """`sign_in_flow` and `register_with_device_flow` replaced: what the command asked of them."""

    flow: DeviceFlow | None = None
    github_client_ids: list[str] = field(default_factory=list)
    workos_client_ids: list[str | None] = field(default_factory=list)
    rotations: list[bool] = field(default_factory=list)

    def sign_in_flow(self, platform: PlatformClient, http: httpx.Client, client_ids: ClientIds) -> SignIn:
        self.github_client_ids.append(client_ids.github)
        self.workos_client_ids.append(client_ids.workos)
        return SignIn(self.flow or DeviceFlow(GITHUB, client_ids.github, http), GITHUB_REGISTER_TIMEOUT)

    def register(
        self, platform: PlatformClient, sign_in: SignIn, *, alias: str | None = None, rotate: bool = False
    ) -> RegisterResponse:
        self.rotations.append(rotate)
        return RegisterResponse.model_validate(ScriptedGateway().body)


def _run(argv: list[str], picked: Picked, patch: pytest.MonkeyPatch) -> None:
    patch.setattr(device_flow, 'sign_in_flow', picked.sign_in_flow)
    patch.setattr(device_flow, 'register_with_device_flow', picked.register)
    patch.setattr(sys, 'argv', ['platform-register', *argv])
    device_flow.main()


@pytest.mark.parametrize(('flag', 'expected'), [(['--rotate'], True), ([], False)])
def test_the_command_asks_for_rotation_only_when_the_flag_is_given(flag: list[str], expected: bool):
    """Without the flag the command sends `rotate=False`, and the account keeps the key it holds."""
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        _run(['--client-id=x', '--platform-url=https://gateway.example', *flag], picked, patch)

    assert picked.rotations == [expected]


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
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(GITHUB_CLIENT_ID_ENV, raising=False)
        if environment is not None:
            patch.setenv(GITHUB_CLIENT_ID_ENV, environment)
        _run(['--platform-url=https://gateway.example', *flag], picked, patch)

    assert picked.github_client_ids == [expected]


@pytest.mark.parametrize(
    ('flag', 'environment', 'expected'),
    [
        ([], None, None),
        ([], ENVIRONMENT_ID, ENVIRONMENT_ID),
        ([f'--workos-client-id={FLAG_ID}'], ENVIRONMENT_ID, FLAG_ID),
        ([f'--workos-client-id={FLAG_ID}'], None, FLAG_ID),
    ],
    ids=['neither', 'the environment', 'both', 'the flag'],
)
def test_the_workos_flag_beats_the_environment_and_both_beat_the_platform_client(
    flag: list[str], environment: str | None, expected: str | None
):
    """A user who sets nothing signs in through the WorkOS client the platform names."""
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(WORKOS_CLIENT_ID_ENV, raising=False)
        if environment is not None:
            patch.setenv(WORKOS_CLIENT_ID_ENV, environment)
        _run(['--platform-url=https://gateway.example', *flag], picked, patch)

    assert picked.workos_client_ids == [expected]


@pytest.mark.parametrize(
    ('flag', 'variable'),
    [
        (['--client-id='], GITHUB_CLIENT_ID_ENV),
        ([], GITHUB_CLIENT_ID_ENV),
        (['--workos-client-id='], WORKOS_CLIENT_ID_ENV),
        ([], WORKOS_CLIENT_ID_ENV),
    ],
    ids=['the GitHub flag', 'the GitHub environment', 'the WorkOS flag', 'the WorkOS environment'],
)
def test_an_override_emptied_rather_than_unset_stops_before_any_request(flag: list[str], variable: str):
    """An empty value overrides the default with nothing, and a provider reads an empty id as a bad request."""
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(variable, '')
        with pytest.raises(SystemExit) as raised:
            _run(['--platform-url=https://gateway.example', *flag], picked, patch)

    assert variable in str(raised.value) and picked.github_client_ids == []


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
    assert device_flow.platform_url_is_allowed(base_url, plaintext_http=False) is allowed


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
    assert device_flow.platform_url_is_allowed(base_url, plaintext_http=True) is allowed


def test_a_malformed_platform_url_exits_with_one_line():
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(SystemExit) as raised:
            _run(['--client-id=x', '--platform-url=https://gateway.example:notaport'], Picked(), patch)

    assert 'notaport' in str(raised.value)


def test_run_registration_gates_the_url_itself():
    """A direct caller reaches no transport with a URL the gate refuses: the check is inside."""
    reached: list[object] = []

    def record(**kwargs: Any) -> httpx.Client:
        reached.append(kwargs)
        raise AssertionError('the token must not reach a client on a refused URL')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, 'Client', record)

        with pytest.raises(SystemExit) as raised:
            device_flow.run_registration('http://gateway.example', client_ids=ClientIds(), alias=None, rotate=False)

    assert 'in the clear' in str(raised.value)
    assert reached == []


def test_the_command_refuses_a_platform_that_would_show_the_token_before_reading_auth_config():
    """A shared-address http platform never sees the sign-in token: the command stops before `auth.config`."""
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(SystemExit) as raised:
            _run(['--client-id=x', '--platform-url=http://203.0.113.5:8080'], picked, patch)

    assert '--plaintext-http' in str(raised.value) and picked.github_client_ids == []


def test_the_flag_admits_a_platform_the_command_would_otherwise_refuse():
    """A staging user reaches an http platform off this machine, so the flow runs."""
    picked = Picked()

    with pytest.MonkeyPatch.context() as patch:
        _run(['--client-id=x', '--platform-url=http://203.0.113.5:8080', '--plaintext-http'], picked, patch)

    assert len(picked.rotations) == 1


def _main_against(gateway: ScriptedGateway, flow: DeviceFlow | None = None) -> SystemExit:
    """Run the command with every client it opens answering as `gateway`, and return how it exited."""
    real_client = httpx.Client
    with pytest.MonkeyPatch.context() as patch:
        if flow is not None:
            patch.setattr(device_flow, 'sign_in_flow', lambda *_args, **_kwargs: SignIn(flow, GITHUB_REGISTER_TIMEOUT))
        patch.setattr(
            httpx,
            'Client',
            lambda **kwargs: real_client(**{**kwargs, 'transport': httpx.MockTransport(gateway.handle)}),
        )
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            device_flow.main()
    return raised.value


def test_a_success_that_is_no_registration_exits_with_one_line():
    """A 2xx from a proxy page or another gateway version ends the command without a traceback."""
    flow, _ = _flow(ScriptedProvider(polls=[dict(GRANTED)]))  # built before the patch, so GitHub keeps its transport

    exited = _main_against(ScriptedGateway(body={'unexpected': 'page'}), flow)

    assert 'cannot read' in str(exited.code)


def test_an_unreadable_auth_config_exits_with_one_line():
    exited = _main_against(ScriptedGateway(config={'unexpected': 'page'}))

    assert 'cannot read' in str(exited.code) and '--rotate' not in str(exited.code)


def test_a_platform_that_takes_no_sign_in_this_client_runs_exits_with_one_line():
    exited = _main_against(ScriptedGateway(config={'github_registration': False, 'workos': None}))

    assert 'takes no sign-in' in str(exited.code)


def test_an_auth_config_lost_in_transit_does_not_name_rotate():
    """No credential left the machine yet, so no key can have been minted."""

    def drop(*_args: object, **_kwargs: object) -> DeviceFlow:
        raise httpx.ReadTimeout(
            'no answer in time', request=httpx.Request('GET', f'https://gateway.example{routes.AUTH_CONFIG}')
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(device_flow, 'sign_in_flow', drop)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            device_flow.main()

    assert isinstance(raised.value.code, str) and 'did not answer' in raised.value.code
    assert '--rotate' not in raised.value.code


def test_the_user_is_shown_the_code_on_stderr_before_the_poll_starts(capsys):
    """Only the short code has to reach the user while the flow runs, and it leaves stdout to the answer."""
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))

    captured = capsys.readouterr()
    assert str(DEVICE_ANSWER['user_code']) in captured.err and str(DEVICE_ANSWER['verification_uri']) in captured.err
    assert 'spam folder' not in captured.err
    assert captured.out == ''


def test_the_user_is_shown_the_link_that_carries_the_code_when_the_provider_sends_one(capsys):
    """The user opens one link and types nothing, and the line says how long the link stays valid."""
    device = dict(DEVICE_ANSWER) | {'verification_uri_complete': COMPLETE_URI, 'expires_in': 300}
    flow, _ = _flow(_workos(device=device, polls=[dict(WORKOS_GRANTED)]))

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))

    err = capsys.readouterr().err
    assert COMPLETE_URI in err and 'expires in 5 minutes' in err


def test_a_workos_sign_in_line_sends_the_user_to_the_spam_folder_for_the_code_mail(capsys):
    """A mail provider can file the WorkOS code mail as spam, so the line says where to look."""
    device = dict(DEVICE_ANSWER) | {'verification_uri_complete': COMPLETE_URI, 'expires_in': 300}
    flow, _ = _flow(_workos(device=device, polls=[dict(WORKOS_GRANTED)]))

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(
        'expires in 5 minutes. If the sign-in page sends you a code by email and it is not in your inbox, '
        'look in your spam folder.'
    )


def test_the_registration_keeps_the_key_the_gateway_minted():
    """The flow leaves the client authenticated, so a caller registering in Python reads on."""
    github = ScriptedProvider(polls=[dict(GRANTED)])
    flow, _ = _flow(github)

    with PlatformClient(client=ScriptedGateway().client()) as platform:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))
        assert platform.api_key == MINTED_KEY


def test_a_github_registration_waits_out_the_gateways_two_github_reads():
    """The gateway reads the account and its addresses inside `users.register`, then does its own work."""
    gateway = ScriptedGateway()
    flow, _ = _flow(ScriptedProvider(polls=[dict(GRANTED)]))
    sign_in = _picked({'github_registration': True, 'workos': None})

    with PlatformClient(client=gateway.client()) as platform:
        register_with_device_flow(platform, SignIn(flow, sign_in.register_timeout))

    assert gateway.requests[-1].extensions['timeout'] == {
        'connect': CONNECT_TIMEOUT_S,
        'read': 3 * MAX_REQUEST_STALL_S,
        'write': WRITE_TIMEOUT_S,
        'pool': POOL_TIMEOUT_S,
    }


def test_a_workos_registration_waits_out_the_deadline_the_gateway_names():
    """`auth.config` states the gateway's deadline for its WorkOS checks; the client adds one request."""
    gateway = ScriptedGateway()
    flow, _ = _flow(_workos(polls=[dict(WORKOS_GRANTED)]))
    settings = WORKOS_SETTINGS | {'register_deadline_s': 120.0}
    sign_in = _picked({'github_registration': False, 'workos': settings})

    with PlatformClient(client=gateway.client()) as platform:
        register_with_device_flow(platform, SignIn(flow, sign_in.register_timeout))

    assert gateway.requests[-1].extensions['timeout'] == {
        'connect': CONNECT_TIMEOUT_S,
        'read': 120.0 + MAX_REQUEST_STALL_S,
        'write': WRITE_TIMEOUT_S,
        'pool': POOL_TIMEOUT_S,
    }


def test_an_answer_lost_after_the_request_names_rotate_as_the_recovery():
    """The gateway may have minted the key before the connection dropped, so a plain retry finds no key."""

    def drop(*_args: object, **_kwargs: object) -> RegisterResponse:
        raise httpx.ReadError(
            'connection reset', request=httpx.Request('POST', f'https://gateway.example{routes.USERS_REGISTER}')
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(device_flow, 'sign_in_flow', Picked().sign_in_flow)
        patch.setattr(device_flow, 'register_with_device_flow', drop)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            device_flow.main()

    assert isinstance(raised.value.code, str) and '--rotate' in raised.value.code


def test_an_answer_lost_from_a_platform_under_a_path_prefix_names_rotate_as_the_recovery():
    def drop(*_args: object, **_kwargs: object) -> RegisterResponse:
        lost = httpx.Request('POST', f'https://gateway.example/prefix{routes.USERS_REGISTER}')
        raise httpx.ReadError('connection reset', request=lost)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(device_flow, 'sign_in_flow', Picked().sign_in_flow)
        patch.setattr(device_flow, 'register_with_device_flow', drop)
        argv = ['platform-register', '--client-id=x', '--platform-url=https://gateway.example/prefix']
        patch.setattr(sys, 'argv', argv)

        with pytest.raises(SystemExit) as raised:
            device_flow.main()

    assert isinstance(raised.value.code, str) and '--rotate' in raised.value.code


@pytest.mark.parametrize(
    ('error', 'path', 'minted'),
    [
        (httpx.ReadTimeout, routes.USERS_REGISTER, True),
        (httpx.DecodingError, routes.USERS_REGISTER, True),
        (httpx.TooManyRedirects, f'/prefix{routes.USERS_REGISTER}', True),
        (httpx.ConnectError, routes.USERS_REGISTER, False),
        (httpx.ConnectTimeout, routes.USERS_REGISTER, False),
        (httpx.ReadTimeout, routes.AUTH_CONFIG, False),
    ],
)
def test_only_a_registration_that_left_the_machine_may_have_minted_a_key(
    error: type[httpx.RequestError], path: str, minted: bool
):
    exc = error('lost', request=httpx.Request('POST', f'https://gateway.example{path}'))

    assert device_flow.may_have_minted_a_key(exc) is minted


@pytest.mark.parametrize(
    'message', ['the access token was already used: sign in again', 'the WorkOS email address is not verified']
)
def test_a_registration_the_gateway_refuses_shows_its_message(message: str):
    refused = httpx.Response(401, json={'error': {'code': 'unauthorized', 'message': message, 'details': {}}})
    flow, _ = _flow(_workos(polls=[dict(WORKOS_GRANTED)]))
    http = httpx.Client(base_url='https://gateway.example', transport=httpx.MockTransport(lambda _: refused))

    with PlatformClient(client=http) as platform, pytest.raises(PlatformError) as raised:
        register_with_device_flow(platform, SignIn(flow, GITHUB_REGISTER_TIMEOUT))

    assert message in str(raised.value)


def test_a_refused_gateway_ends_the_command_rather_than_raising():
    """A dial the gateway refuses is a transport error, which no other layer here converts."""

    def refuse(*_args: object, **_kwargs: object) -> DeviceFlow:
        raise httpx.ConnectError('connection refused')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(device_flow, 'sign_in_flow', refuse)
        patch.setattr(sys, 'argv', ['platform-register', '--client-id=x', '--platform-url=https://gateway.example'])

        with pytest.raises(SystemExit) as raised:
            device_flow.main()

    # A string exit code prints and exits 1, which is how the argument faults report too.
    assert isinstance(raised.value.code, str) and 'gateway.example' in raised.value.code


# --- the timeout policy -------------------------------------------------------


def test_every_phase_of_a_provider_request_carries_a_bound():
    assert max_stall_s(REQUEST_TIMEOUT) == MAX_REQUEST_STALL_S


def test_every_phase_of_a_registration_carries_a_bound():
    assert max_stall_s(GITHUB_REGISTER_TIMEOUT) > MAX_REQUEST_STALL_S


def test_a_timeout_that_leaves_a_phase_open_bounds_no_stall():
    open_pool = httpx.Timeout(connect=1.0, read=1.0, write=1.0, pool=None)

    with pytest.raises(ValueError, match='unbounded'):
        max_stall_s(open_pool)


@pytest.mark.parametrize('module', ['device_flow', 'identity_providers'])
def test_the_flow_sets_no_bare_number_as_a_timeout(module: str):
    """The mechanical half of the rule above: a bare number reads as a deadline and is not one."""
    source = (Path(device_flow.__file__).parent / f'{module}.py').read_text()
    # `timeout=` followed by a digit: the bare-number form, which httpx applies to every phase.
    offending = [line.strip() for line in source.splitlines() if re.search(r'timeout\s*=\s*[\d.]', line)]

    assert offending == []
