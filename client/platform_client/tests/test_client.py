"""PlatformClient: one method per endpoint, over a stub transport."""

from __future__ import annotations

import json

import httpx
import pytest
from platform_client import routes
from platform_client.boards import BoardRef
from platform_client.client import (
    API_KEY_ENV,
    API_URL_ENV,
    DEFAULT_PLATFORM_URL,
    PlatformClient,
    resolve_api_key,
    resolve_base_url,
)
from platform_client.enums import (
    BoardVisibility,
    ErrorCode,
    KeyStatus,
    OnExhausted,
    QuotaSubject,
    ReasonCode,
    SubmissionStatus,
)
from platform_client.errors import EVALS_DETAIL, REASON_CODE_DETAIL, PlatformError
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, SubmissionId
from platform_client.policy_images import PolicyImage
from platform_client.requests import CancelRequest, RegisterRequest, SubmissionCreateRequest
from platform_client.responses import (
    QUOTA_SUBMISSIONS_DAY,
    BoardListResponse,
    CancelResponse,
    MeResponse,
    PendingSubmissionView,
    RankingsResponse,
    RegisterResponse,
    SubmissionCreateResponse,
    SubmissionListResponse,
)
from pydantic import ValidationError

BASE = 'http://gateway.test'
KEY = ApiKey('pk_live_secret')
AT = '2026-03-04T05:06:07Z'


class Gateway:
    """Records the request it was handed and answers a canned payload."""

    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self.payload = payload
        self.seen: httpx.Request | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen = request
        return httpx.Response(self.status, json=self.payload)

    def request(self) -> httpx.Request:
        assert self.seen is not None, 'no request reached the gateway'
        return self.seen

    def body(self) -> dict:
        return json.loads(self.request().content)


def make_client(gateway: Gateway, *, api_key: ApiKey | None = KEY) -> PlatformClient:
    transport = httpx.MockTransport(gateway)
    return PlatformClient(client=httpx.Client(base_url=BASE, transport=transport), api_key=api_key)


def test_register_posts_the_body_unauthenticated_and_parses_the_response():
    gateway = Gateway(200, {'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})
    client = make_client(gateway, api_key=None)

    response = client.register(RegisterRequest(credential='token', alias='demo'))

    assert isinstance(response, RegisterResponse)
    assert response.user_id == 0xA0
    assert response.key_status is KeyStatus.existing
    assert response.api_key is None
    assert gateway.request().url.path == routes.USERS_REGISTER
    assert gateway.request().method == 'POST'
    assert 'authorization' not in gateway.request().headers
    assert gateway.body() == {'credential': 'token', 'alias': 'demo', 'rotate': False}


def test_register_sends_no_key_even_when_the_client_holds_one():
    gateway = Gateway(200, {'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})
    make_client(gateway).register(RegisterRequest(credential='token'))
    assert 'authorization' not in gateway.request().headers


def test_register_carries_back_a_minted_key():
    gateway = Gateway(
        200,
        {'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'api_key': 'pk_live_new', 'key_status': 'created'},
    )
    response = make_client(gateway, api_key=None).register(RegisterRequest(credential='token'))
    assert response.api_key == 'pk_live_new'
    assert response.key_status is KeyStatus.created


def test_register_keeps_the_key_it_is_given_so_the_next_call_is_authenticated():
    gateway = Gateway(
        200,
        {'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'api_key': 'pk_live_new', 'key_status': 'created'},
    )
    client = make_client(gateway, api_key=None)

    client.register(RegisterRequest(credential='token'))

    assert client.api_key == 'pk_live_new'


def test_a_registration_that_carries_no_key_leaves_the_one_already_held():
    # A repeat registration answers `existing` with no key: the caller's working key is not a thing
    # to clear, and clearing it would break the very next authenticated call.
    gateway = Gateway(200, {'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})
    client = make_client(gateway)

    client.register(RegisterRequest(credential='token'))

    assert client.api_key == KEY


def test_me_sends_the_bearer_token_and_parses_every_limit():
    gateway = Gateway(
        200,
        {
            'user_id': 'a0',
            'alias': 'demo',
            'tenant': 'nebius-2026',
            'plan': 'nebius_competition_2026',
            'quota': [
                {
                    'key': QUOTA_SUBMISSIONS_DAY,
                    'meter': 'submissions',
                    'unit': 'submission',
                    'scale': 1,
                    'window': 'day',
                    'subject': 'user',
                    'scope': [],
                    'limit': 2,
                    'used': 1,
                    'resets_at': AT,
                    'on_exhausted': 'block',
                }
            ],
        },
    )
    response = make_client(gateway).me()

    assert isinstance(response, MeResponse)
    assert response.tenant == 'nebius-2026'
    limit = response.quota_for(QUOTA_SUBMISSIONS_DAY)
    assert limit is not None
    assert (limit.remaining, limit.subject, limit.on_exhausted) == (1, QuotaSubject.user, OnExhausted.block)
    assert gateway.request().url.path == routes.USERS_ME
    assert gateway.request().headers['authorization'] == f'Bearer {KEY}'


def test_create_submission_sends_the_run_defining_fields():
    gateway = Gateway(200, {'submission_id': '1f', 'status': 'pending', 'policy_image_digest': 'sha256:abc'})
    client = make_client(gateway)

    response = client.create_submission(
        SubmissionCreateRequest(policy_image=PolicyImage('org/policy:v1'), eval=EvalRef('fake.smoke'))
    )

    assert isinstance(response, SubmissionCreateResponse)
    assert response.submission_id == 0x1F
    assert response.status is SubmissionStatus.pending
    assert gateway.request().url.path == routes.SUBMISSIONS_CREATE
    assert gateway.body()['policy_image'] == 'org/policy:v1'
    assert gateway.body()['transaction_key'] is None


def test_create_submission_reports_a_terminal_unpullable_image_as_a_response():
    gateway = Gateway(200, {'submission_id': '1f', 'status': 'errored', 'reason_code': 'image_unpullable'})
    response = make_client(gateway).create_submission(
        SubmissionCreateRequest(policy_image=PolicyImage('nope'), eval=EvalRef('fake.smoke'))
    )
    assert response.status is SubmissionStatus.errored
    assert response.reason_code is ReasonCode.image_unpullable


def test_list_submissions_parses_every_row():
    gateway = Gateway(
        200,
        {
            'submissions': [
                {
                    'id': '1f',
                    'user_id': 'a0',
                    'alias': None,
                    'status': 'finished',
                    'eval': 'fake.smoke',
                    'received_at': AT,
                    'reason_code': None,
                }
            ]
        },
    )
    response = make_client(gateway).list_submissions()

    assert isinstance(response, SubmissionListResponse)
    assert response.submissions[0].status is SubmissionStatus.finished
    assert gateway.request().url.path == routes.SUBMISSIONS_LIST


def test_get_submission_sends_the_hex_id_and_resolves_the_variant():
    gateway = Gateway(200, {'id': '1f', 'received_at': AT, 'queued_at': AT, 'queue_position': 2, 'status': 'pending'})
    view = make_client(gateway).get_submission(SubmissionId(0x1F))

    assert isinstance(view, PendingSubmissionView)
    assert view.queue_position == 2
    assert gateway.request().url.path == routes.SUBMISSIONS_GET
    assert dict(gateway.request().url.params) == {'id': '1f'}


def test_a_redirect_is_a_failure_rather_than_a_body_to_parse():
    # httpx follows no redirect by default, so a 3xx arrives here with a body that is not an envelope.
    gateway = Gateway(302, {'error': {'code': 'not_found', 'message': 'moved'}})
    with pytest.raises(PlatformError) as raised:
        make_client(gateway).me()
    assert raised.value.http_status == 302


def test_a_numeric_id_is_refused_at_the_boundary():
    # The wire contract is hex text. Decoding the body first would have taken the number.
    gateway = Gateway(200, {'submission_id': 31, 'status': 'pending'})
    with pytest.raises(ValidationError):
        make_client(gateway).create_submission(
            SubmissionCreateRequest(policy_image=PolicyImage('org/policy:v1'), eval=EvalRef('fake.smoke'))
        )


def test_cancel_submission_posts_the_id():
    gateway = Gateway(200, {'status': 'cancelled', 'refunded': True})
    response = make_client(gateway).cancel_submission(CancelRequest(id=SubmissionId(0x1F)))

    assert isinstance(response, CancelResponse)
    assert response.refunded is True
    assert gateway.body() == {'id': '1f'}


def rankings_gateway(eval_version: str = 'smoke@0123456789ab') -> Gateway:
    return Gateway(
        200,
        {
            'board': 'smoke',
            'eval': 'fake.smoke',
            'eval_version': eval_version,
            'primary_metric': 'success_rate',
            'rankings': [
                {
                    'rank': 1,
                    'display_name': 'demo',
                    'tag': '0ddba7',
                    'scores': {'primary': 0.75},
                    'submission_id': '1f',
                    'submitted_at': AT,
                }
            ],
        },
    )


def test_rankings_names_the_board_and_omits_an_unset_version():
    gateway = rankings_gateway()

    response = make_client(gateway, api_key=None).rankings(board=BoardRef('smoke'))

    assert isinstance(response, RankingsResponse)
    assert response.board == 'smoke'
    assert response.rankings[0].scores.primary == 0.75
    assert dict(gateway.request().url.params) == {'board': 'smoke'}
    assert 'authorization' not in gateway.request().headers


def test_rankings_pins_a_past_board_when_a_version_is_given():
    gateway = rankings_gateway('smoke@old')
    make_client(gateway, api_key=None).rankings(board=BoardRef('smoke'), eval_version='smoke@old')
    assert dict(gateway.request().url.params)['eval_version'] == 'smoke@old'


def test_a_board_read_sends_the_key_when_one_is_set():
    gateway = rankings_gateway()
    make_client(gateway).rankings(board=BoardRef('nebius-2026/robolab/public_subset'))
    assert gateway.request().headers['authorization'] == f'Bearer {KEY}'


def test_list_boards_asks_without_a_key_and_parses_every_board():
    gateway = Gateway(
        200,
        {
            'boards': [
                {
                    'board': 'smoke',
                    'title': 'Smoke',
                    'eval': 'fake.smoke',
                    'eval_version': 'smoke@0123456789ab',
                    'primary_metric': 'success_rate',
                    'visibility': 'public',
                }
            ]
        },
    )

    response = make_client(gateway, api_key=None).list_boards()

    assert isinstance(response, BoardListResponse)
    assert response.boards[0].visibility is BoardVisibility.public
    assert gateway.request().url.path == routes.RANKINGS_LIST
    assert 'authorization' not in gateway.request().headers


def test_an_error_envelope_becomes_the_typed_exception():
    gateway = Gateway(
        400,
        {
            'error': {
                'code': 'bad_request',
                'message': 'image not pullable',
                'details': {REASON_CODE_DETAIL: 'image_unpullable'},
            }
        },
    )
    with pytest.raises(PlatformError) as raised:
        make_client(gateway).create_submission(
            SubmissionCreateRequest(policy_image=PolicyImage('nope'), eval=EvalRef('fake.smoke'))
        )

    assert raised.value.code is ErrorCode.bad_request
    assert raised.value.reason_code is ReasonCode.image_unpullable
    assert raised.value.http_status == 400


def test_an_authenticated_call_without_a_key_fails_before_the_request():
    gateway = Gateway(200, {})
    with pytest.raises(ValueError, match='API key'):
        make_client(gateway, api_key=None).me()
    assert gateway.seen is None


def test_a_client_given_nothing_reaches_the_default_platform(monkeypatch):
    # A user should never have to know a URL, so an unconfigured client is the ordinary case.
    monkeypatch.delenv(API_URL_ENV, raising=False)
    assert str(PlatformClient()._client.base_url) == DEFAULT_PLATFORM_URL


def test_the_url_precedence_is_argument_then_environment_then_the_default(monkeypatch):
    monkeypatch.setenv(API_URL_ENV, 'http://from.env')
    assert resolve_base_url('http://from.argument') == 'http://from.argument'
    assert resolve_base_url() == 'http://from.env'
    monkeypatch.delenv(API_URL_ENV)
    assert resolve_base_url() == DEFAULT_PLATFORM_URL


def test_the_key_precedence_is_argument_then_environment_then_none(monkeypatch):
    # Same shape as the URL above: a caller that exported the key its registration printed is
    # configured, and passing it a second time through every construction site is not the contract.
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_env')
    assert resolve_api_key(ApiKey('pk_live_argument')) == 'pk_live_argument'
    assert resolve_api_key() == 'pk_live_env'
    monkeypatch.delenv(API_KEY_ENV)
    assert resolve_api_key() is None


@pytest.mark.parametrize('empty', ['', '   '])
def test_a_url_supplied_but_empty_is_refused_rather_than_read_as_unset(monkeypatch, empty):
    # `--platform-url=` names a platform, so falling through would send that run to the default —
    # the production platform — which is the one place it was least meant to go.
    monkeypatch.setenv(API_URL_ENV, 'http://from.env')
    with pytest.raises(ValueError, match='base_url is empty'):
        resolve_base_url(empty)
    with pytest.raises(ValueError, match='base_url is empty'):
        PlatformClient(empty)

    monkeypatch.setenv(API_URL_ENV, empty)
    with pytest.raises(ValueError, match=f'{API_URL_ENV} is empty'):
        resolve_base_url()


def test_a_client_of_your_own_carries_its_own_base_url():
    with pytest.raises(ValueError, match='one or the other'):
        PlatformClient(BASE, client=httpx.Client())


def test_a_supplied_client_with_no_base_url_is_refused():
    # Every endpoint sends a path relative to it, so without one the request reaches no host at all.
    with pytest.raises(ValueError, match='no base_url'):
        PlatformClient(client=httpx.Client())


def test_closing_leaves_a_caller_supplied_client_open():
    supplied = httpx.Client(base_url=BASE, transport=httpx.MockTransport(Gateway(200, {})))
    PlatformClient(client=supplied).close()
    assert not supplied.is_closed

    owned = PlatformClient(BASE)
    owned.close()
    assert owned._client.is_closed


def test_every_endpoint_has_exactly_one_method():
    # A route is a path under the prefix, which `routes` also holds query-parameter names beside.
    declared = {
        name
        for name, value in vars(routes).items()
        if name != 'API_PREFIX' and isinstance(value, str) and value.startswith(routes.API_PREFIX)
    }
    methods = {
        'register',
        'me',
        'create_submission',
        'list_submissions',
        'get_submission',
        'cancel_submission',
        'rankings',
        'list_boards',
    }
    assert len(declared) == len(methods)
    assert methods <= set(vars(PlatformClient))


def test_an_unknown_eval_comes_back_carrying_the_ones_on_offer():
    # The platform owns the set, so a caller who names one it does not have learns the real names
    # from the refusal rather than from a list this client would have to keep current.
    gateway = Gateway(
        404,
        {
            'error': {
                'code': 'not_found',
                'message': "unknown eval 'fake.smokey'",
                'details': {EVALS_DETAIL: ['fake.smoke', 'robolab.public_subset']},
            }
        },
    )
    with pytest.raises(PlatformError) as caught:
        make_client(gateway).create_submission(
            SubmissionCreateRequest(policy_image=PolicyImage('org/policy:v1'), eval=EvalRef('fake.smokey'))
        )
    assert caught.value.evals == ['fake.smoke', 'robolab.public_subset']


def test_a_failure_that_names_no_evals_is_told_apart_from_one_that_names_none():
    gateway = Gateway(403, {'error': {'code': 'forbidden', 'message': 'no', 'details': {}}})
    with pytest.raises(PlatformError) as caught:
        make_client(gateway).list_submissions()
    assert caught.value.evals is None


def test_a_supplied_client_carrying_an_authorization_default_is_refused():
    # httpx merges client-level headers into every request, so a default here would reach
    # `users.register` — which this module declares unauthenticated.
    with pytest.raises(ValueError, match='Authorization'):
        PlatformClient(client=httpx.Client(base_url=BASE, headers={'Authorization': 'Bearer leaked'}))


def test_a_malformed_eval_list_raises_rather_than_reading_as_no_list():
    # A short list is worse than none: a caller would pick from it believing it whole.
    gateway = Gateway(404, {'error': {'code': 'not_found', 'message': 'x', 'details': {EVALS_DETAIL: 'fake.smoke'}}})
    with pytest.raises(PlatformError) as caught:
        make_client(gateway).list_submissions()
    with pytest.raises(ValidationError):
        _ = caught.value.evals


def test_a_malformed_quota_detail_raises_rather_than_reading_as_no_rule():
    gateway = Gateway(429, {'error': {'code': 'quota_exceeded', 'message': 'x', 'details': {'quota': 'all of it'}}})
    with pytest.raises(PlatformError) as caught:
        make_client(gateway).list_submissions()
    with pytest.raises(ValidationError):
        _ = caught.value.quota
