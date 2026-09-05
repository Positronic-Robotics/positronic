"""Every request and response model survives its own JSON form, and the status union routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest
from platform_client.boards import BoardRef
from platform_client.enums import (
    BoardVisibility,
    CameraVantage,
    EndpointKind,
    ErrorCode,
    KeyStatus,
    OnExhausted,
    Placement,
    QuotaSubject,
    ReasonCode,
    RequestStatus,
    SubmissionStatus,
)
from platform_client.errors import QUOTA_DETAIL, REASON_CODE_DETAIL, ApiErrorBody, ErrorEnvelope, PlatformError
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, RequestId, SubmissionId, TransactionKey, UserId
from platform_client.policy_images import PolicyImage
from platform_client.requests import (
    CancelRequest,
    EndpointAsk,
    RankingsQuery,
    RegisterRequest,
    RequestCreate,
    RequestGetQuery,
    RequestListQuery,
    SceneAsk,
    SubmissionCreateRequest,
    SubmissionGetQuery,
    TaskAsk,
)
from platform_client.responses import (
    ID_FIELD,
    QUOTA_SUBMISSIONS_CONCURRENT,
    QUOTA_SUBMISSIONS_DAY,
    STATUS_FIELD,
    ArtifactRefs,
    BoardListResponse,
    BoardSummary,
    CancelledSubmissionView,
    CancelResponse,
    EpisodeCounts,
    ErroredSubmissionView,
    FinishedSubmissionView,
    MeResponse,
    PendingSubmissionView,
    QuotaLimit,
    RankingRow,
    RankingsResponse,
    RegisterResponse,
    RequestCreated,
    RequestListResponse,
    RequestView,
    RunningSubmissionView,
    RunSummary,
    Scores,
    SubmissionCreateResponse,
    SubmissionListResponse,
    SubmissionListRow,
    SubmissionView,
)
from platform_client.slug import slug_of
from platform_client.tasks import TaskRef
from pydantic import BaseModel, Tag, TypeAdapter, ValidationError

AT = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
SUB = SubmissionId(0x1F)
USER = UserId(0xA0)

SCORES = Scores(primary=0.75)

DAILY = QuotaLimit(
    key=QUOTA_SUBMISSIONS_DAY,
    meter='submissions',
    unit='submission',
    scale=1,
    window='day',
    subject=QuotaSubject.user,
    limit=2,
    used=1,
    resets_at=AT,
    on_exhausted=OnExhausted.block,
)

CREDITS = QuotaLimit(
    key='credits.period',
    meter='credits',
    unit='credit',
    scale=6,
    window='24 Jul – 23 Aug',
    subject=QuotaSubject.tenant,
    scope=['real'],
    limit=600,
    used=630,
    resets_at=None,
    on_exhausted=OnExhausted.meter,
)

REQUEST = RequestId(0x2A)

SCENE = SceneAsk(
    tote_placement=Placement.random, camera_vantage=CameraVantage.phail, external_cameras={'side': Placement.left}
)

ASK = RequestCreate(
    tasks=[
        TaskAsk(task_id=TaskRef('eight-spoons-into-grey-tote')),
        TaskAsk(
            task_id=TaskRef('stack-the-cubes'),
            episodes_per_endpoint=2,
            cap_per_episode_sec=90,
            policy_preset='other',
            scene=SCENE,
            endpoints=[EndpointAsk(name='gyros'), EndpointAsk(name='ours', url='wss://ours.example/ws')],
        ),
    ],
    endpoints=[
        EndpointAsk(name='gyros', url='wss://gyros.example/ws'),
        EndpointAsk(name='pi05', kind=EndpointKind.served, provider='droid_cohost', spec='pi05'),
    ],
    episodes_per_endpoint=10,
    cap_per_episode_sec=180,
    policy_preset='runway_ziyi',
    scene=SceneAsk(tote_placement=Placement.left),
    slug='ziyi',
    transaction_key=TransactionKey('round-1'),
)

VIEW = RequestView(
    request_id=REQUEST,
    status=RequestStatus.running,
    slug='2026-09-04-runway-ziyi',
    episodes=EpisodeCounts(total=24, done=3, outstanding=21),
    runs=[RunSummary(run_tag='blind_20260904-160621', started_at=AT), RunSummary(run_tag='blind_20260904-170000')],
    artifacts='s3://inference/runway/040926/ziyi-0/',
)

SUBMISSION_VIEWS = TypeAdapter(SubmissionView)

MODELS: list[BaseModel] = [
    Scores(),
    SCORES,
    DAILY,
    CREDITS,
    ArtifactRefs(result='s3://pp-artifacts/users/a0/submissions/1f/result.json'),
    RegisterRequest(credential='token', alias='demo', rotate=True),
    SubmissionCreateRequest(
        policy_image=PolicyImage('org/policy:v1'),
        eval=EvalRef('fake.smoke'),
        alias='demo',
        transaction_key=TransactionKey('key-1'),
    ),
    CancelRequest(id=SUB),
    SubmissionGetQuery(id=SUB),
    RankingsQuery(board=BoardRef('smoke')),
    RegisterResponse(
        user_id=USER,
        artifact_location='s3://pp-artifacts/users/a0/',
        api_key=ApiKey('pk_live_secret'),
        key_status=KeyStatus.created,
    ),
    RegisterResponse(user_id=USER, artifact_location='s3://pp-artifacts/users/a0/', key_status=KeyStatus.existing),
    MeResponse(user_id=USER, alias='demo', tenant='nebius-2026', plan='nebius_competition_2026', quota=[DAILY]),
    SubmissionCreateResponse(submission_id=SUB, status=SubmissionStatus.pending, policy_image_digest='sha256:abc'),
    SubmissionCreateResponse(
        submission_id=SUB, status=SubmissionStatus.errored, reason_code=ReasonCode.image_unpullable
    ),
    SubmissionListResponse(),
    SubmissionListResponse(
        submissions=[
            SubmissionListRow(
                id=SUB,
                user_id=USER,
                alias='demo',
                status=SubmissionStatus.running,
                eval=EvalRef('fake.smoke'),
                received_at=AT,
            )
        ]
    ),
    PendingSubmissionView(id=SUB, alias='demo', received_at=AT, queued_at=AT, queue_position=1),
    RunningSubmissionView(id=SUB, running_since=AT, stage='evaluating', stage_detail='task 2/10'),
    ErroredSubmissionView(id=SUB, reason_code=ReasonCode.policy_oom, reason='policy ran out of memory'),
    FinishedSubmissionView(id=SUB, scores=SCORES, artifacts=ArtifactRefs(result='s3://b/result.json')),
    CancelledSubmissionView(id=SUB, cancelled_at=AT),
    CancelResponse(status=SubmissionStatus.cancelled, refunded=True),
    RankingsResponse(
        board=BoardRef('smoke'),
        eval=EvalRef('fake.smoke'),
        primary_metric='success_rate',
        rankings=[
            RankingRow(rank=1, display_name='demo', tag='0ddba7', scores=SCORES, submission_id=SUB, submitted_at=AT)
        ],
    ),
    BoardListResponse(),
    BoardListResponse(
        boards=[
            BoardSummary(
                board=BoardRef('smoke'),
                title='Smoke',
                eval=EvalRef('fake.smoke'),
                primary_metric='success_rate',
                visibility=BoardVisibility.public,
            )
        ]
    ),
    ErrorEnvelope(error=ApiErrorBody(code=ErrorCode.quota_exceeded, message='daily quota spent')),
    ASK,
    RequestCreate(
        tasks=[TaskAsk(task_id=TaskRef('stack-the-cubes'))], endpoints=[EndpointAsk(name='a')], episodes_per_endpoint=1
    ),
    RequestGetQuery(id=REQUEST),
    RequestListQuery(after=REQUEST, limit=50),
    RequestListQuery(),
    RequestCreated(request_id=REQUEST, status=RequestStatus.received),
    VIEW,
    RequestView(
        request_id=REQUEST,
        status=RequestStatus.errored,
        episodes=EpisodeCounts(total=1, done=0, outstanding=1),
        error='no task named it',
    ),
    RequestListResponse(requests=[VIEW], next=REQUEST),
    RequestListResponse(),
]


@pytest.mark.parametrize('model', MODELS, ids=lambda m: type(m).__name__)
def test_a_model_round_trips_through_its_json_form(model: BaseModel):
    assert type(model).model_validate(model.model_dump(mode='json')) == model


@pytest.mark.parametrize('model', MODELS, ids=lambda m: type(m).__name__)
def test_a_model_round_trips_through_a_real_json_string(model: BaseModel):
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_ids_and_statuses_leave_as_wire_values():
    payload = SubmissionListResponse(
        submissions=[
            SubmissionListRow(
                id=SUB,
                user_id=USER,
                status=SubmissionStatus.errored,
                eval=EvalRef('fake.smoke'),
                received_at=AT,
                reason_code=ReasonCode.image_unpullable,
            )
        ]
    ).model_dump(mode='json')
    row = payload['submissions'][0]
    assert row['id'] == '1f'
    assert row['user_id'] == 'a0'
    assert row['status'] == 'errored'
    assert row['reason_code'] == 'image_unpullable'
    assert row['received_at'].startswith('2026-03-04T05:06:07')


def test_a_request_rejects_an_unknown_field():
    with pytest.raises(ValidationError):
        SubmissionCreateRequest.model_validate({
            'policy_image': 'i',
            'eval': 'fake.smoke',
            'evals': 'fake.smoke',  # a plausible typo of eval
        })


def test_a_policy_image_the_registry_could_never_resolve_is_refused_here():
    with pytest.raises(ValidationError):
        SubmissionCreateRequest.model_validate({
            'policy_image': 'org/policy@',  # a digest separator with nothing behind it
            'eval': 'fake.smoke',
        })


def test_a_digest_pinned_image_is_taken_whole_and_parsed():
    request = SubmissionCreateRequest(policy_image=PolicyImage('org/policy@sha256:abc'), eval=EvalRef('fake.smoke'))
    assert isinstance(request.policy_image, PolicyImage)
    assert request.policy_image.name == 'org/policy'
    assert request.policy_image.digest == 'sha256:abc'


def test_a_reason_code_is_refused_on_a_status_that_did_not_fail():
    # `ReasonCode` says why a run FAILED. A pending payload carrying one is a malformed response,
    # and validating it would report a submission as accepted while naming the reason it was not.
    with pytest.raises(ValidationError):
        SubmissionCreateResponse.model_validate({
            'submission_id': '1f',
            'status': 'pending',
            'reason_code': 'image_unpullable',
        })


def test_a_listed_row_refuses_the_same_pairing():
    with pytest.raises(ValidationError):
        SubmissionListRow.model_validate({
            'id': '1f',
            'user_id': 'a0',
            'status': 'finished',
            'eval': 'fake.smoke',
            'received_at': '2026-08-13T10:00:00Z',
            'reason_code': 'policy_oom',
        })


def test_an_errored_row_keeps_its_reason_and_a_clean_one_needs_none():
    # The boundary of the rule above: it constrains the PAIRING, not either field on its own.
    errored = SubmissionCreateResponse.model_validate({
        'submission_id': '1f',
        'status': 'errored',
        'reason_code': 'image_unpullable',
    })
    assert errored.reason_code is ReasonCode.image_unpullable
    assert SubmissionCreateResponse.model_validate({'submission_id': '1f', 'status': 'pending'}).reason_code is None
    # An errored submission need not say why — the taxonomy is optional, the pairing is not.
    assert SubmissionCreateResponse.model_validate({'submission_id': '1f', 'status': 'errored'}).reason_code is None


def test_a_board_slug_that_could_never_be_one_is_refused():
    with pytest.raises(ValidationError):
        RankingsQuery.model_validate({'board': '   '})
    with pytest.raises(ValidationError):
        RankingsQuery.model_validate({'board': ''})


def test_a_board_slug_arrives_as_its_own_type_on_both_sides():
    assert isinstance(RankingsQuery(board=BoardRef('smoke')).board, BoardRef)
    listed = BoardListResponse.model_validate({
        'boards': [
            {
                'board': 'smoke',
                'title': 'Smoke',
                'eval': 'fake.smoke',
                'primary_metric': 'success_rate',
                'visibility': 'public',
            }
        ]
    })
    assert isinstance(listed.boards[0].board, BoardRef)


def test_an_id_reaches_the_query_string_in_its_hex_wire_form():
    assert SubmissionGetQuery(id=SUB).model_dump(mode='json') == {'id': SUB.to_str()}


def test_an_empty_transaction_key_is_a_client_bug_not_an_absent_one():
    with pytest.raises(ValidationError):
        SubmissionCreateRequest.model_validate({'policy_image': 'i', 'eval': 'fake.smoke', 'transaction_key': ''})


@pytest.mark.parametrize(
    ('payload', 'expected'),
    [
        (
            {'id': '1f', 'received_at': AT, 'queued_at': AT, 'queue_position': 3, 'status': 'pending'},
            PendingSubmissionView,
        ),
        ({'id': '1f', 'running_since': AT, 'stage': 'evaluating', 'status': 'running'}, RunningSubmissionView),
        ({'id': '1f', 'reason_code': 'policy_oom', 'reason': 'oom', 'status': 'errored'}, ErroredSubmissionView),
        (
            {'id': '1f', 'scores': {}, 'artifacts': {'result': 's3://b/result.json'}, 'status': 'finished'},
            FinishedSubmissionView,
        ),
        ({'id': '1f', 'cancelled_at': AT, 'status': 'cancelled'}, CancelledSubmissionView),
    ],
    ids=['pending', 'running', 'errored', 'finished', 'cancelled'],
)
def test_the_status_slug_selects_the_view_variant(payload: dict, expected: type[BaseModel]):
    assert type(SUBMISSION_VIEWS.validate_python(payload)) is expected


@pytest.mark.parametrize(
    'variant',
    [
        PendingSubmissionView,
        RunningSubmissionView,
        ErroredSubmissionView,
        FinishedSubmissionView,
        CancelledSubmissionView,
    ],
)
def test_the_published_field_names_are_ones_every_variant_declares(variant: type[BaseModel]):
    # `positronic eval status` prints these two on its header line and excludes them from the body
    # by these names, so a name that outlived its field would print it twice.
    assert {ID_FIELD, STATUS_FIELD} <= set(variant.model_fields)


def test_a_view_refuses_a_status_that_is_not_its_own_tag():
    # `submitting` is an internal state the union has no variant for; a gateway building a pending
    # view from such a record must fail here rather than emit a tag no caller can route.
    with pytest.raises(ValidationError):
        PendingSubmissionView(
            id=SUB, received_at=AT, queued_at=AT, queue_position=1, status=SubmissionStatus.submitting
        )


def test_a_view_keeps_its_own_tag():
    view = PendingSubmissionView(id=SUB, received_at=AT, queued_at=AT, queue_position=1)
    assert view.status is SubmissionStatus.pending


def test_every_variant_is_tagged_with_the_slug_of_the_status_it_declares():
    # The discriminator computes a tag from the payload's slug, so a tag spelled any other way names
    # a wire value nothing produces and the variant becomes unreachable.
    variants = get_args(get_args(SubmissionView)[0])
    assert len(variants) == 5
    for variant in variants:
        model, tag = get_args(variant)
        assert isinstance(tag, Tag)
        assert tag.tag == slug_of(model.model_fields[STATUS_FIELD].default)


def test_a_minted_outcome_without_its_key_is_refused():
    with pytest.raises(ValidationError):
        RegisterResponse(user_id=USER, artifact_location='s3://b/', key_status=KeyStatus.created)


def test_an_existing_registration_carrying_a_key_is_refused():
    with pytest.raises(ValidationError):
        RegisterResponse(user_id=USER, artifact_location='s3://b/', api_key=ApiKey('pk'), key_status=KeyStatus.existing)


def test_each_outcome_paired_with_its_own_key_state_is_kept():
    minted = RegisterResponse(
        user_id=USER, artifact_location='s3://b/', api_key=ApiKey('pk'), key_status=KeyStatus.rotated
    )
    existing = RegisterResponse(user_id=USER, artifact_location='s3://b/', key_status=KeyStatus.existing)
    assert minted.api_key is not None and existing.api_key is None


def test_an_unknown_status_does_not_resolve_to_a_variant():
    with pytest.raises(ValidationError):
        SUBMISSION_VIEWS.validate_python({'id': '1f', 'status': 'submitting'})


def test_a_view_round_trips_back_to_its_own_variant():
    view = RunningSubmissionView(id=SUB, running_since=AT, stage='scoring')
    assert SUBMISSION_VIEWS.validate_python(SUBMISSION_VIEWS.dump_python(view, mode='json')) == view


def test_the_error_exception_exposes_the_envelope_and_the_reason_code():
    payload = {
        'error': {
            'code': 'bad_request',
            'message': 'image not pullable',
            'details': {REASON_CODE_DETAIL: 'image_unpullable'},
        }
    }
    err = PlatformError.from_payload(400, payload)
    assert err.code is ErrorCode.bad_request
    assert err.message == 'image not pullable'
    assert err.reason_code is ReasonCode.image_unpullable
    assert err.http_status == 400


def test_an_unparseable_error_body_still_raises_the_same_exception():
    err = PlatformError.from_payload(502, '<html>bad gateway</html>')
    assert err.code is ErrorCode.internal_error
    assert err.http_status == 502
    assert err.details['body'] == '<html>bad gateway</html>'


def test_a_reason_code_this_client_cannot_read_is_invalid_rather_than_absent():
    err = PlatformError.from_payload(
        400,
        {'error': {'code': 'bad_request', 'message': 'm', 'details': {REASON_CODE_DETAIL: 'from_a_newer_taxonomy'}}},
    )
    assert err.reason_code is ReasonCode.INVALID


def test_a_failure_carrying_no_reason_reports_none():
    err = PlatformError.from_payload(400, {'error': {'code': 'bad_request', 'message': 'm'}})
    assert err.reason_code is None


def test_a_defect_inside_validation_surfaces_instead_of_becoming_an_unparseable_body(monkeypatch):
    def boom(_payload: object) -> ErrorEnvelope:
        raise RuntimeError('a validator bug, not a malformed payload')

    monkeypatch.setattr(ErrorEnvelope, 'model_validate', staticmethod(boom))
    with pytest.raises(RuntimeError):
        PlatformError.from_payload(500, {'error': {'code': 'internal_error', 'message': 'm'}})


def test_a_queue_position_below_one_is_refused():
    with pytest.raises(ValidationError):
        PendingSubmissionView(id=SUB, received_at=AT, queued_at=AT, queue_position=0)


def test_the_first_place_in_the_queue_is_kept():
    assert PendingSubmissionView(id=SUB, received_at=AT, queued_at=AT, queue_position=1).queue_position == 1


def test_an_error_without_a_reason_code_reports_none():
    err = PlatformError.from_payload(429, {'error': {'code': 'quota_exceeded', 'message': 'spent'}})
    assert err.reason_code is None


def test_a_quota_limit_leaves_with_its_slugs_and_an_empty_scope():
    payload = DAILY.model_dump(mode='json')
    assert payload['on_exhausted'] == 'block'
    assert payload['subject'] == 'user'
    assert payload['scope'] == []
    assert CREDITS.model_dump(mode='json')['scope'] == ['real']


def test_remaining_is_computed_and_never_negative():
    assert DAILY.remaining == 1
    assert CREDITS.remaining == 0
    assert 'remaining' not in DAILY.model_dump(mode='json')


def test_the_published_keys_are_the_ones_a_caller_looks_up():
    me = MeResponse(user_id=USER, tenant='t', plan='p', quota=[DAILY])
    assert me.quota_for(QUOTA_SUBMISSIONS_DAY) is DAILY
    assert me.quota_for(QUOTA_SUBMISSIONS_CONCURRENT) is None  # a plan need not declare every rule


def test_a_limit_is_found_by_its_rule_key():
    me = MeResponse(user_id=USER, tenant='nebius-2026', plan='nebius_competition_2026', quota=[DAILY, CREDITS])
    assert me.quota_for('credits.period') is CREDITS
    assert me.quota_for(QUOTA_SUBMISSIONS_CONCURRENT) is None


def test_a_quota_refusal_carries_the_whole_rule_that_refused_it():
    err = PlatformError.from_payload(
        429,
        {
            'error': {
                'code': 'quota_exceeded',
                'message': 'daily submission quota exhausted',
                'details': {
                    QUOTA_DETAIL: {
                        'key': QUOTA_SUBMISSIONS_DAY,
                        'meter': 'submissions',
                        'unit': 'submission',
                        'scale': 1,
                        'window': 'day',
                        'subject': 'user',
                        'scope': [],
                        'limit': 2,
                        'used': 2,
                        'resets_at': '2026-08-12T00:00:00Z',
                        'on_exhausted': 'block',
                    }
                },
            }
        },
    )
    assert err.quota is not None
    assert err.quota.key == QUOTA_SUBMISSIONS_DAY
    assert err.quota.remaining == 0
    assert err.quota.on_exhausted is OnExhausted.block
    assert err.quota.resets_at == datetime(2026, 8, 12, tzinfo=UTC)


def test_an_error_without_a_quota_detail_reports_none():
    err = PlatformError.from_payload(400, {'error': {'code': 'bad_request', 'message': 'nope'}})
    assert err.quota is None


def test_a_scale_of_zero_is_refused_at_the_boundary():
    # Consumers divide by it, so a zero would validate here and raise ZeroDivisionError there.
    payload = DAILY.model_dump(mode='json') | {'scale': 0}
    with pytest.raises(ValidationError):
        QuotaLimit.model_validate(payload)


@pytest.mark.parametrize(
    'model, payload',
    [
        (SubmissionCreateResponse, {'submission_id': 'ff', 'status': 'submitting'}),
        (CancelResponse, {'status': 'submitting', 'refunded': False}),
    ],
)
def test_the_internal_claim_state_never_reaches_a_caller(model: type[BaseModel], payload: dict):
    # The enum says the gateway reports `submitting` as `pending`; a payload carrying it is a
    # gateway that forgot, refused here rather than left for every consumer to normalise.
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize('status', ['pending', 'running', 'finished', 'errored', 'cancelled'])
def test_every_status_a_caller_can_see_is_kept(status: str):
    assert SubmissionCreateResponse.model_validate({'submission_id': 'ff', 'status': status}).status.name == status


# --- the request models' own rules -----------------------------------------------------------


def test_a_request_counts_every_episode_it_asks_for():
    # Task one inherits: two request endpoints, ten each. Task two states its own: two endpoints, two each.
    assert ASK.episodes_total == 2 * 10 + 2 * 2
    assert [entry.name for entry in ASK.task_endpoints(ASK.tasks[0])] == ['gyros', 'pi05']
    assert [entry.name for entry in ASK.task_endpoints(ASK.tasks[1])] == ['gyros', 'ours']


def test_a_served_endpoint_names_its_bring_up_and_no_address():
    with pytest.raises(ValidationError, match='no provider or no spec'):
        EndpointAsk(name='pi05', kind=EndpointKind.served)
    with pytest.raises(ValidationError, match='names a url'):
        EndpointAsk(name='pi05', kind=EndpointKind.served, provider='droid_cohost', spec='pi05', url='wss://x/ws')
    with pytest.raises(ValidationError, match='only a served endpoint carries'):
        EndpointAsk(name='gyros', provider='droid_cohost')


def test_an_endpoint_says_whether_it_names_a_locator():
    assert EndpointAsk(name='gyros').names_a_locator is False
    assert EndpointAsk(name='gyros', url='wss://x/ws').names_a_locator is True
    assert EndpointAsk(name='pi05', kind=EndpointKind.served, provider='droid_cohost', spec='pi05').names_a_locator


def test_a_request_names_each_task_and_each_endpoint_once():
    with pytest.raises(ValidationError, match='more than once'):
        RequestCreate(
            tasks=[TaskAsk(task_id=TaskRef('a')), TaskAsk(task_id=TaskRef('a'))],
            endpoints=[EndpointAsk(name='e')],
            episodes_per_endpoint=1,
        )
    with pytest.raises(ValidationError, match='more than once'):
        RequestCreate(
            tasks=[TaskAsk(task_id=TaskRef('a'))],
            endpoints=[EndpointAsk(name='e'), EndpointAsk(name='e')],
            episodes_per_endpoint=1,
        )
    with pytest.raises(ValidationError, match='more than once'):
        TaskAsk(task_id=TaskRef('a'), endpoints=[EndpointAsk(name='e'), EndpointAsk(name='e')])


def test_a_task_endpoint_naming_no_locator_names_one_the_request_defines():
    with pytest.raises(ValidationError, match='name no endpoint the request defines'):
        RequestCreate(
            tasks=[TaskAsk(task_id=TaskRef('a'), endpoints=[EndpointAsk(name='elsewhere')])],
            endpoints=[EndpointAsk(name='e')],
            episodes_per_endpoint=1,
        )
    # An entry that carries its own address needs no definition.
    RequestCreate(
        tasks=[TaskAsk(task_id=TaskRef('a'), endpoints=[EndpointAsk(name='elsewhere', url='wss://x/ws')])],
        episodes_per_endpoint=1,
    )


def test_every_task_runs_on_at_least_one_endpoint():
    with pytest.raises(ValidationError, match='runs on no endpoint'):
        RequestCreate(tasks=[TaskAsk(task_id=TaskRef('a'))], episodes_per_endpoint=1)
    with pytest.raises(ValidationError):
        TaskAsk(task_id=TaskRef('a'), endpoints=[])


def test_a_count_below_one_is_refused_at_every_level():
    with pytest.raises(ValidationError):
        RequestCreate(tasks=[TaskAsk(task_id=TaskRef('a'))], endpoints=[EndpointAsk(name='e')], episodes_per_endpoint=0)
    with pytest.raises(ValidationError):
        TaskAsk(task_id=TaskRef('a'), episodes_per_endpoint=0)
    with pytest.raises(ValidationError):
        RequestListQuery(limit=0)


def test_a_view_carries_an_error_only_when_it_errored():
    with pytest.raises(ValidationError, match='an error on a running request'):
        RequestView.model_validate({
            'request_id': '2a',
            'status': 'running',
            'episodes': {'total': 1, 'done': 0, 'outstanding': 1},
            'error': 'x',
        })


def test_the_scene_leaves_as_slugs():
    assert SCENE.model_dump(mode='json') == {
        'tote_placement': 'random',
        'camera_vantage': 'phail',
        'external_cameras': {'side': 'left'},
    }
    with pytest.raises(ValidationError):
        SceneAsk.model_validate({'tote_placement': 'middle'})
