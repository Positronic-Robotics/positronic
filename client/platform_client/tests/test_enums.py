"""The persisted enum values are pinned.

Every value here is stored durably, so this test is the ratchet: adding a member is expected and
updates the map below; changing or reusing a value silently re-reads every existing row as
something else, and that is what must fail.
"""

from __future__ import annotations

from enum import IntEnum

import pytest
from platform_client.enums import (
    ACTIVE_STATUSES,
    REQUEST_TERMINAL_STATUSES,
    TERMINAL_STATUSES,
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

ERROR_CODE_VALUES = {
    'INVALID': 0,
    'bad_request': 1,
    'unauthorized': 2,
    'forbidden': 3,
    'not_found': 4,
    'quota_exceeded': 5,
    'transaction_conflict': 6,
    'registry_unreachable': 7,
    'upstream_unavailable': 8,
    'eval_unavailable': 9,
    'internal_error': 10,
}

REASON_CODE_VALUES = {
    'INVALID': 0,
    'image_unpullable': 1,
    'image_too_large': 2,
    'invalid_flags': 3,
    'policy_setup_crash': 4,
    'policy_inference_crash': 5,
    'policy_oom': 6,
    'latency_budget_exceeded': 7,
    'wall_clock_exceeded': 8,
    'internal_error': 9,
    'quota_exceeded': 10,
    'provision_wedged': 11,
    'runner_unresponsive': 12,
}

SUBMISSION_STATUS_VALUES = {
    'INVALID': 0,
    'pending': 1,
    'submitting': 2,
    'running': 3,
    'finished': 4,
    'errored': 5,
    'cancelled': 6,
}

KEY_STATUS_VALUES = {'INVALID': 0, 'created': 1, 'existing': 2, 'rotated': 3}

ON_EXHAUSTED_VALUES = {'INVALID': 0, 'block': 1, 'meter': 2}

QUOTA_SUBJECT_VALUES = {'INVALID': 0, 'user': 1, 'tenant': 2}

BOARD_VISIBILITY_VALUES = {'INVALID': 0, 'public': 1, 'tenant': 2}

REQUEST_STATUS_VALUES = {'INVALID': 0, 'received': 1, 'filed': 2, 'running': 3, 'done': 4, 'cancelled': 5, 'errored': 6}

ENDPOINT_KIND_VALUES = {'INVALID': 0, 'remote': 1, 'served': 2}

PLACEMENT_VALUES = {'INVALID': 0, 'left': 1, 'right': 2, 'random': 3, 'none': 4}

CAMERA_VANTAGE_VALUES = {'INVALID': 0, 'droid': 1, 'phail': 2}

PERSISTED_ENUMS: list[tuple[type[IntEnum], dict[str, int]]] = [
    (ErrorCode, ERROR_CODE_VALUES),
    (ReasonCode, REASON_CODE_VALUES),
    (SubmissionStatus, SUBMISSION_STATUS_VALUES),
    (KeyStatus, KEY_STATUS_VALUES),
    (OnExhausted, ON_EXHAUSTED_VALUES),
    (QuotaSubject, QUOTA_SUBJECT_VALUES),
    (BoardVisibility, BOARD_VISIBILITY_VALUES),
    (RequestStatus, REQUEST_STATUS_VALUES),
    (EndpointKind, ENDPOINT_KIND_VALUES),
    (Placement, PLACEMENT_VALUES),
    (CameraVantage, CAMERA_VANTAGE_VALUES),
]


@pytest.mark.parametrize(('enum_cls', 'expected'), PERSISTED_ENUMS, ids=lambda p: getattr(p, '__name__', ''))
def test_the_name_to_value_mapping_is_pinned(enum_cls: type[IntEnum], expected: dict[str, int]):
    assert {m.name: m.value for m in enum_cls} == expected


@pytest.mark.parametrize(('enum_cls', 'expected'), PERSISTED_ENUMS, ids=lambda p: getattr(p, '__name__', ''))
def test_zero_is_the_unset_sentinel(enum_cls: type[IntEnum], expected: dict[str, int]):
    assert enum_cls(0).name == 'INVALID'


@pytest.mark.parametrize(('enum_cls', 'expected'), PERSISTED_ENUMS, ids=lambda p: getattr(p, '__name__', ''))
def test_no_value_is_reused(enum_cls: type[IntEnum], expected: dict[str, int]):
    assert len(set(expected.values())) == len(expected)


def test_the_status_sets_partition_the_decided_from_the_undecided():
    assert ACTIVE_STATUSES & TERMINAL_STATUSES == frozenset()
    assert ACTIVE_STATUSES | TERMINAL_STATUSES == set(SubmissionStatus) - {SubmissionStatus.INVALID}


def test_the_request_terminal_set_is_what_the_coordinator_is_finished_with():
    assert REQUEST_TERMINAL_STATUSES == {RequestStatus.done, RequestStatus.cancelled, RequestStatus.errored}
    assert (
        RequestStatus.received not in REQUEST_TERMINAL_STATUSES and RequestStatus.filed not in REQUEST_TERMINAL_STATUSES
    )
