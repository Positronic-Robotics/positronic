"""The persisted enum values are pinned.

The platform stores these integers, so this test is the ratchet: adding a member is expected and
updates the map below; changing or reusing a value silently re-reads every existing row as
something else, and that is what must fail. A value moves only beside the migration that moves the
rows, which is how `SubmissionStatus` closed the gaps two platform-only states had left.
"""

from __future__ import annotations

from enum import IntEnum

import pytest
from platform_client.enums import (
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
    'running': 2,
    'finished': 3,
    'errored': 4,
    'cancelled': 5,
    'blocked': 6,
}

KEY_STATUS_VALUES = {'INVALID': 0, 'created': 1, 'existing': 2, 'rotated': 3}

ON_EXHAUSTED_VALUES = {'INVALID': 0, 'block': 1, 'meter': 2}

QUOTA_SUBJECT_VALUES = {'INVALID': 0, 'user': 1, 'tenant': 2}

BOARD_VISIBILITY_VALUES = {'INVALID': 0, 'public': 1, 'tenant': 2}

ENDPOINT_KIND_VALUES = {'INVALID': 0, 'remote': 1, 'served': 2, 'image': 3}

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


def test_every_status_is_terminal_or_still_in_flight():
    # `blocked` is in flight too: it is undecided, and a later report moves it on. Naming the
    # in-flight ones here fails when a status arrives that belongs to neither camp.
    in_flight = {SubmissionStatus.pending, SubmissionStatus.running, SubmissionStatus.blocked}
    assert in_flight & TERMINAL_STATUSES == frozenset()
    assert in_flight | TERMINAL_STATUSES == set(SubmissionStatus) - {SubmissionStatus.INVALID}


def test_the_platform_only_states_are_absent():
    """A state no caller is shown is the platform's own, and it keeps those in a field of its own."""
    assert {m.name for m in SubmissionStatus}.isdisjoint({'submitting', 'mirroring'})
