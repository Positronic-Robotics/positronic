"""The closed sets a caller sees: error codes, terminal reason codes, submission and key status.

The values are stored durably, so members are append-only forever: add, never renumber or reuse.
`INVALID = 0` is the unset/parse-failure sentinel; the wire form is the slug (`platform_client.slug`).
"""

from __future__ import annotations

from enum import IntEnum, unique


@unique
class ErrorCode(IntEnum):
    """The `code` in an error envelope — the outcome of a REQUEST.

    A run that fails is reported through submission status plus a `ReasonCode`, never one of these.
    """

    INVALID = 0
    bad_request = 1
    unauthorized = 2
    forbidden = 3
    not_found = 4
    quota_exceeded = 5
    transaction_conflict = 6  # one transaction key, two different submissions
    registry_unreachable = 7  # retryable: no submission row, no quota event
    # Retryable 503: an upstream dependency (the identity provider, the orchestrator) timed out,
    # 5xx'd or rate-limited — distinct from a real auth failure, which is a 401.
    upstream_unavailable = 8
    # The eval exists but is not accepting submissions; one that does not exist is a not_found,
    # whose details carry the evals this platform does offer.
    eval_unavailable = 9
    internal_error = 10


@unique
class ReasonCode(IntEnum):
    """Closed terminal-failure taxonomy for a submission.

    Split by fault: a platform fault refunds the caller's quota, a caller fault charges it.
    """

    INVALID = 0

    # caller faults (charged)
    image_unpullable = 1
    image_too_large = 2
    invalid_flags = 3
    policy_setup_crash = 4
    policy_inference_crash = 5
    policy_oom = 6
    latency_budget_exceeded = 7
    wall_clock_exceeded = 8

    # platform / provider faults (refunded)
    internal_error = 9
    quota_exceeded = 10
    provision_wedged = 11
    runner_unresponsive = 12


@unique
class SubmissionStatus(IntEnum):
    """The submission lifecycle: pending -> submitting -> running -> finished|errored|cancelled.

    `submitting` is the internal claim state; the gateway reports it as `pending`, so it never
    reaches a caller.
    """

    INVALID = 0
    pending = 1
    submitting = 2
    running = 3
    finished = 4
    errored = 5
    cancelled = 6


@unique
class KeyStatus(IntEnum):
    """What `users.register` did with the caller's API key.

    `existing` carries no key: the plaintext is unreconstructable, so a re-register without
    `rotate` returns none.
    """

    INVALID = 0
    created = 1
    existing = 2
    rotated = 3


@unique
class OnExhausted(IntEnum):
    """What a quota rule does once its limit is spent: refuse the request, or admit it and bill it."""

    INVALID = 0
    block = 1
    meter = 2


@unique
class QuotaSubject(IntEnum):
    """Whose allowance a quota rule draws on."""

    INVALID = 0
    user = 1
    tenant = 2


@unique
class BoardVisibility(IntEnum):
    """Who may read a board: anyone, or the members of the tenant that owns it."""

    INVALID = 0
    public = 1
    tenant = 2


# Charged, undecided, still holding a concurrency slot.
ACTIVE_STATUSES: frozenset[SubmissionStatus] = frozenset({
    SubmissionStatus.pending,
    SubmissionStatus.submitting,
    SubmissionStatus.running,
})

# Decided and immutable.
TERMINAL_STATUSES: frozenset[SubmissionStatus] = frozenset({
    SubmissionStatus.finished,
    SubmissionStatus.errored,
    SubmissionStatus.cancelled,
})

# Decided, and there will never be a result to read. Derived, so a fourth terminal status joins it
# by being terminal rather than by every caller remembering to name it.
NO_RESULT_STATUSES: frozenset[SubmissionStatus] = TERMINAL_STATUSES - {SubmissionStatus.finished}


@unique
class RequestStatus(IntEnum):
    """A customer request's lifecycle: received -> filed -> running -> done|cancelled|errored.

    `received` is the gateway's own row. `filed` is the coordinator holding it; every later status
    is what the coordinator reports back.
    """

    INVALID = 0
    received = 1
    filed = 2
    running = 3
    done = 4
    cancelled = 5
    errored = 6


@unique
class EndpointKind(IntEnum):
    """Where a request's policy comes from: an address the caller holds up, or a checkpoint the platform serves."""

    INVALID = 0
    remote = 1
    served = 2


@unique
class Placement(IntEnum):
    """Which side of the rig a piece of the scene sits on.

    `random` draws a side per run; `none` states the piece is absent.
    """

    INVALID = 0
    left = 1
    right = 2
    random = 3
    none = 4


@unique
class CameraVantage(IntEnum):
    """How steeply the external camera looks down: each value names the dataset whose geometry the rig matches."""

    INVALID = 0
    droid = 1
    phail = 2


# A request the coordinator has finished with, one way or another.
REQUEST_TERMINAL_STATUSES: frozenset[RequestStatus] = frozenset({
    RequestStatus.done,
    RequestStatus.cancelled,
    RequestStatus.errored,
})
