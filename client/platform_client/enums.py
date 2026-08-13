"""The closed sets a caller sees: error codes, terminal reason codes, submission and key status.

Every member carries an explicit value and `INVALID = 0` is the unset/parse-failure sentinel. The
values are stored durably, so they are append-only forever: add members, never renumber or reuse.
Caller-visible surfaces carry the slug (`name.lower()`) — see `platform_client.slug`.
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
