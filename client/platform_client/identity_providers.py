"""The identity providers the platform takes a sign-in from, and how long the gateway takes to check their tokens.

`users.register` checks the credential with its provider inside the call, so a caller has to wait out
the gateway's requests to that provider. A caller that gives up first leaves behind a user and a key it
can never read.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from platform_client.responses import AuthConfig, WorkOSSettings

# The scope the minted GitHub token carries: the profile and the verified email. It reads no repository
# and writes nothing, so the platform holds no power over the account.
IDENTITY_SCOPE = 'read:user user:email'

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


# The longest one provider request can stall, read off the phases so the two cannot drift. The gateway
# bounds each of its own provider requests by the same phases.
MAX_REQUEST_STALL_S = max_stall_s(REQUEST_TIMEOUT)


@dataclass(frozen=True)
class IdentityProvider:
    """A provider the platform takes a sign-in from, as the device flow and `users.register` see it."""

    name: str
    device_authorization_url: str
    token_url: str
    # Sent with the device-code request when set (RFC 8628 §3.1).
    scope: str | None
    # Where to look for a code mail the sign-in page sends; None for a provider that sends no mail.
    code_mail_hint: str | None = None


GITHUB = IdentityProvider(
    name='GitHub',
    device_authorization_url='https://github.com/login/device/code',
    token_url='https://github.com/login/oauth/access_token',
    scope=IDENTITY_SCOPE,
)

WORKOS = IdentityProvider(
    name='WorkOS',
    device_authorization_url='https://api.workos.com/user_management/authorize/device',
    token_url='https://api.workos.com/user_management/authenticate',
    scope=None,
    code_mail_hint=(
        'If the sign-in page sends you a code by email and it is not in your inbox, look in your spam folder.'
    ),
)


def timeout_after_deadline(deadline_s: float) -> httpx.Timeout:
    """The bound on `users.register` when the gateway's identity check takes up to `deadline_s`.

    The read phase covers that deadline plus one request for the gateway's own work.
    """
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=deadline_s + MAX_REQUEST_STALL_S, write=WRITE_TIMEOUT_S, pool=POOL_TIMEOUT_S
    )


# The gateway reads the GitHub account, plus the addresses of an account whose address is private.
# TODO: `auth.config` states no GitHub deadline, so this count is a copy of the gateway's; it goes with GitHub
# sign-in.
GITHUB_REGISTER_TIMEOUT = timeout_after_deadline(2 * MAX_REQUEST_STALL_S)


def workos_register_timeout(settings: WorkOSSettings) -> httpx.Timeout:
    """The bound on `users.register` for a WorkOS token, from the deadline the platform states."""
    return timeout_after_deadline(settings.register_deadline_s)


def longest_register_timeout(config: AuthConfig) -> httpx.Timeout:
    """The bound on `users.register` for a credential of any sign-in the platform takes.

    A platform that names no WorkOS client takes the GitHub check.
    """
    if config.workos is None:
        return GITHUB_REGISTER_TIMEOUT
    workos = workos_register_timeout(config.workos)
    if not config.github_registration:
        return workos
    return max(GITHUB_REGISTER_TIMEOUT, workos, key=max_stall_s)
