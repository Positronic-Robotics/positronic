import os

import configuronic as cfn

from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, bearer
from positronic.policy import RemotePolicy
from positronic.utils import nebius


@cfn.config()
def unset():
    """No policy. It lives in this package so a relative `--policy=.act` still resolves against it,
    and instantiates to None rather than raising, so the absence is a value a caller can act on."""
    return None


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


remote = cfn.Config(RemotePolicy, url='ws://localhost:8000')


@cfn.config()
def bearer_headers():
    token = os.environ.get(AUTH_TOKEN_ENV)
    if not token:
        raise ValueError(f'{AUTH_TOKEN_ENV} is not set; export the endpoint token before running inference')
    return {AUTH_HEADER: bearer(token)}


@cfn.config()
def nebius_bearer_headers():
    return {AUTH_HEADER: bearer(nebius.auth_token())}


# Neither carries a default URL: it names one specific endpoint, and a token must not be handed to
# whichever host a stale default points at.
authed_remote = cfn.Config(RemotePolicy, headers=bearer_headers)
nebius_remote = cfn.Config(RemotePolicy, headers=nebius_bearer_headers)
