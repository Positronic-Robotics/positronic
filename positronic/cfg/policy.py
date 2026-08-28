import json
import os
from pathlib import Path

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


@cfn.config()
def file_headers(path: str) -> dict[str, str]:
    """The header set at ``path``: a JSON object of header name to value.

    A path, so no credential reaches this process's command line.
    """
    file = Path(path).expanduser()
    try:
        parsed = json.loads(file.read_text())
    except Exception as exc:
        problem = f'could not be read as a header set ({type(exc).__name__})'
    else:
        if isinstance(parsed, dict) and parsed and all(isinstance(s, str) for kv in parsed.items() for s in kv):
            return parsed
        problem = 'must hold a JSON object of at least one header, every name and value a string'
        del parsed  # A frame's locals reach a traceback that renders them, and this frame raises below.
    # `from None`: a failed read or parse keeps what it choked on, a `UnicodeDecodeError` its whole
    # byte string, and a chained cause carries that into the traceback.
    raise ValueError(f'{file}: {problem}') from None


# The caller names the URL: a default would hand the credential to whatever host it points at.
authed_remote = cfn.Config(RemotePolicy, headers=bearer_headers)
nebius_remote = cfn.Config(RemotePolicy, headers=nebius_bearer_headers)
file_authed_remote = cfn.Config(RemotePolicy, headers=file_headers)
