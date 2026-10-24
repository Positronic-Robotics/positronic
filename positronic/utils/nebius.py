"""Values held in a Nebius project, read through the `nebius` CLI rather than an API client."""

import json
import os
import subprocess
from typing import Any

from platform_client.policy_container import AUTH_TOKEN_ENV

# The project holding the secrets, and the secret `serve.sh` injects into every endpoint it creates.
# `workflows/nebius/common.sh` carries the same two values under the same overrides for the shell scripts;
# no constant can span both languages. ``or`` rather than a ``get`` default, because those overrides are
# read there as ``${VAR:-default}``, which falls back on a variable set to nothing as well as an unset one.
PARENT_ID = os.environ.get('NEBIUS_PARENT_ID') or 'project-e00f38wexevrr52b8j'
AUTH_TOKEN_SECRET = os.environ.get('NEBIUS_AUTH_TOKEN_SECRET') or 'positronic-serverless-inference-token'

# Nebius injects a secret as `--env-secret <KEY>=<secret>` and takes the payload key from that same KEY,
# so the payload key is not free: it is whichever variable the server reads the token from.
AUTH_TOKEN_KEY = AUTH_TOKEN_ENV

_CLI_TIMEOUT_SEC = 30.0


class NebiusError(RuntimeError):
    """A `nebius` call that failed."""


class NebiusNotInstalled(NebiusError):
    """The `nebius` CLI is not on PATH."""


class NebiusNotFound(NebiusError):
    """The resource that a `nebius` call names does not exist."""


# The gRPC status that the CLI prints for a resource that does not exist.
_NOT_FOUND_STATUS = 'code = NotFound'


def call(*args: str, profile: str | None = None, structured: bool = True, timeout: float = _CLI_TIMEOUT_SEC) -> Any:
    """One `nebius` call: its parsed JSON, or its raw output when it is not `structured`."""
    command = ['nebius', *args, *(['--profile', profile] if profile else [])]
    command += ['--format', 'json'] if structured else []
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError:
        raise NebiusNotInstalled('The `nebius` CLI is not on PATH: https://docs.nebius.com/cli/install') from None
    except subprocess.TimeoutExpired:
        # An expired CLI credential turns any call into a browser login that waits for a human, which would
        # hang a run rather than fail it.
        raise NebiusError(
            f'`{" ".join(command)}` did not answer in {timeout:.0f}s. Run a `nebius` command '
            f'yourself to complete the login, then retry.'
        ) from None
    except subprocess.CalledProcessError as e:
        error = NebiusNotFound if _NOT_FOUND_STATUS in e.stderr else NebiusError
        raise error(f'`{" ".join(command)}` failed: {e.stderr.strip()}') from None
    if not structured:
        return completed.stdout
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise NebiusError(f'`{" ".join(command)}` printed no JSON: {completed.stdout[:200]!r}') from None


def auth_token(secret: str = AUTH_TOKEN_SECRET, parent_id: str = PARENT_ID) -> str:
    """The bearer token gating served endpoints, read from the MysteryBox secret holding it.

    Two calls because the payload is addressable only by secret id, which the name has to be resolved to.
    """
    try:
        found = call('mysterybox', 'secret', 'get-by-name', '--parent-id', parent_id, '--name', secret)
    except NebiusNotInstalled as e:
        raise NebiusNotInstalled(f'{e} Install it, or export {AUTH_TOKEN_ENV} and use `.authed_remote`.') from None
    payload = call(
        'mysterybox', 'payload', 'get-by-key', '--secret-id', found['metadata']['id'], '--key', AUTH_TOKEN_KEY
    )
    token = payload['data']['string_value']
    if not token:
        raise RuntimeError(f'Secret {secret} carries no {AUTH_TOKEN_KEY} value')
    return token
