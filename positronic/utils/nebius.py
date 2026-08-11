"""Values held in a Nebius project, read through the `nebius` CLI rather than an API client.

The CLI is what the workflow scripts under `workflows/nebius/` already require and authenticate, so reading
a value the same way needs no second set of credentials.
"""

import json
import os
import subprocess

from positronic.offboard.server import AUTH_TOKEN_ENV

# The project holding the secrets, and the secret `serve.sh` injects into every endpoint it creates.
# `workflows/nebius/common.sh` carries the same two values under the same overrides for the shell scripts;
# no constant can span both languages.
PARENT_ID = os.environ.get('NEBIUS_PARENT_ID', 'project-e00f38wexevrr52b8j')
AUTH_TOKEN_SECRET = os.environ.get('NEBIUS_AUTH_TOKEN_SECRET', 'positronic-serverless-inference-token')

# Nebius injects a secret as `--env-secret <KEY>=<secret>` and takes the payload key from that same KEY,
# so the payload key is not free: it is whichever variable the server reads the token from.
AUTH_TOKEN_KEY = AUTH_TOKEN_ENV

_CLI_TIMEOUT_SEC = 30.0


def _nebius(*args: str) -> dict:
    """The parsed JSON of one `nebius` call."""
    command = ['nebius', *args, '--format', 'json']
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=_CLI_TIMEOUT_SEC, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            'The `nebius` CLI is not on PATH. Install it, or export AUTH_TOKEN and use `.authed_remote`.'
        ) from None
    except subprocess.TimeoutExpired:
        # An expired CLI credential turns any call into a browser login that waits for a human, which would
        # hang a run rather than fail it.
        raise RuntimeError(
            f'`{" ".join(command)}` did not answer in {_CLI_TIMEOUT_SEC:.0f}s. Run a `nebius` command '
            f'yourself to complete the login, then retry.'
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'`{" ".join(command)}` failed: {e.stderr.strip()}') from None
    return json.loads(completed.stdout)


def auth_token(secret: str = AUTH_TOKEN_SECRET, parent_id: str = PARENT_ID) -> str:
    """The bearer token gating served endpoints, read from the MysteryBox secret holding it.

    Two calls because the payload is addressable only by secret id, which the name has to be resolved to.
    """
    found = _nebius('mysterybox', 'secret', 'get-by-name', '--parent-id', parent_id, '--name', secret)
    payload = _nebius(
        'mysterybox', 'payload', 'get-by-key', '--secret-id', found['metadata']['id'], '--key', AUTH_TOKEN_KEY
    )
    token = payload['data']['string_value']
    if not token:
        raise RuntimeError(f'Secret {secret} carries no {AUTH_TOKEN_KEY} value')
    return token
