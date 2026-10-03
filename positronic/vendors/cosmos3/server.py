"""NVIDIA's Cosmos3 action server, behind the run's bearer token.

It runs in NVIDIA's own environment, so it imports nothing from positronic. It wraps the `process_request`
hook of openpi's server, so the token is checked on every request before openpi answers `/healthz` or
opens a session.

Usage
  python server.py [NVIDIA's server arguments]
"""

from __future__ import annotations

import hmac
import http
import os
from collections.abc import Callable, Mapping
from typing import Any

# rules-allow: hardcoded-keys — positronic is not installed where this file runs; tests/test_server.py pins
# each name to positronic's own.
AUTH_TOKEN_ENV = 'AUTH_TOKEN'
AUTH_HEADER = 'Authorization'

ProcessRequest = Callable[[Any, Any], Any]


def run_token(environ: Mapping[str, str]) -> str | None:
    """The token the server checks, or None to serve open.

    Refuses a set token that an `Authorization` header cannot carry, as positronic's server does.
    """
    token = environ.get(AUTH_TOKEN_ENV)
    if token is None:
        return None
    if not (token and all('!' <= c <= '~' for c in token)):
        raise ValueError(f'{AUTH_TOKEN_ENV} must be non-empty printable ASCII without spaces; unset it to serve open')
    return token


def carries_token(headers: Mapping[str, str], token: str | None) -> bool:
    if token is None:
        return True
    presented = headers.get(AUTH_HEADER) or ''
    return hmac.compare_digest(presented.encode(), f'Bearer {token}'.encode())


def gated(process_request: ProcessRequest, token: str | None) -> ProcessRequest:
    """`process_request`, behind a 401 for a request that does not carry `token`."""

    def check(connection, request):
        if not carries_token(request.headers, token):
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, 'Invalid or missing bearer token\n')
        return process_request(connection, request)

    return check


def main() -> None:
    # Read before the model loads, so a bad token fails in seconds, not minutes.
    token = run_token(os.environ)

    # Only the image's environment has openpi's server and NVIDIA's package.
    from openpi_server import websocket_policy_server  # noqa: PLC0415

    # `WebsocketPolicyServer.run` reads this module global when it binds. An openpi without it raises here, so
    # the server never serves without the check.
    websocket_policy_server._health_check = gated(websocket_policy_server._health_check, token)

    from cosmos_framework.scripts import action_policy_server_robolab  # noqa: PLC0415

    action_policy_server_robolab.main()


if __name__ == '__main__':
    main()
