"""Serve FLUX 3 Action's DROID policy over the roboarena wire, behind the run's bearer token.

`docker/Dockerfile.flux3-action` runs this file in Black Forest Labs' own environment, which has
`flux_action`, torch and websockets and no positronic, so it imports nothing from positronic. It loads the
policy as `flux-action serve-robolab` does and answers each session with BFL's handler. It adds the whole
roboarena server config to the handshake, and a `/healthz` route that answers once the model is warm.

Usage
  python server.py --checkpoint black-forest-labs/flux-3-action-droid --revision <commit> \
      --subfolder variants/gd [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hmac
import http
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import websockets.asyncio.server
import websockets.http11

# rules-allow: hardcoded-keys — positronic is not installed where this file runs; tests/test_server.py pins
# each name to positronic's own.
AUTH_TOKEN_ENV = 'AUTH_TOKEN'
AUTH_HEADER = 'Authorization'
HEALTH_PATH = '/healthz'

# What a roboarena client reads on connect. The client encodes each view at this (height, width), and BFL
# composes the wrist view above the two exteriors at half size into the 540x640 frame it pads to its canvas.
# rules-allow: hardcoded-keys — the roboarena wire's own field names; tests/test_server.py pins each to
# positronic's client.
SERVER_CONFIG: dict[str, Any] = {
    'image_resolution': [360, 640],
    'needs_wrist_camera': True,
    'n_external_cameras': 2,
    'needs_stereo_camera': False,
    'needs_session_id': False,
    # Seven absolute joint positions in radians and a gripper closed fraction.
    'action_space': 'joint_position',
}

Handle = Callable[[websockets.asyncio.server.ServerConnection], Awaitable[None]]
ProcessRequest = Callable[
    [websockets.asyncio.server.ServerConnection, websockets.http11.Request], websockets.http11.Response | None
]


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


def answer_health(
    connection: websockets.asyncio.server.ServerConnection, request: websockets.http11.Request
) -> websockets.http11.Response | None:
    """A 200 on `HEALTH_PATH`. Every other request goes on to the WebSocket handshake."""
    if request.path == HEALTH_PATH:
        return connection.respond(http.HTTPStatus.OK, 'OK\n')
    return None


def server(handle: Handle, host: str, port: int, token: str | None):
    """The server behind `token`, with the transport settings of BFL's `serve_async`.

    It binds only when entered. `main` enters it after the policy is loaded and warm.
    """
    return websockets.asyncio.server.serve(
        handle, host, port, compression=None, max_size=None, process_request=gated(answer_health, token)
    )


async def serve_forever(handle: Handle, host: str, port: int, token: str | None) -> None:
    async with server(handle, host, port, token) as bound:
        await bound.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve FLUX 3 Action's DROID policy over the roboarena wire.")
    parser.add_argument('--checkpoint', required=True, help='a Hugging Face repository id or a local directory')
    parser.add_argument('--revision', help='the repository commit')
    parser.add_argument('--subfolder', help='the policy package in the repository, e.g. variants/gd')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    # Read before the model loads, so a bad token fails in seconds, not minutes.
    token = run_token(os.environ)

    # Only the image's environment has BFL's package and torch.
    import torch  # noqa: PLC0415
    from flux_action.serving import robolab  # noqa: PLC0415

    # FOOTGUN: without this capture the warm-up dies on "Inplace update to inference tensor outside
    # InferenceMode" as it captures the DiT's CUDA graph, in `serve-robolab` too. The graph lives for the run.
    first_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first_graph):
        torch.zeros(1, device='cuda')

    # The defaults of `serve-robolab`: DiT in bfloat16, compiled, one warm-up request before the bind.
    policy = robolab.load_serving_policy(
        args.checkpoint, revision=args.revision, subfolder=args.subfolder, device='cuda', dtype='bfloat16'
    )
    adapter = robolab.RoboLabPolicy(policy)
    metadata = {**SERVER_CONFIG, 'serving_setup': policy.serving_setup}
    print(
        json.dumps({'serving': args.checkpoint, 'revision': args.revision, 'subfolder': args.subfolder, **metadata}),
        flush=True,
    )
    handle = functools.partial(robolab._handle, adapter=adapter, metadata=metadata)
    asyncio.run(serve_forever(handle, args.host, args.port, token))


if __name__ == '__main__':
    main()
