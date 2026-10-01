"""The FLUX 3 Action server's transport and token gate, run without the model it serves."""

import asyncio
import urllib.error
import urllib.request

import pytest
import websockets.asyncio.client
from websockets.exceptions import InvalidStatus

from positronic.offboard import protocol
from positronic.offboard.roboarena import RoboarenaClient
from positronic.utils.serialization import serialize
from positronic.vendors.dreamzero import roboarena, roboarena_policy
from positronic.vendors.flux3_action import server

TOKEN = 'run-token'
GREETING = b'metadata'


def _get(url: str, headers: dict[str, str]) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as answer:
            return answer.status
    except urllib.error.HTTPError as refused:
        return refused.code


async def _greet(connection) -> None:
    await connection.send(GREETING)


async def _first_frame_or_refusal(port: int, headers: dict[str, str]) -> bytes | str | int:
    try:
        async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}', additional_headers=headers) as client:
            return await client.recv()
    except InvalidStatus as refused:
        return refused.response.status_code


def _exchange(token: str | None, headers: dict[str, str]) -> tuple[int, bytes | str | int]:
    """The health route's status, and a session's first frame or its refusal status, both sent with `headers`."""

    async def exchange() -> tuple[int, bytes | str | int]:
        async with server.server(_greet, '127.0.0.1', 0, token) as bound:
            port = bound.sockets[0].getsockname()[1]
            url = f'http://127.0.0.1:{port}{server.HEALTH_PATH}'
            return await asyncio.to_thread(_get, url, headers), await _first_frame_or_refusal(port, headers)

    return asyncio.run(exchange())


def test_without_a_token_the_server_answers_the_health_route_and_hands_a_session_to_the_handler():
    assert _exchange(None, {}) == (200, GREETING)


def test_the_runs_token_opens_the_health_route_and_a_session():
    assert _exchange(TOKEN, {server.AUTH_HEADER: f'Bearer {TOKEN}'}) == (200, GREETING)


@pytest.mark.parametrize('headers', [{}, {server.AUTH_HEADER: 'Bearer other'}], ids=['none', 'wrong'])
def test_a_caller_without_the_runs_token_is_refused_on_both_routes(headers):
    assert _exchange(TOKEN, headers) == (401, 401)


@pytest.mark.parametrize('token', ['', 'two words', 'line\n'])
def test_a_token_no_header_can_carry_is_refused(token):
    with pytest.raises(ValueError, match=server.AUTH_TOKEN_ENV):
        server.run_token({server.AUTH_TOKEN_ENV: token})


def test_an_unset_token_serves_open():
    assert server.run_token({}) is None


def test_a_bad_token_fails_before_the_model_loads(monkeypatch):
    """The positronic environment has neither torch nor `flux_action`, so a check after the import fails on it."""
    monkeypatch.setenv(server.AUTH_TOKEN_ENV, 'two words')
    with pytest.raises(ValueError, match=server.AUTH_TOKEN_ENV):
        server.main(['--checkpoint', 'unused'])


def test_the_gate_reads_the_variable_and_header_of_positronics_servers():
    assert (server.AUTH_TOKEN_ENV, server.AUTH_HEADER) == (protocol.AUTH_TOKEN_ENV, protocol.AUTH_HEADER)


def test_the_announced_config_names_the_fields_positronics_roboarena_client_reads():
    assert set(server.SERVER_CONFIG) == {
        roboarena.RESOLUTION,
        roboarena.NEEDS_WRIST_CAMERA,
        roboarena.NUM_EXTERIOR_CAMERAS,
        roboarena.NEEDS_STEREO_CAMERA,
        roboarena.NEEDS_SESSION_ID,
        roboarena.ACTION_SPACE,
    }
    assert server.SERVER_CONFIG[roboarena.ACTION_SPACE] == roboarena_policy.JOINT_POSITION_SPACE


def test_positronics_roboarena_client_reads_the_config_through_the_gate():
    async def announce(connection) -> None:
        await connection.send(serialize(server.SERVER_CONFIG))

    async def connect() -> dict:
        async with server.server(announce, '127.0.0.1', 0, TOKEN) as bound:
            port = bound.sockets[0].getsockname()[1]
            client = RoboarenaClient('127.0.0.1', port, headers={server.AUTH_HEADER: f'Bearer {TOKEN}'})
            try:
                return await asyncio.to_thread(client.connect)
            finally:
                await asyncio.to_thread(client.close)

    assert asyncio.run(connect()) == server.SERVER_CONFIG
