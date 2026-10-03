"""The token gate the Cosmos3 image serves behind, run without NVIDIA's model or openpi."""

import asyncio
import http
import sys
import types
import urllib.error
import urllib.request

import pytest
import websockets.asyncio.client
import websockets.asyncio.server
from websockets.exceptions import InvalidStatus

from positronic.offboard import protocol
from positronic.vendors.cosmos3 import server

TOKEN = 'run-token'
HEALTH_PATH = '/healthz'
ANNOUNCEMENT = b'config'


def openpi_health_check(connection, request):
    """The hook openpi's server passes to `websockets` as its `process_request`."""
    if request.path == HEALTH_PATH:
        return connection.respond(http.HTTPStatus.OK, 'OK\n')
    return None


async def _announce(connection) -> None:
    await connection.send(ANNOUNCEMENT)


def _get(url: str, headers: dict[str, str]) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as answer:
            return answer.status
    except urllib.error.HTTPError as refused:
        return refused.code


async def _first_frame_or_refusal(port: int, headers: dict[str, str]) -> bytes | str | int:
    try:
        async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}', additional_headers=headers) as client:
            return await client.recv()
    except InvalidStatus as refused:
        return refused.response.status_code


def _exchange(headers: dict[str, str]) -> tuple[int, bytes | str | int]:
    """The health route's status, and a session's first frame or its refusal status, both sent with `headers`."""

    async def exchange() -> tuple[int, bytes | str | int]:
        process_request = server.gated(openpi_health_check, TOKEN)
        async with websockets.asyncio.server.serve(_announce, '127.0.0.1', 0, process_request=process_request) as bound:
            port = bound.sockets[0].getsockname()[1]
            url = f'http://127.0.0.1:{port}{HEALTH_PATH}'
            return await asyncio.to_thread(_get, url, headers), await _first_frame_or_refusal(port, headers)

    return asyncio.run(exchange())


def test_the_runs_token_opens_the_health_route_and_a_session():
    assert _exchange({server.AUTH_HEADER: f'Bearer {TOKEN}'}) == (200, ANNOUNCEMENT)


@pytest.mark.parametrize('headers', [{}, {server.AUTH_HEADER: 'Bearer other'}], ids=['none', 'wrong'])
def test_a_caller_without_the_runs_token_is_refused_on_both_routes(headers):
    assert _exchange(headers) == (401, 401)


@pytest.mark.parametrize('token', ['', 'two words', 'line\n'])
def test_a_token_no_header_can_carry_is_refused(token):
    with pytest.raises(ValueError, match=server.AUTH_TOKEN_ENV):
        server.run_token({server.AUTH_TOKEN_ENV: token})


def test_an_unset_token_serves_open():
    assert server.run_token({}) is None
    assert server.carries_token({}, None)


def test_the_gate_reads_the_variable_and_header_of_positronics_servers():
    assert (server.AUTH_TOKEN_ENV, server.AUTH_HEADER) == (protocol.AUTH_TOKEN_ENV, protocol.AUTH_HEADER)


def _stand_in(monkeypatch, name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    vars(module).update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition('.')
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


def _stand_ins(monkeypatch, served: list[str], **openpi_attributes) -> types.ModuleType:
    _stand_in(monkeypatch, 'openpi_server')
    openpi = _stand_in(monkeypatch, 'openpi_server.websocket_policy_server', **openpi_attributes)
    _stand_in(monkeypatch, 'cosmos_framework')
    _stand_in(monkeypatch, 'cosmos_framework.scripts')

    def nvidia_main() -> None:
        served.append('NVIDIA')

    _stand_in(monkeypatch, 'cosmos_framework.scripts.action_policy_server_robolab', main=nvidia_main)
    return openpi


def test_main_gates_openpis_hook_and_then_runs_nvidias_server(monkeypatch):
    served: list[str] = []
    openpi = _stand_ins(monkeypatch, served, _health_check=openpi_health_check)
    monkeypatch.setenv(server.AUTH_TOKEN_ENV, TOKEN)

    server.main()

    assert served == ['NVIDIA']
    assert openpi._health_check is not openpi_health_check


def test_main_does_not_serve_when_openpi_has_no_hook(monkeypatch):
    served: list[str] = []
    _stand_ins(monkeypatch, served)
    monkeypatch.setenv(server.AUTH_TOKEN_ENV, TOKEN)

    with pytest.raises(AttributeError):
        server.main()
    assert served == []


def test_a_bad_token_fails_before_anything_is_imported(monkeypatch):
    """The positronic environment has neither openpi nor NVIDIA's package, so a check after the import fails on it."""
    monkeypatch.setenv(server.AUTH_TOKEN_ENV, 'two words')
    with pytest.raises(ValueError, match=server.AUTH_TOKEN_ENV):
        server.main()
