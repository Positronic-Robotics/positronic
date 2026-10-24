"""`walkthrough.py`: the board reads it makes with no key, whatever the environment holds."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import routes
from platform_client.client import API_KEY_ENV, API_URL_ENV, AUTH_HEADER, PlatformClient
from platform_client.config import CONFIG_DIR_ENV, Config, write_config
from platform_client.ids import ApiKey

from positronic.cli.conftest import SERVES_THIS_CLIENT

SCRIPT = Path(__file__).resolve().parents[1] / 'walkthrough.py'
BASE = 'http://gateway.test'
BOARDS = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'boards': [
        {
            'board': 'a.board',
            'title': 'Board A',
            'eval': 'a.eval',
            'primary_metric': 'success_rate',
            'visibility': 'public',
        }
    ]
}


@pytest.fixture(autouse=True)
def no_saved_record(monkeypatch, tmp_path):
    """A config directory with no record and no key in the environment, so the box's own never reach a test."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_URL_ENV, raising=False)


@pytest.fixture(scope='module')
def walkthrough() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


class Platform:
    """Answers `rankings.list` and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.path == routes.RANKINGS_LIST
        return httpx.Response(200, json=BOARDS, headers=SERVES_THIS_CLIENT)


def test_the_discovery_read_sends_no_key_even_when_the_environment_holds_one(walkthrough: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    platform = Platform()
    transport = httpx.MockTransport(platform)
    with walkthrough['anonymous_client'](client=httpx.Client(base_url=BASE, transport=transport)) as public:
        public.list_boards()

    assert AUTH_HEADER not in platform.requests[0].headers


def test_an_empty_platform_url_is_a_cli_error_before_any_request(walkthrough: dict[str, Any], capsys):
    with pytest.raises(SystemExit) as exit_info:
        walkthrough['main'](['--platform-url='])
    assert exit_info.value.code == 2
    assert 'base_url is empty' in capsys.readouterr().err


@pytest.mark.parametrize('key_status', ['created', 'rotated'])
def test_authentication_spends_the_token_on_one_registration(walkthrough: dict[str, Any], key_status: str):
    """A WorkOS token registers once, so a second call with the same token would be refused."""
    sent: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == routes.AUTH_CONFIG:
            return httpx.Response(200, json={'github_registration': True, 'workos': None}, headers=SERVES_THIS_CLIENT)
        sent.append(json.loads(request.content))
        body = {'user_id': 'a0', 'artifact_location': 's3://b/a0/', 'api_key': 'pk_new', 'key_status': key_status}
        return httpx.Response(200, json=body, headers=SERVES_THIS_CLIENT)

    client = PlatformClient(client=httpx.Client(base_url=BASE, transport=httpx.MockTransport(answer)))
    walkthrough['authenticate'](client, credential='token', alias='demo')

    assert sent == [{'credential': 'token', 'alias': 'demo', 'rotate': True}]
    assert client.api_key == 'pk_new'


def test_a_run_takes_the_key_and_platform_that_register_saved(walkthrough: dict[str, Any], monkeypatch, tmp_path):
    write_config(tmp_path, Config(platform_url=BASE, api_key=ApiKey('pk_live_saved')))
    clients: list[PlatformClient] = []
    monkeypatch.setitem(walkthrough['main'].__globals__, 'walkthrough', lambda client, **_: clients.append(client))

    walkthrough['main'](['--eval=a.eval', '--policy-image=org/policy@sha256:abc'])

    assert [(client.api_key, client.base_url) for client in clients] == [('pk_live_saved', BASE)]


def test_the_discovery_run_lists_the_boards_of_the_platform_that_register_saved(
    walkthrough: dict[str, Any], monkeypatch, tmp_path, capsys
):
    write_config(tmp_path, Config(platform_url=BASE, api_key=ApiKey('pk_live_saved')))
    platform = Platform()
    platforms: list[str | None] = []

    def anonymous_client(platform_url: str | None) -> PlatformClient:
        platforms.append(platform_url)
        return walkthrough['anonymous_client'](
            client=httpx.Client(base_url=BASE, transport=httpx.MockTransport(platform))
        )

    monkeypatch.setitem(walkthrough['main'].__globals__, 'anonymous_client', anonymous_client)

    walkthrough['main']([])

    assert platforms == [BASE] and AUTH_HEADER not in platform.requests[0].headers
    assert 'a.board' in capsys.readouterr().out
