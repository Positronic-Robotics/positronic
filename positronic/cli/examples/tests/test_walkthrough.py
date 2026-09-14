"""`walkthrough.py`: the board reads it makes with no key, whatever the environment holds."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import routes
from platform_client.client import API_KEY_ENV, AUTH_HEADER

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
        return httpx.Response(200, json=BOARDS)


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
