"""`standings.py` over a stub platform: the routes it reads, the key it does not send, and what it prints."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import routes
from platform_client.boards import BoardRef
from platform_client.client import AUTH_HEADER, PlatformClient

SCRIPT = Path(__file__).resolve().parents[1] / 'standings.py'
BASE = 'http://gateway.test'
SUBMISSION = '5f3a91c2b7d40e18'
AT = '2026-03-04T05:06:07Z'

BOARDS = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'boards': [
        {
            'board': 'a.board',
            'title': 'Board A',
            'eval': 'a.eval',
            'primary_metric': 'success_rate',
            'visibility': 'public',
        },
        {'board': 'b.board', 'title': 'Board B', 'eval': 'b.eval', 'primary_metric': 'reward', 'visibility': 'public'},
    ]
}
STANDINGS = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'board': 'a.board',
    'eval': 'a.eval',
    'primary_metric': 'success_rate',
    'rankings': [
        {
            'rank': 1,
            'display_name': 'ateam',
            'tag': '0ddba7',
            'scores': {'primary': 0.75},
            'submission_id': SUBMISSION,
            'submitted_at': AT,
        },
        {
            'rank': 2,
            'display_name': 'ateam',
            'tag': 'e2fb9b',
            'scores': {},
            'submission_id': '3903c0d9e7d4b5a8',
            'submitted_at': AT,
        },
    ],
}
NOT_FOUND = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'error': {'code': 'not_found', 'message': 'board not found', 'details': {}}
}


@pytest.fixture(scope='module')
def standings() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


class Platform:
    """Answers each route with its canned payload and records every request."""

    def __init__(self, answers: dict[str, tuple[object, int]]) -> None:
        self.answers = answers
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        payload, status = self.answers[request.url.path]
        return httpx.Response(status, json=payload)


def make_client(platform: Platform) -> PlatformClient:
    return PlatformClient(client=httpx.Client(base_url=BASE, transport=httpx.MockTransport(platform)), api_key=None)


def test_with_no_board_it_lists_every_board_and_the_eval_each_ranks(standings: dict[str, Any]):
    platform = Platform({routes.RANKINGS_LIST: (BOARDS, 200)})

    lines = standings['run'](make_client(platform), None)

    assert lines == [
        'board    eval    primary metric  title',
        'a.board  a.eval  success_rate    Board A',
        'b.board  b.eval  reward          Board B',
    ]
    assert [request.url.path for request in platform.requests] == [routes.RANKINGS_LIST]
    assert AUTH_HEADER not in platform.requests[0].headers


def test_with_a_board_it_prints_rank_name_tag_score_and_submission(standings: dict[str, Any]):
    platform = Platform({routes.RANKINGS_GET: (STANDINGS, 200)})

    lines = standings['run'](make_client(platform), BoardRef('a.board'))

    assert lines == [
        'a.board: ranks a.eval by success_rate',
        'rank  name#tag      success_rate  submission',
        '1     ateam#0ddba7  0.750         5f3a91c2b7d40e18',
        '2     ateam#e2fb9b  -             3903c0d9e7d4b5a8',
    ]
    query = platform.requests[0].url.params
    assert (
        query['board'] == 'a.board'
    )  # rules-allow: hardcoded-keys — the wire spelling of the query field is what this test pins


def test_an_empty_board_says_so(standings: dict[str, Any]):
    empty = {**STANDINGS, 'rankings': []}  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    platform = Platform({routes.RANKINGS_GET: (empty, 200)})
    assert standings['run'](make_client(platform), BoardRef('a.board')) == [
        'a.board: ranks a.eval by success_rate',
        'no entries',
    ]


def test_an_unknown_board_names_the_boards_on_offer(standings: dict[str, Any]):
    platform = Platform({routes.RANKINGS_GET: (NOT_FOUND, 404), routes.RANKINGS_LIST: (BOARDS, 200)})

    with pytest.raises(SystemExit) as exit_info:
        standings['run'](make_client(platform), BoardRef('nope'))

    assert str(exit_info.value) == 'board not found: nope\nboards on offer: a.board, b.board'


def test_an_empty_board_slug_is_refused_before_any_request(standings: dict[str, Any]):
    with pytest.raises(SystemExit) as exit_info:
        standings['main'](['--board='])
    assert str(exit_info.value) == "not a board slug: ''"
