"""`standings.py` over a stub platform: the routes it reads, the key it does not send, and what it prints."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import routes
from platform_client.client import API_KEY_ENV, API_URL_ENV, AUTH_HEADER, PlatformClient
from platform_client.config import CONFIG_DIR_ENV, Config, write_config
from platform_client.ids import ApiKey
from platform_client.leaderboards import LeaderboardRef

from positronic.cli.conftest import SERVES_THIS_CLIENT
from positronic.cli.examples.nebius_competition.leaderboards import LEADERBOARD_TAG

SCRIPT = Path(__file__).resolve().parents[1] / 'standings.py'
BASE = 'http://gateway.test'
SUBMISSION = '5f3a91c2b7d40e18'
AT = '2026-03-04T05:06:07Z'

DESCRIPTOR = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'eval': 'a.eval',
    'metrics': [{'key': 'primary', 'name': 'success_rate', 'decimals': 3}],
    'ranked_by': 'primary',
}
LEADERBOARDS = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'tag': LEADERBOARD_TAG,
    'leaderboards': [
        {
            'leaderboard': 'a.board',
            'title': 'Board A',
            'method': 'best_submission',
            'tags': [{'tag': LEADERBOARD_TAG, 'position': 0}],
            'descriptor': DESCRIPTOR,
        },
        {
            'leaderboard': 'b.rig',
            'title': 'Rig B',
            'method': 'a_method_this_client_has_no_name_for',
            'tags': [{'tag': LEADERBOARD_TAG, 'position': 1}],
            'descriptor': {
                'robot': 'A robot this client has no name for',
                'metrics': [{'key': 'reward', 'name': 'reward', 'decimals': 1}],
                'ranked_by': 'reward',
            },
        },
    ],
}


def row(key: str, rank: int | None, name: str, tag: str, score: float | None, submission: str) -> dict[str, Any]:
    """A team row as `leaderboards.get` carries it. A row with no rank is a baseline."""
    return {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
        'key': key,
        'display_name': name,
        'user_tag': tag,
        'rank': None if rank is None else {'best': rank, 'worst': rank},
        'baseline': {'publisher': 'Example Labs'} if rank is None else None,
        'metrics': {} if score is None else {'primary': {'value': score, 'low': score, 'high': score}},
        'submission': {'id': submission, 'submitted_at': AT},
    }


STANDINGS = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'leaderboard': 'a.board',
    'title': 'Board A',
    'method': 'best_submission',
    'descriptor': DESCRIPTOR,
    'computed_at': AT,
    'rows': [
        row('team-1', 1, 'ateam', '0ddba7', 0.75, SUBMISSION),
        row('team-2', None, 'reference', '9c41de', 0.5, '7b2e04c1d9f3a650'),
        row('team-3', 2, 'ateam', 'e2fb9b', None, '3903c0d9e7d4b5a8'),
    ],
}
NOT_FOUND = {  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    'error': {'code': 'not_found', 'message': 'leaderboard not found', 'details': {}}
}


@pytest.fixture(autouse=True)
def no_saved_record(monkeypatch, tmp_path):
    """A config directory with no record and no platform in the environment, so the box's own never reach a test."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_URL_ENV, raising=False)


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
        return httpx.Response(status, json=payload, headers=SERVES_THIS_CLIENT)


def make_client(platform: Platform) -> PlatformClient:
    return PlatformClient(client=httpx.Client(base_url=BASE, transport=httpx.MockTransport(platform)), api_key=None)


def test_with_no_board_it_lists_every_tagged_leaderboard_and_the_eval_each_ranks(standings: dict[str, Any]):
    platform = Platform({routes.LEADERBOARDS_LIST: (LEADERBOARDS, 200)})

    lines = standings['run'](make_client(platform), None)

    assert lines == [
        'board    eval    ranked by     title',
        'a.board  a.eval  success_rate  Board A',
        'b.rig    -       reward        Rig B',
    ]
    assert [request.url.path for request in platform.requests] == [routes.LEADERBOARDS_LIST]
    assert dict(platform.requests[0].url.params) == {'tag': LEADERBOARD_TAG}
    assert AUTH_HEADER not in platform.requests[0].headers


def test_with_a_board_it_prints_rank_name_tag_score_and_submission_and_marks_a_baseline(standings: dict[str, Any]):
    platform = Platform({routes.LEADERBOARDS_GET: (STANDINGS, 200)})

    lines = standings['run'](make_client(platform), LeaderboardRef('a.board'))

    assert lines == [
        'a.board: ranks a.eval by success_rate',
        'rank      name#tag          success_rate  submission',
        '1         ateam#0ddba7      0.750         5f3a91c2b7d40e18',
        'baseline  reference#9c41de  0.500         7b2e04c1d9f3a650',
        '2         ateam#e2fb9b      -             3903c0d9e7d4b5a8',
    ]
    query = platform.requests[0].url.params
    assert (
        query['leaderboard'] == 'a.board'
    )  # rules-allow: hardcoded-keys — the wire spelling of the query field is what this test pins


def test_rows_whose_places_are_a_range_print_the_range(standings: dict[str, Any]):
    tied = {'best': 1, 'worst': 2}  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    rows = [
        {**row('team-1', 1, 'ateam', '0ddba7', 0.75, SUBMISSION), 'rank': tied},
        {**row('team-3', 2, 'bteam', 'e2fb9b', 0.75, '3903c0d9e7d4b5a8'), 'rank': tied},
    ]
    platform = Platform({routes.LEADERBOARDS_GET: ({**STANDINGS, 'rows': rows}, 200)})

    lines = standings['run'](make_client(platform), LeaderboardRef('a.board'))

    assert [line.split()[0] for line in lines[2:]] == ['1-2', '1-2']


def test_an_empty_board_says_so(standings: dict[str, Any]):
    empty = {**STANDINGS, 'rows': []}  # rules-allow: hardcoded-keys — a wire fixture pins the server's spelling
    platform = Platform({routes.LEADERBOARDS_GET: (empty, 200)})
    assert standings['run'](make_client(platform), LeaderboardRef('a.board')) == [
        'a.board: ranks a.eval by success_rate',
        'no entries',
    ]


def test_a_leaderboard_that_ranks_no_eval_names_only_its_metric(standings: dict[str, Any]):
    rig = {**STANDINGS, 'descriptor': LEADERBOARDS['leaderboards'][1]['descriptor'], 'rows': []}
    platform = Platform({routes.LEADERBOARDS_GET: (rig, 200)})
    assert standings['run'](make_client(platform), LeaderboardRef('a.board')) == [
        'a.board: ranks by reward',
        'no entries',
    ]


def test_an_unknown_board_names_the_leaderboards_on_offer(standings: dict[str, Any]):
    platform = Platform({routes.LEADERBOARDS_GET: (NOT_FOUND, 404), routes.LEADERBOARDS_LIST: (LEADERBOARDS, 200)})

    with pytest.raises(SystemExit) as exit_info:
        standings['run'](make_client(platform), LeaderboardRef('nope'))

    assert str(exit_info.value) == 'leaderboard not found: nope\nleaderboards on offer: a.board, b.rig'


def test_an_empty_board_slug_is_refused_before_any_request(standings: dict[str, Any]):
    with pytest.raises(SystemExit) as exit_info:
        standings['main'](['--board='])
    assert str(exit_info.value) == "not a LeaderboardRef: ''"


def test_the_client_sends_no_key_even_when_the_environment_holds_one(standings: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    platform = Platform({routes.LEADERBOARDS_LIST: (LEADERBOARDS, 200)})
    transport = httpx.MockTransport(platform)
    with standings['anonymous_client'](client=httpx.Client(base_url=BASE, transport=transport)) as client:
        standings['run'](client, None)
    assert AUTH_HEADER not in platform.requests[0].headers


def test_an_empty_platform_url_is_a_cli_error_before_any_request(standings: dict[str, Any]):
    with pytest.raises(SystemExit) as exit_info:
        standings['main'](['--platform-url='])
    assert 'base_url is empty' in str(exit_info.value)


SAVED = 'http://saved.test'


def listed_platforms(standings: dict[str, Any], monkeypatch, argv: list[str]) -> list[str | None]:
    """Run `main` with `argv` and return the platform each anonymous client was made for."""
    platforms: list[str | None] = []

    def anonymous_client(platform_url: str | None) -> PlatformClient:
        platforms.append(platform_url)
        return make_client(Platform({routes.LEADERBOARDS_LIST: (LEADERBOARDS, 200)}))

    monkeypatch.setitem(standings['main'].__globals__, 'anonymous_client', anonymous_client)
    standings['main'](argv)
    return platforms


def test_the_listing_reads_the_platform_that_register_saved(standings: dict[str, Any], monkeypatch, tmp_path):
    write_config(tmp_path, Config(platform_url=SAVED, api_key=ApiKey('pk_live_saved')))
    assert listed_platforms(standings, monkeypatch, []) == [SAVED]


def test_a_platform_url_flag_wins_over_the_saved_platform(standings: dict[str, Any], monkeypatch, tmp_path):
    write_config(tmp_path, Config(platform_url=SAVED, api_key=ApiKey('pk_live_saved')))
    assert listed_platforms(standings, monkeypatch, [f'--platform-url={BASE}']) == [BASE]
