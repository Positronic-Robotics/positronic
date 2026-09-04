"""`positronic-platform`: where the key and the platform come from, and what each command sends."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import cli, github_device_flow, routes
from platform_client.cli import API_KEY_FILENAME, CONFIG_DIR_ENV, PLATFORM_URL_FILENAME
from platform_client.client import API_KEY_ENV, API_URL_ENV
from platform_client.enums import CameraVantage, Placement
from platform_client.errors import TASKS_DETAIL
from platform_client.ids import ApiKey
from platform_client.requests import SceneAsk
from platform_client.responses import RegisterResponse

PLATFORM = 'https://gateway.example'
KEY = ApiKey('pk_live_secret')
REQUEST_VIEW: dict[str, Any] = {
    'request_id': '2a',
    'status': 'received',
    'episodes': {'total': 10, 'done': 0, 'outstanding': 10},
    'runs': [],
}


class Gateway:
    """Answers one canned payload and keeps every request that reached it."""

    def __init__(self, status: int = 200, payload: object = None) -> None:
        self.status = status
        self.payload = REQUEST_VIEW if payload is None else payload
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    def request(self) -> httpx.Request:
        assert self.requests, 'no request reached the gateway'
        return self.requests[-1]


@pytest.fixture
def config(tmp_path, monkeypatch) -> Path:
    """A config directory of this test's own, with no key and no platform in the environment."""
    directory = tmp_path / 'config'
    monkeypatch.setenv(CONFIG_DIR_ENV, str(directory))
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_URL_ENV, raising=False)
    return directory


@pytest.fixture
def gateway(monkeypatch) -> Gateway:
    """Every client the command opens answers as this gateway."""
    answering = Gateway()
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, 'Client', lambda **kwargs: real_client(**{**kwargs, 'transport': httpx.MockTransport(answering.handle)})
    )
    return answering


def _registered(config: Path) -> None:
    cli.write_config(config, api_key=KEY, platform_url=PLATFORM)


# --- register ---------------------------------------------------------------------------------


def test_register_writes_the_key_by_path_and_the_platform_beside_it(config, monkeypatch, capsys):
    asked: list[tuple[str | None, str, str | None, bool]] = []

    def record(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        asked.append((client_id, base_url, alias, rotate))
        return RegisterResponse.model_validate({
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'api_key': KEY,
            'key_status': 'created',
        })

    monkeypatch.setattr(github_device_flow, 'run_registration', record)
    cli.main(['register', '--client-id=x', f'--platform-url={PLATFORM}', '--alias=demo'])

    assert asked == [('x', PLATFORM, 'demo', False)]
    key_path = config / API_KEY_FILENAME
    assert key_path.read_text() == f'{KEY}\n'
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert (config / PLATFORM_URL_FILENAME).read_text() == f'{PLATFORM}\n'
    shown = capsys.readouterr().out
    assert str(key_path) in shown and KEY not in shown


def test_a_registration_that_issues_no_key_leaves_the_key_file_alone(config, monkeypatch):
    _registered(config)

    def existing(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        return RegisterResponse.model_validate({
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'key_status': 'existing',
        })

    monkeypatch.setattr(github_device_flow, 'run_registration', existing)
    cli.main(['register', '--client-id=x', f'--platform-url={PLATFORM}'])

    assert (config / API_KEY_FILENAME).read_text() == f'{KEY}\n'


def test_register_refuses_a_platform_that_would_show_the_token(config, monkeypatch):
    reached: list[object] = []
    monkeypatch.setattr(github_device_flow, 'run_registration', lambda *args, **kwargs: reached.append(args))

    with pytest.raises(SystemExit) as raised:
        cli.main(['register', '--client-id=x', '--platform-url=http://203.0.113.5:8080'])

    assert '--plaintext-http' in str(raised.value) and reached == []
    assert not (config / PLATFORM_URL_FILENAME).exists()


# --- where the key and the platform come from -------------------------------------------------


def test_the_key_and_the_platform_are_read_from_the_config_directory(config, gateway):
    _registered(config)
    cli.main(['requests', 'get', '2a'])

    sent = gateway.request()
    assert str(sent.url) == f'{PLATFORM}{routes.REQUESTS_GET}?id=2a'
    assert sent.headers['authorization'] == f'Bearer {KEY}'


def test_the_environment_wins_over_the_config_directory(config, gateway, monkeypatch):
    _registered(config)
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_env')
    monkeypatch.setenv(API_URL_ENV, 'https://env.example')

    cli.main(['requests', 'get', '2a'])

    sent = gateway.request()
    assert sent.url.host == 'env.example' and sent.headers['authorization'] == 'Bearer pk_live_env'


def test_the_flags_win_over_the_environment(config, gateway, monkeypatch, tmp_path):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_env')
    monkeypatch.setenv(API_URL_ENV, 'https://env.example')
    key_file = tmp_path / 'other_key'
    key_file.write_text('pk_live_file\n')

    cli.main(['requests', 'get', '2a', f'--platform-url={PLATFORM}', f'--api-key-file={key_file}'])

    sent = gateway.request()
    assert sent.url.host == 'gateway.example'
    # The environment names a key, so the flag's file is read only when the environment names none.
    assert sent.headers['authorization'] == 'Bearer pk_live_env'


def test_no_key_anywhere_ends_the_command_naming_register(config, gateway):
    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'get', '2a'])
    assert 'register' in str(raised.value) and gateway.requests == []


# --- requests create --------------------------------------------------------------------------


def test_create_from_flags_posts_the_ask(config, gateway, capsys):
    _registered(config)
    gateway.payload = {'request_id': '2a', 'status': 'received'}

    cli.main([
        'requests',
        'create',
        '--tasks',
        'eight-spoons-into-grey-tote',
        'stack-the-cubes',
        '--endpoints',
        'gyros=wss://gyros.example/ws',
        'pi05',
        '--episodes-per-endpoint',
        '10',
        '--cap',
        '180',
        '--preset',
        'runway_ziyi',
        '--scene',
        'tote_placement=random',
        '--scene',
        'camera.side=left',
        '--slug',
        'ziyi',
        '--transaction-key',
        'round-1',
    ])

    sent = gateway.request()
    assert sent.url.path == routes.REQUESTS_CREATE
    body = json.loads(sent.content)
    assert [task['task_id'] for task in body['tasks']] == ['eight-spoons-into-grey-tote', 'stack-the-cubes']
    assert [(entry['name'], entry['url']) for entry in body['endpoints']] == [
        ('gyros', 'wss://gyros.example/ws'),
        ('pi05', None),
    ]
    assert body['episodes_per_endpoint'] == 10 and body['cap_per_episode_sec'] == 180
    assert body['policy_preset'] == 'runway_ziyi' and body['slug'] == 'ziyi'
    assert body['transaction_key'] == 'round-1'
    assert body['scene'] == {'tote_placement': 'random', 'camera_vantage': None, 'external_cameras': {'side': 'left'}}
    assert json.loads(capsys.readouterr().out) == {'request_id': '2a', 'status': 'received'}


def test_create_from_a_file_posts_the_file_whole(config, gateway, tmp_path):
    _registered(config)
    gateway.payload = {'request_id': '2a', 'status': 'received'}
    ask = {
        'tasks': [{'task_id': 'stack-the-cubes'}],
        'endpoints': [{'name': 'pi05', 'kind': 'served', 'provider': 'droid_cohost', 'spec': 'pi05'}],
        'episodes_per_endpoint': 3,
    }
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(ask))

    cli.main(['requests', 'create', '--from', str(path)])

    body = json.loads(gateway.request().content)
    assert body['endpoints'][0]['kind'] == 'served' and body['endpoints'][0]['provider'] == 'droid_cohost'
    assert body['episodes_per_endpoint'] == 3


def test_a_file_and_the_flags_together_are_refused(config, gateway, tmp_path):
    _registered(config)
    path = tmp_path / 'request.json'
    path.write_text('{}')
    with pytest.raises(SystemExit, match='alone'):
        cli.main(['requests', 'create', '--from', str(path), '--tasks', 'a'])
    assert gateway.requests == []


def test_a_file_that_is_not_a_request_names_the_field(config, gateway, tmp_path):
    _registered(config)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps({'tasks': [], 'episodes_per_endpoint': 1}))
    with pytest.raises(SystemExit, match='tasks'):
        cli.main(['requests', 'create', '--from', str(path)])


def test_create_without_the_tasks_or_the_count_names_what_is_missing(config, gateway):
    _registered(config)
    with pytest.raises(SystemExit, match='--tasks and --episodes-per-endpoint'):
        cli.main(['requests', 'create', '--endpoints', 'gyros'])
    assert gateway.requests == []


def test_scene_pairs_become_the_scene():
    assert cli.scene_from_pairs([]) is None
    assert cli.scene_from_pairs(['tote_placement=random', 'camera_vantage=phail', 'camera.side=left']) == SceneAsk(
        tote_placement=Placement.random, camera_vantage=CameraVantage.phail, external_cameras={'side': Placement.left}
    )


@pytest.mark.parametrize('pair', ['tote_placement', 'tote=left', 'camera.=left', 'tote_placement=middle'])
def test_a_scene_pair_that_names_no_field_or_no_side_is_refused(pair: str):
    with pytest.raises(SystemExit):
        cli.scene_from_pairs([pair])


def test_the_scene_keys_are_the_fields_the_model_declares():
    assert {cli.SCENE_TOTE, cli.SCENE_VANTAGE} < set(SceneAsk.model_fields)


# --- requests list, and a refusal -------------------------------------------------------------


def test_list_sends_the_cursor_and_prints_the_page(config, gateway, capsys):
    _registered(config)
    gateway.payload = {'requests': [REQUEST_VIEW], 'next': '2a'}

    cli.main(['requests', 'list', '--after', '1f', '--limit', '1'])

    assert dict(gateway.request().url.params) == {'after': '1f', 'limit': '1'}
    assert json.loads(capsys.readouterr().out)['next'] == '2a'


def test_a_refusal_ends_the_command_with_the_code_and_the_catalogue(config, gateway):
    _registered(config)
    gateway.status = 400
    gateway.payload = {
        'error': {
            'code': 'bad_request',
            'message': "unknown task 'nope'",
            'details': {TASKS_DETAIL: ['stack-the-cubes']},
        }
    }

    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'create', '--tasks', 'nope', '--endpoints', 'gyros', '--episodes-per-endpoint', '1'])

    assert str(raised.value) == "bad_request: unknown task 'nope'\nthe catalogue holds: stack-the-cubes"


def test_an_id_that_is_not_hex_is_refused_before_any_request(config, gateway):
    _registered(config)
    with pytest.raises(SystemExit, match='not a request id'):
        cli.main(['requests', 'get', 'zz'])
    assert gateway.requests == []
