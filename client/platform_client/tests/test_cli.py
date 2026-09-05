"""`positronic-platform`: where the key and the platform come from, and what each command sends."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from platform_client import cli, github_device_flow, routes
from platform_client.cli import CONFIG_DIR_ENV, CONFIG_FILENAME, Config
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
    cli.write_config(config, Config(platform_url=PLATFORM, api_key=KEY))


def _record(config: Path) -> dict[str, str]:
    return json.loads((config / CONFIG_FILENAME).read_text())


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
    record_path = config / CONFIG_FILENAME
    assert _record(config) == {'platform_url': PLATFORM, 'api_key': KEY}
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    shown = capsys.readouterr().out
    assert json.loads(shown) == {
        'user_id': 'a1',
        'key_status': 'created',
        'platform_url': PLATFORM,
        'config_file': str(record_path),
    }
    assert KEY not in shown


def test_a_registration_that_issues_no_key_leaves_the_saved_pair_alone(config, monkeypatch, capsys):
    """The record pairs a key with its platform; a registration elsewhere that mints none changes neither."""
    _registered(config)

    def existing(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        return RegisterResponse.model_validate({
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'key_status': 'existing',
        })

    monkeypatch.setattr(github_device_flow, 'run_registration', existing)
    cli.main(['register', '--client-id=x', '--platform-url=https://other.example'])

    assert _record(config) == {'platform_url': PLATFORM, 'api_key': KEY}
    assert json.loads(capsys.readouterr().out)['config_file'] is None


def test_register_refuses_a_platform_that_would_show_the_token(config, monkeypatch):
    reached: list[object] = []
    monkeypatch.setattr(github_device_flow, 'run_registration', lambda *args, **kwargs: reached.append(args))

    with pytest.raises(SystemExit) as raised:
        cli.main(['register', '--client-id=x', '--platform-url=http://203.0.113.5:8080'])

    assert '--plaintext-http' in str(raised.value) and reached == []
    assert not (config / CONFIG_FILENAME).exists()


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


def test_a_key_file_is_read_before_the_record(config, gateway, tmp_path):
    _registered(config)
    key_file = tmp_path / 'other_key'
    key_file.write_text('pk_live_file\n')

    cli.main(['requests', 'get', '2a', f'--api-key-file={key_file}'])

    sent = gateway.request()
    assert sent.headers['authorization'] == 'Bearer pk_live_file' and sent.url.host == 'gateway.example'


def test_a_key_file_that_is_missing_or_empty_ends_the_command_naming_it(config, gateway, tmp_path):
    missing = tmp_path / 'no_such_key'
    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'get', '2a', f'--api-key-file={missing}'])
    assert str(missing) in str(raised.value) and 'No such file' in str(raised.value)

    empty = tmp_path / 'empty_key'
    empty.write_text('\n')
    with pytest.raises(SystemExit, match='holds no key'):
        cli.main(['requests', 'get', '2a', f'--api-key-file={empty}'])
    assert gateway.requests == []


def test_an_unreadable_record_ends_the_command_naming_the_file(config, gateway):
    (config / CONFIG_FILENAME).mkdir(parents=True)
    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'get', '2a'])
    assert str(config / CONFIG_FILENAME) in str(raised.value) and 'cannot be read' in str(raised.value)
    assert gateway.requests == []


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


@pytest.mark.parametrize(
    'flag', [['--tasks', 'a'], ['--episodes-per-endpoint', '0'], ['--cap', '0'], ['--scene', 'tote_placement=left']]
)
def test_a_file_and_a_flag_together_are_refused(config, gateway, tmp_path, flag: list[str]):
    """Every flag given counts, a zero included: the file carries the whole request."""
    _registered(config)
    path = tmp_path / 'request.json'
    path.write_text('{}')
    with pytest.raises(SystemExit, match='alone'):
        cli.main(['requests', 'create', '--from', str(path), *flag])
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


def test_an_empty_transaction_key_is_refused_rather_than_dropped(config, gateway):
    _registered(config)
    with pytest.raises(SystemExit, match='transaction_key'):
        cli.main([
            'requests',
            'create',
            '--tasks',
            'stack-the-cubes',
            '--endpoints',
            'gyros',
            '--episodes-per-endpoint',
            '1',
            '--transaction-key=',
        ])
    assert gateway.requests == []


def test_register_leaves_only_the_answer_on_stdout(config, monkeypatch, capsys):
    """The device-flow prompt goes to stderr, so a caller parses stdout as one JSON document."""

    def prompting(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        print('open https://github.com/login/device and enter WDJB-MJHT', file=sys.stderr)
        return RegisterResponse.model_validate({
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'api_key': KEY,
            'key_status': 'created',
        })

    monkeypatch.setattr(github_device_flow, 'run_registration', prompting)
    cli.main(['register', '--client-id=x', f'--platform-url={PLATFORM}'])
    captured = capsys.readouterr()
    assert json.loads(captured.out)['user_id'] == 'a1' and 'WDJB-MJHT' in captured.err


def test_a_torn_write_leaves_the_previous_record_whole(config, monkeypatch):
    """The record is staged beside its path and renamed in, so a write that dies leaves the pair as it was."""
    _registered(config)

    def dying(staged: str | Path, path: str | Path) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(os, 'replace', dying)
    with pytest.raises(OSError):
        cli.write_config(config, Config(platform_url='https://other.example', api_key=ApiKey('pk_live_new')))
    assert _record(config) == {'platform_url': PLATFORM, 'api_key': KEY}
    # The staged file went with the failure: no key sits beside the record.
    assert not list(config.glob('.*'))
    assert 'pk_live_new' not in ''.join(p.read_text() for p in config.iterdir())


def test_a_path_planted_beside_the_record_is_not_written_through(config):
    """The staged file is created for the write alone, so a link planted under a guessable name stays a link."""
    config.mkdir(mode=0o700)
    target = config / 'elsewhere'
    target.write_text('')
    planted = config / f'.{CONFIG_FILENAME}.{os.getpid()}'
    planted.symlink_to(target)

    cli.write_config(config, Config(platform_url=PLATFORM, api_key=KEY))

    assert target.read_text() == '' and planted.is_symlink()
    assert _record(config) == {'platform_url': PLATFORM, 'api_key': KEY}
    assert stat.S_IMODE((config / CONFIG_FILENAME).stat().st_mode) == 0o600


def test_a_fresh_registration_replaces_the_record_whole(config, monkeypatch, capsys):
    """Both halves move together: the new key never sits beside the old platform."""
    _registered(config)

    def minted(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        return RegisterResponse.model_validate({
            'user_id': 'b2',
            'artifact_location': 's3://artifacts/b2',
            'api_key': 'pk_live_new',
            'key_status': 'created',
        })

    monkeypatch.setattr(github_device_flow, 'run_registration', minted)
    cli.main(['register', '--client-id=x', '--platform-url=https://other.example'])

    assert _record(config) == {'platform_url': 'https://other.example', 'api_key': 'pk_live_new'}
    assert not list(config.glob('.*'))  # the staged file was renamed in, not left beside the record
    assert 'pk_live_new' not in capsys.readouterr().out


@pytest.mark.parametrize('limit', ['0', '-1', 'ten'])
def test_a_limit_that_is_not_a_positive_integer_ends_at_the_parser(config, gateway, limit: str):
    _registered(config)
    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'list', '--limit', limit])
    assert raised.value.code == 2 and gateway.requests == []


# --- the record's key reaches the record's platform and no other ------------------------------


@pytest.mark.parametrize('how', ['flag', 'env'])
def test_a_saved_key_is_not_sent_to_another_platform(config, gateway, monkeypatch, how: str):
    _registered(config)
    argv = ['requests', 'get', '2a']
    if how == 'flag':
        argv.append('--platform-url=https://other.example')
    else:
        monkeypatch.setenv(API_URL_ENV, 'https://other.example')

    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    assert PLATFORM in str(raised.value) and 'https://other.example' in str(raised.value)
    assert gateway.requests == []


def test_a_key_given_for_another_platform_reaches_it(config, gateway, monkeypatch, tmp_path):
    """The refusal guards the record's key alone: a key file or an environment key is the caller's own."""
    _registered(config)
    key_file = tmp_path / 'other_key'
    key_file.write_text('pk_live_other\n')

    cli.main(['requests', 'get', '2a', '--platform-url=https://other.example', f'--api-key-file={key_file}'])
    sent = gateway.request()
    assert sent.url.host == 'other.example' and sent.headers['authorization'] == 'Bearer pk_live_other'

    monkeypatch.setenv(API_KEY_ENV, 'pk_live_env')
    cli.main(['requests', 'get', '2a', '--platform-url=https://other.example'])
    assert gateway.request().headers['authorization'] == 'Bearer pk_live_env'


def test_a_platform_flag_naming_the_records_platform_reads_the_record(config, gateway):
    _registered(config)
    cli.main(['requests', 'get', '2a', f'--platform-url={PLATFORM}'])
    assert gateway.request().headers['authorization'] == f'Bearer {KEY}'


def test_a_trailing_slash_names_the_same_platform_and_another_path_does_not(config, gateway):
    cli.write_config(config, Config(platform_url=f'{PLATFORM}/', api_key=KEY))
    cli.main(['requests', 'get', '2a', f'--platform-url={PLATFORM}'])
    assert gateway.request().headers['authorization'] == f'Bearer {KEY}'
    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'get', '2a', f'--platform-url={PLATFORM}/other'])
    assert f'{PLATFORM}/other' in str(raised.value) and len(gateway.requests) == 1


# --- a minted key that could not be saved, and a record that cannot be read -------------------


def test_a_failed_save_after_a_minted_key_says_what_to_do_and_shows_no_key(config, monkeypatch, capsys):
    def minted(client_id: str | None, base_url: str, *, alias: str | None, rotate: bool) -> RegisterResponse:
        return RegisterResponse.model_validate({
            'user_id': 'a1',
            'artifact_location': 's3://artifacts/a1',
            'api_key': 'pk_live_new',
            'key_status': 'created',
        })

    def unwritable(directory: Path, record: Config) -> None:
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(github_device_flow, 'run_registration', minted)
    monkeypatch.setattr(cli, 'write_config', unwritable)
    with pytest.raises(SystemExit) as raised:
        cli.main(['register', '--client-id=x', f'--platform-url={PLATFORM}'])

    message = str(raised.value)
    assert 'issued a key' in message and str(config / CONFIG_FILENAME) in message and 'operator' in message
    assert 'pk_live_new' not in message and 'pk_live_new' not in capsys.readouterr().out


def test_a_malformed_record_is_refused_with_one_line_naming_the_file(config, gateway):
    config.mkdir()
    (config / CONFIG_FILENAME).write_text('{"platform_url": "https://gateway.example", "api_key": pk_live_secret')

    with pytest.raises(SystemExit) as raised:
        cli.main(['requests', 'get', '2a'])

    message = str(raised.value)
    assert str(config / CONFIG_FILENAME) in message and 'register' in message
    assert 'pk_live_secret' not in message and gateway.requests == []


def test_a_record_a_command_needs_nothing_from_is_not_read(config, gateway, monkeypatch, tmp_path):
    """The key and the platform both come from the caller, so the malformed record stays unread."""
    config.mkdir()
    (config / CONFIG_FILENAME).write_text('{"platform_url": "https://gateway.example", "api_key": pk_live_secret')
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_env')
    monkeypatch.setenv(API_URL_ENV, 'https://env.example')

    cli.main(['requests', 'get', '2a'])
    assert gateway.request().url.host == 'env.example'

    # A key of the caller's own with no platform of their own still needs the record's platform.
    monkeypatch.delenv(API_URL_ENV)
    with pytest.raises(SystemExit, match='not a config record'):
        cli.main(['requests', 'get', '2a'])


def test_an_empty_after_is_refused_as_an_id_rather_than_read_as_no_cursor(config, gateway):
    _registered(config)
    with pytest.raises(SystemExit, match='not a request id'):
        cli.main(['requests', 'list', '--after='])
    assert gateway.requests == []

    cli.main(['requests', 'list'])
    assert 'after' not in gateway.request().url.params


def test_a_task_id_that_could_never_be_a_catalogue_key_ends_the_command(config, gateway):
    _registered(config)
    with pytest.raises(SystemExit, match='not a task id'):
        cli.main([
            'requests',
            'create',
            '--tasks',
            'Eight Spoons',
            '--endpoints',
            'gyros',
            '--episodes-per-endpoint',
            '1',
        ])
    assert gateway.requests == []
