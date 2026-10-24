import queue
import re
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import configuronic as cfn
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

import pimm
from pimm.world import SystemClock
from positronic import inference
from positronic.eval import Task
from positronic.gui.station import Episode, Outcome, Phase, RunView, Station, terminal_payload
from positronic.gui.web import STATIC_DIR, CameraFeed, CameraView, EndBody, InstructionBody, StationConsole, Status
from positronic.policy import Policy, PolicyEnd
from positronic.policy import keys as policy_keys
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

CONFIGURED = 'put the cup in the tote'
OVERRIDE = 'put the red cup in the grey tote'
CAMERA = 'image.exterior_2'


def _trial() -> Task:
    return Task(instruction_source=CONFIGURED, timeout_sec=None)


def _frame(i: int) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, : 20 * i % 640] = 200
    return frame


class _Policy(Policy):
    def meta(self):
        return {policy_keys.TYPE: 'remote'}

    def run(self, runtime):
        yield
        return PolicyEnd.COMPLETE


class _Console:
    """A station page served in-process, with queued tasks and emitted verdicts collected for inspection."""

    def __init__(self, host: str = '127.0.0.1', allowed_hosts: list[str] | None = None):
        self.tasks: list[Task] = []
        self.done = RecordingEmitter[dict[str, Any]]()
        self.feed = CameraFeed()
        self.should_stop = ManualCommandReceiver[bool]()
        self.should_stop.push(False)
        self.station = Station(_trial)
        console = StationConsole(_trial, _Policy(), None, host=host, port=0, allowed_hosts=allowed_hosts)
        console.done._bind(self.done, clock=SystemClock())
        app = console.build_app(self.station, {CAMERA: self.feed}, self.tasks.append, SystemClock(), self.should_stop)
        self.client = TestClient(app, base_url='http://127.0.0.1:8080')

    def post(self, path: str, body: EndBody | InstructionBody | None = None) -> Status:
        response = self.client.post(path, json=None if body is None else body.model_dump(mode='json'))
        response.raise_for_status()
        return Status.model_validate(response.json())


# The origin of the station page that the client of `_Console` loads.
PAGE_ORIGIN = {'Origin': 'http://127.0.0.1:8080'}
# The test client dials a WebSocket on `testserver` unless the URL names the host.
PAGE_SOCKET = 'ws://127.0.0.1:8080'


@pytest.fixture
def make_console() -> Iterator[Callable[..., _Console]]:
    """Build a `_Console` with the given bind, and stop each one that it built."""
    built: list[_Console] = []

    def make(host: str = '127.0.0.1', allowed_hosts: list[str] | None = None) -> _Console:
        built.append(_Console(host, allowed_hosts))
        return built[-1]

    yield make
    for console in built:
        console.should_stop.push(True)
        console.feed.stream.close()


@pytest.fixture
def console(make_console) -> _Console:
    return make_console()


def test_the_page_and_every_asset_it_links_are_served(console):
    page = console.client.get('/')
    assert 'Station console' in page.text
    assets = [link for link in re.findall(r'(?:href|src)="([^"]+)"', page.text) if not link.startswith('data:')]
    assert sorted(assets) == ['station.css', 'station.js']
    for asset in assets:
        assert console.client.get(f'/{asset}').status_code == 200


def test_the_status_is_the_json_the_page_reads(console):
    console.feed.push(_frame(1), SystemClock().now())
    status = console.client.get('/status').json()
    status['run'].pop('now')
    assert isinstance(status['run'].pop('generation'), int)
    assert status == {
        'run': {'phase': 'ready', 'configured': CONFIGURED, 'override': None, 'override_since': None, 'episodes': []},
        'cameras': [{'name': CAMERA, 'label': 'exterior 2', 'live': True, 'fps': 0.0, 'width': 640, 'height': 360}],
        'policy': 'remote',
        'host': '127.0.0.1',
    }


def test_start_hands_the_trial_to_the_control_loop_once(console):
    console.post('/instruction', InstructionBody(override=OVERRIDE))
    assert console.post('/episode/start').run.phase is Phase.RUNNING
    [task] = console.tasks
    assert isinstance(task, Task) and task.instruction == OVERRIDE

    again = console.client.post('/episode/start')
    assert again.status_code == 409
    assert 'already running' in again.json()['detail']
    assert len(console.tasks) == 1


def test_each_verdict_emits_done_without_waiting_for_the_control_loop(console):
    console.post('/episode/start')
    assert console.post('/episode/end', EndBody(verdict=Outcome.FAIL)).run.phase is Phase.ENDING
    assert len(console.tasks) == 1
    [(_, payload)] = console.done.emitted
    assert payload == terminal_payload(Outcome.FAIL)
    assert console.post('/episode/end', EndBody(verdict=Outcome.PASS)).run.phase is Phase.ENDING
    assert [payload for _, payload in console.done.emitted] == [
        terminal_payload(Outcome.FAIL),
        terminal_payload(Outcome.PASS),
    ]


def test_the_page_offers_only_the_operator_verdicts(console):
    console.post('/episode/start')
    assert console.client.post('/episode/end', json={'verdict': Outcome.TIMEOUT.value}).status_code == 422
    assert len(console.tasks) == 1


def test_the_instruction_is_refused_while_an_episode_runs(console):
    console.post('/episode/start')
    response = console.client.post('/instruction', json=InstructionBody(override=OVERRIDE).model_dump())
    assert response.status_code == 409
    assert console.station.view(now=0.0).override is None


def test_end_run_is_refused_while_an_episode_runs_and_ends_the_run_after_it(console):
    console.post('/episode/start')
    refused = console.client.post('/run/end')
    assert refused.status_code == 409
    assert refused.json()['detail'] == 'finish the episode first, then end the run'
    console.station.close(Outcome.PASS, now=1.0)
    assert console.post('/run/end').run.phase is Phase.RUN_ENDED


def test_a_post_from_another_site_is_refused(console):
    refused = console.client.post('/episode/start', headers={'Origin': 'http://example.com'})
    assert refused.status_code == 403
    assert console.tasks == []
    accepted = console.client.post('/episode/start', headers=PAGE_ORIGIN)
    assert accepted.status_code == 200


def test_a_tile_streams_its_codec_its_init_segment_and_then_fragments_to_the_page(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with console.client.websocket_connect(f'{PAGE_SOCKET}/video/{CAMERA}', headers=PAGE_ORIGIN) as socket:
        assert socket.receive_text().startswith('avc1.')
        assert socket.receive_bytes() == console.feed.stream.init_segment
        for i in range(20, 40):
            console.feed.push(_frame(i), SystemClock().now())
        assert socket.receive_bytes()[4:8] == b'moof'
        console.should_stop.push(True)


def test_a_tile_is_refused_to_a_page_from_another_site(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with pytest.raises(WebSocketDisconnect):
        with console.client.websocket_connect(
            f'{PAGE_SOCKET}/video/{CAMERA}', headers={'Origin': 'http://example.com'}
        ) as socket:
            socket.receive_text()


def test_a_tile_for_an_unknown_camera_is_refused(console):
    with pytest.raises(WebSocketDisconnect):
        with console.client.websocket_connect(f'{PAGE_SOCKET}/video/image.nowhere') as socket:
            socket.receive_text()


def test_a_camera_reads_live_with_its_rate_until_its_frames_stop():
    feed = CameraFeed()
    for i in range(11):
        feed.push(_frame(i), now=100.0 + i * 0.1)
    live = feed.view(CAMERA, now=101.05)
    assert live.live and live.fps == pytest.approx(10.0)
    stale = feed.view(CAMERA, now=103.0)
    assert not stale.live and stale.fps == 0.0
    assert (stale.width, stale.height) == (640, 360)
    feed.stream.close()


class _Frame:
    def __init__(self, array: np.ndarray):
        self.array = array


def _console_with_camera() -> tuple[StationConsole, ManualCommandReceiver, CameraFeed]:
    console = StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0)
    camera = ManualCommandReceiver()
    console.cameras[CAMERA]._bind(camera)
    return console, camera, CameraFeed()


def test_a_camera_with_no_data_keeps_its_last_tile():
    console, camera, feed = _console_with_camera()
    camera.push(_Frame(_frame(1)))
    console._push_frames({CAMERA: feed}, SystemClock())
    camera.push(pimm.SignalError('camera lost'))
    console._push_frames({CAMERA: feed}, SystemClock())

    assert feed.view(CAMERA, now=0.0).width == 640
    feed.stream.close()


def test_a_camera_that_gives_data_again_updates_its_tile():
    console, camera, feed = _console_with_camera()
    camera.push(pimm.SignalError('camera lost'))
    console._push_frames({CAMERA: feed}, SystemClock())
    camera.push(_Frame(_frame(1)))
    console._push_frames({CAMERA: feed}, SystemClock())

    assert feed.view(CAMERA, now=0.0).width == 640
    feed.stream.close()


class _Failed(pimm.calls.Answer[dict[str, Any]]):
    """An episode answer that the harness failed with ``error``."""

    def __init__(self, error: Exception):
        self._error = error

    def done(self) -> bool:
        return True

    def result(self) -> dict[str, Any]:
        raise self._error


def test_an_arm_error_that_fails_an_episode_reaches_the_page_and_is_not_skipped_as_a_frame_is():
    console = StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0)
    station = Station(_trial)
    station.start(now=0.0)
    console._episode = _Failed(pimm.SignalError('Release the emergency stop button'))

    console._drive_episode(station, queue.SimpleQueue(), SystemClock())

    [episode] = station.view(now=0.0).episodes
    assert episode.outcome is Outcome.ERROR
    assert episode.error is not None and episode.error.endswith('Release the emergency stop button')


SCRIPT = (STATIC_DIR / 'station.js').read_text()


def _script_part(pattern: str, flags: re.RegexFlag) -> str:
    match = re.search(pattern, SCRIPT, flags)
    assert match is not None, f'the page script has no {pattern}'
    return match.group(1)


def _script_object(name: str) -> str:
    """The body of ``const <name> = {...};`` in the page script."""
    return _script_part(rf'const {name} = {{(.*?)}};', re.DOTALL)


def test_the_page_names_the_phases_and_outcomes_the_server_sends():
    assert set(re.findall(r"'([^']*)'", _script_object('PHASE'))) == {phase.value for phase in Phase}
    assert set(re.findall(r"'([^']*)'", _script_object('OUTCOME'))) == {outcome.value for outcome in Outcome}


def test_the_page_sends_the_request_fields_the_server_reads():
    sent = set(re.findall(r'\(\{ (\w+):', _script_object('REQUEST')))
    assert sent == set(InstructionBody.model_fields) | set(EndBody.model_fields)


def test_the_page_reads_every_status_field_the_server_sends():
    reader = _script_part(r'^function readStatus\(answer\) \{(.*?)^\}$', re.MULTILINE | re.DOTALL)
    read = set(re.findall(r'\b(?:answer|run|e|c)\.(\w+)', reader))
    assert read == {field for model in (Status, RunView, Episode, CameraView) for field in model.model_fields}


def test_the_page_calls_the_routes_the_server_serves(console):
    served = {route.path for route in console.client.app.routes}
    routes = dict(re.findall(r"(\w+): '([^']*)'", _script_object('ROUTES')))
    video = routes.pop('video')
    assert set(routes.values()) <= served
    assert f'{video}/{{name}}' in served


def test_a_status_taken_before_a_change_is_older_than_the_answer_to_the_change(console):
    before = Status.model_validate(console.client.get('/status').json())
    after = console.post('/instruction', InstructionBody(override=OVERRIDE))
    assert after.run.generation > before.run.generation


def _rebound(console: '_Console') -> TestClient:
    """A client under a name that a DNS rebind points at the console."""
    return TestClient(console.client.app, base_url='http://attacker.example:8080')


def test_a_page_under_another_host_name_cannot_read_the_console(console):
    assert _rebound(console).get('/status').status_code == 400


def test_a_page_under_another_host_name_cannot_start_an_episode(console):
    rebound = _rebound(console)
    assert rebound.post('/episode/start', headers={'origin': 'http://attacker.example:8080'}).status_code == 400
    assert console.tasks == []


def test_a_page_under_another_host_name_gets_no_video(console):
    with pytest.raises(WebSocketDenialResponse):
        with _rebound(console).websocket_connect(f'ws://attacker.example:8080/video/{CAMERA}'):
            pass


def test_the_console_answers_its_loopback_names(console):
    for name in ('127.0.0.1', 'localhost'):
        assert TestClient(console.client.app, base_url=f'http://{name}:8080').get('/status').status_code == 200


@pytest.mark.parametrize(
    ('host', 'allowed_hosts', 'answered', 'refused'),
    [
        ('192.168.1.20', None, ['192.168.1.20'], ['attacker.example', 'localhost']),
        ('0.0.0.0', None, ['attacker.example', 'station.lab'], []),
        ('0.0.0.0', ['station.lab'], ['station.lab'], ['attacker.example']),
        ('::1', None, ['localhost'], ['attacker.example']),
    ],
)
def test_the_console_answers_the_names_of_its_bind_or_the_names_it_is_given(
    make_console, host, allowed_hosts, answered, refused
):
    console = make_console(host, allowed_hosts)
    for name in answered:
        assert TestClient(console.client.app, base_url=f'http://{name}:8080').get('/status').status_code == 200
    for name in refused:
        assert TestClient(console.client.app, base_url=f'http://{name}:8080').get('/status').status_code == 400


@pytest.mark.parametrize(('host', 'allowed_hosts'), [('fd00::20', None), ('127.0.0.1', ['::1'])])
def test_an_ipv6_address_that_the_host_check_cannot_read_is_refused(host, allowed_hosts):
    with pytest.raises(ValueError, match='IPv6'):
        StationConsole(_trial, _Policy(), None, host=host, port=0, allowed_hosts=allowed_hosts)


def test_a_name_in_capitals_matches_the_lower_case_host_that_a_browser_sends(make_console):
    console = make_console('192.168.1.20', ['Station.LAB'])
    assert TestClient(console.client.app, base_url='http://station.lab:8080').get('/status').status_code == 200


@pytest.mark.parametrize(
    ('allowed_hosts', 'match'),
    [
        ([], 'names no host'),
        (['station.lab:8080'], 'port or an IPv6 address'),
        (['foo*'], 'no wildcard'),
        (['*.a*.lab'], 'no wildcard'),
        ('station.lab', 'not the string'),
    ],
)
def test_allowed_hosts_that_the_host_check_cannot_use_are_refused_at_once(allowed_hosts, match):
    with pytest.raises((ValueError, TypeError), match=match):
        StationConsole(_trial, _Policy(), None, host='127.0.0.1', port=0, allowed_hosts=allowed_hosts)


def test_a_wildcard_among_the_given_names_logs_that_the_console_answers_any_name(caplog):
    StationConsole(_trial, _Policy(), None, host='192.168.1.20', port=0, allowed_hosts=['*'])
    assert 'answers any host name' in caplog.text


@pytest.mark.parametrize(
    ('flag', 'names'), [('station.lab', ['station.lab']), ('["a.lab", "b.lab"]', ['a.lab', 'b.lab'])]
)
def test_the_web_command_takes_a_bare_name_or_a_list_for_allowed_hosts(monkeypatch, flag, names):
    built = {}

    def console(*args, **kwargs):
        built.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(inference, 'StationConsole', console)
    monkeypatch.setattr(sys, 'argv', ['web', f'--allowed_hosts={flag}'])
    with pytest.raises(SystemExit):
        cfn.cli(
            cfn.Config(
                inference.web, policy=_Policy(), embodiment=types.SimpleNamespace(simulated=False), next_task=_trial
            )
        )
    assert built['allowed_hosts'] == names
