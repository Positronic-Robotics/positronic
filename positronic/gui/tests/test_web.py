import queue
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import pimm
from pimm.world import SystemClock
from positronic.eval import Task
from positronic.gui.station import Episode, Outcome, Phase, RunView, Station, terminal_payload
from positronic.gui.web import (
    STATIC_DIR,
    Action,
    CameraFeed,
    CameraView,
    EndBody,
    EndTrial,
    InstructionBody,
    StationConsole,
    Status,
)
from positronic.tests.testing_coutils import ManualCommandReceiver

CONFIGURED = 'put the cup in the tote'
OVERRIDE = 'put the red cup in the grey tote'
CAMERA = 'image.exterior_2'


def _trial() -> Task:
    return Task(instruction_source=CONFIGURED, timeout_sec=None)


def _frame(i: int) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, : 20 * i % 640] = 200
    return frame


class _Console:
    """A station page served in-process, with the actions it hands to the control loop collected in a list."""

    def __init__(self):
        self.actions: list[Action] = []
        self.feed = CameraFeed()
        self.should_stop = ManualCommandReceiver[bool]()
        self.should_stop.push(False)
        self.station = Station(_trial)
        console = StationConsole(_trial, policy='remote', host='127.0.0.1', port=0)
        app = console.build_app(self.station, {CAMERA: self.feed}, self.actions.append, SystemClock(), self.should_stop)
        self.client = TestClient(app)

    def post(self, path: str, body: EndBody | InstructionBody | None = None) -> Status:
        response = self.client.post(path, json=None if body is None else body.model_dump(mode='json'))
        response.raise_for_status()
        return Status.model_validate(response.json())


@pytest.fixture
def console() -> Iterator[_Console]:
    console = _Console()
    yield console
    console.should_stop.push(True)
    console.feed.stream.close()


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
    [task] = console.actions
    assert isinstance(task, Task) and task.instruction == OVERRIDE

    again = console.client.post('/episode/start')
    assert again.status_code == 409
    assert 'already running' in again.json()['detail']
    assert len(console.actions) == 1


def test_a_verdict_hands_the_end_of_the_trial_to_the_control_loop(console):
    console.post('/episode/start')
    assert console.post('/episode/end', EndBody(verdict=Outcome.FAIL)).run.phase is Phase.ENDING
    assert console.actions[-1] == EndTrial(terminal_payload(Outcome.FAIL))


def test_the_page_offers_only_the_operator_verdicts(console):
    console.post('/episode/start')
    assert console.client.post('/episode/end', json={'verdict': Outcome.TIMEOUT.value}).status_code == 422
    assert len(console.actions) == 1


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
    assert console.actions == []
    accepted = console.client.post('/episode/start', headers={'Origin': 'http://testserver'})
    assert accepted.status_code == 200


PAGE_ORIGIN = {'Origin': 'http://testserver'}


def test_a_tile_streams_its_codec_its_init_segment_and_then_fragments_to_the_page(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with console.client.websocket_connect(f'/video/{CAMERA}', headers=PAGE_ORIGIN) as socket:
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
        with console.client.websocket_connect(f'/video/{CAMERA}', headers={'Origin': 'http://example.com'}) as socket:
            socket.receive_text()


def test_a_tile_for_an_unknown_camera_is_refused(console):
    with pytest.raises(WebSocketDisconnect):
        with console.client.websocket_connect('/video/image.nowhere') as socket:
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
    console = StationConsole(_trial, policy='remote', host='127.0.0.1', port=0)
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
    console = StationConsole(_trial, policy='remote', host='127.0.0.1', port=0)
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
