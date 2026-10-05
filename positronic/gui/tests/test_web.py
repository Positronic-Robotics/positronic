from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pimm.world import SystemClock
from positronic.eval import Task
from positronic.gui.station import Outcome, Station, terminal_payload
from positronic.gui.web import Action, CameraFeed, StationConsole
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


@pytest.fixture
def console() -> Iterator[_Console]:
    console = _Console()
    yield console
    console.should_stop.push(True)
    console.feed.stream.close()


def test_the_page_and_its_assets_are_served(console):
    assert 'Station console' in console.client.get('/').text
    assert console.client.get('/static/station.js').status_code == 200
    assert console.client.get('/static/station.css').status_code == 200


def test_the_status_carries_the_run_the_cameras_and_the_policy(console):
    console.feed.push(_frame(1), SystemClock().now())
    status = console.client.get('/status').json()
    assert status['run']['phase'] == 'ready'
    assert status['run']['configured'] == CONFIGURED
    assert status['run']['override'] is None
    assert status['cameras'] == [
        {'name': CAMERA, 'label': 'exterior 2', 'live': True, 'fps': 0.0, 'width': 640, 'height': 360}
    ]
    assert (status['policy'], status['host']) == ('remote', '127.0.0.1')


def test_start_hands_the_trial_to_the_control_loop_once(console):
    console.client.post('/instruction', json={'override': OVERRIDE})
    response = console.client.post('/episode/start')
    assert response.status_code == 200
    assert response.json()['run']['phase'] == 'running'
    [task] = console.actions
    assert isinstance(task, Task) and task.instruction == OVERRIDE

    again = console.client.post('/episode/start')
    assert again.status_code == 409
    assert 'already running' in again.json()['detail']
    assert len(console.actions) == 1


def test_a_verdict_hands_the_done_payload_to_the_control_loop(console):
    console.client.post('/episode/start')
    response = console.client.post('/episode/end', json={'verdict': 'fail'})
    assert response.json()['run']['phase'] == 'ending'
    assert console.actions[-1] == terminal_payload(Outcome.FAIL)


def test_the_page_offers_only_the_operator_verdicts(console):
    console.client.post('/episode/start')
    assert console.client.post('/episode/end', json={'verdict': 'timeout'}).status_code == 422
    assert len(console.actions) == 1


def test_the_instruction_is_refused_while_an_episode_runs(console):
    console.client.post('/episode/start')
    response = console.client.post('/instruction', json={'override': OVERRIDE})
    assert response.status_code == 409
    assert console.station.view(now=0.0).override is None


def test_a_post_from_another_site_is_refused(console):
    refused = console.client.post('/episode/start', headers={'Origin': 'http://example.com'})
    assert refused.status_code == 403
    assert console.actions == []
    accepted = console.client.post('/episode/start', headers={'Origin': 'http://testserver'})
    assert accepted.status_code == 200


def test_a_tile_streams_its_codec_its_init_segment_and_then_fragments(console):
    for i in range(20):
        console.feed.push(_frame(i), SystemClock().now())
    with console.client.websocket_connect(f'/video/{CAMERA}') as socket:
        assert socket.receive_text().startswith('avc1.')
        assert socket.receive_bytes() == console.feed.stream.init_segment
        for i in range(20, 40):
            console.feed.push(_frame(i), SystemClock().now())
        assert socket.receive_bytes()[4:8] == b'moof'
        console.should_stop.push(True)


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
