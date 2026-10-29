import io
import re
import socket
import time

import httpx
import numpy as np
import PIL.Image
import pytest
from fastapi.testclient import TestClient
from positronic_model_server.serialization import encode_jpeg
from positronic_model_server.websocket_wire import ServedUnixSocket

from positronic import geom, keys
from positronic.policy.base import PolicyEnd
from positronic.policy.keys import END
from positronic.teleop import protocol
from positronic.teleop import service as service_module
from positronic.teleop.protocol import session_query
from positronic.teleop.service import Phase, TeleopService
from positronic.teleop.window import STATIC_DIR, Window, WindowStatus, window_app
from positronic.tests.testing_inference import free_port


@pytest.fixture
def service(socket_path):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=0.1, hold_sec=0.2) as served:
        yield served


@pytest.fixture
def window(service):
    return TestClient(window_app(service), base_url='http://127.0.0.1:5006')


FRAME_REQUEST = {
    protocol.TASK: 'pick up the cube',
    protocol.EE_POSE: geom.Transform3D().as_vector(geom.Rotation.Representation.QUAT),
    protocol.GRIP: [0.0],
}


def _waiting_connection(service, socket_path, connect, query: str = ''):
    """A session that waits for Ready."""
    connection = connect(socket_path, query)
    while service.phase is not Phase.READY_WAIT:
        time.sleep(0.005)
    return connection


@pytest.fixture
def waiting(service, socket_path, connect):
    return _waiting_connection(service, socket_path, connect)


def test_the_window_says_when_no_trial_waits(window):
    assert window.get('/status').json() == {'phase': 'idle', 'task': '', 'control': False, 'cameras': []}


def _ready_from_the_headset(service, waiting) -> None:
    """Press B on the headset, as only the headset or the phone gives Ready."""
    buttons = [0.0] * 6
    buttons[service_module._B] = 1.0
    service.take_sample(None, buttons)
    waiting.wait_open()


def test_the_window_offers_no_ready(window, waiting):
    assert window.post('/ready').status_code in (404, 405)
    assert 'id="ready"' not in window.get('/').text
    assert window.get('/status').json()['phase'] == 'ready_wait'


def test_cancel_from_the_window_cancels_the_trial(window, waiting):
    assert window.post('/cancel').status_code == 200
    with pytest.raises(RuntimeError, match='cancelled the trial'):
        waiting.wait_open()
    assert window.get('/status').json()['phase'] == 'idle'


def test_cancel_with_no_trial_that_waits_is_refused(window):
    answer = window.post('/cancel')
    assert answer.status_code == 409
    assert answer.json()['detail'] == 'no trial waits for Ready'


def test_a_page_on_another_site_cannot_cancel_a_trial(window, waiting):
    assert window.post('/cancel', headers={'origin': 'https://example.com'}).status_code == 403
    assert window.get('/status').json()['phase'] == 'ready_wait'


def test_the_window_lists_and_serves_each_camera_of_the_run(window, service, socket_path, connect):
    images = {
        keys.WRIST_IMAGE: np.full((24, 32, 3), 50, np.uint8),
        keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 200, np.uint8),
    }
    waiting = _waiting_connection(service, socket_path, connect, session_query(cameras=list(images)))
    _ready_from_the_headset(service, waiting)
    encoded = {name: encode_jpeg(image) for name, image in images.items()}
    waiting.frame({**FRAME_REQUEST, **encoded})
    assert window.get('/status').json()['cameras'] == [keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE]
    for name, image in images.items():
        answer = window.get(f'/camera/{name}.jpg')
        assert answer.headers['content-type'] == 'image/jpeg'
        assert np.asarray(PIL.Image.open(io.BytesIO(answer.content)).convert('RGB')).shape == image.shape
    assert window.get('/camera/image.nowhere.jpg').status_code == 404


def test_the_page_offers_cancel_and_reads_each_status_field_by_its_name(window):
    page = window.get('/').text
    assert 'id="cancel"' in page
    for field in ('phase', 'task', 'cameras'):
        assert field in WindowStatus.model_fields and f"{field}: '{field}'" in page


def test_the_window_serves_on_its_port_while_open(service):
    port = free_port()
    with Window(window_app(service), port) as served:
        for _ in range(200):
            try:
                assert httpx.get(served.url + 'status', timeout=1.0).status_code == 200
                break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            raise AssertionError('the window never answered')


def test_a_window_that_cannot_bind_its_port_raises(service):
    with socket.socket() as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(RuntimeError, match='could not serve'):
            with Window(window_app(service), port):
                pass


def test_a_request_under_another_host_name_gets_no_answer(service, waiting):
    rebound = TestClient(window_app(service), base_url='http://attacker.example:5006')
    assert rebound.get('/status').status_code == 400
    answer = rebound.post('/cancel', headers={'origin': 'http://attacker.example:5006'})
    assert answer.status_code == 400
    assert service.phase is Phase.READY_WAIT


def test_a_policy_that_leaves_mid_trial_leaves_no_old_task_in_the_window(window, service, waiting):
    _ready_from_the_headset(service, waiting)
    # A session that leaves after one frame stays for a client that opens it again, so this one makes two.
    waiting.frame(FRAME_REQUEST)
    waiting.frame(FRAME_REQUEST)
    assert window.get('/status').json()['task'] == 'pick up the cube'
    waiting.close()
    deadline = time.monotonic() + 5
    while service.phase is not Phase.IDLE:
        assert time.monotonic() < deadline, 'the session never ended'
        time.sleep(0.005)
    assert window.get('/status').json() == {'phase': 'idle', 'task': '', 'control': False, 'cameras': []}


def test_the_window_shows_no_trial_after_the_last_session_ends_with_complete(window, service, waiting):
    _ready_from_the_headset(service, waiting)
    pressed, released = [0.0] * 6, [0.0] * 6
    pressed[service_module._B] = 1.0
    service.take_sample(None, released)
    service.take_sample(None, pressed)
    service.take_sample(None, released)
    service.take_sample(None, pressed)
    time.sleep(0.3)
    service.take_sample(None, pressed)
    assert waiting.frame(FRAME_REQUEST)[END] == PolicyEnd.COMPLETE.value
    assert window.get('/status').json()['phase'] == 'ended'
    waiting.close()
    deadline = time.monotonic() + 5
    while service.phase is not Phase.IDLE:
        assert time.monotonic() < deadline, 'the session never ended'
        time.sleep(0.005)
    assert window.get('/status').json() == {'phase': 'idle', 'task': '', 'control': False, 'cameras': []}


def test_the_window_page_reads_the_fields_of_the_status():
    page = (STATIC_DIR / 'window.html').read_text()
    [fields] = re.findall(r'const STATUS_FIELD = \{([^}]*)\};', page)
    assert set(re.findall(r"'(\w+)'", fields)) == set(WindowStatus.model_fields)
