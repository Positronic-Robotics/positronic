import io
import socket
import time

import httpx
import numpy as np
import PIL.Image
import pytest
from fastapi.testclient import TestClient
from positronic_model_server.serialization import deserialise, encode_jpeg, serialise
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, UnixSocketAddress

from positronic import keys
from positronic.teleop import service as service_module
from positronic.teleop import session
from positronic.teleop.service import Phase, TeleopService
from positronic.teleop.window import Window, window_app
from positronic.tests.testing_inference import free_port


@pytest.fixture
def service(socket_path):
    with TeleopService(socket_path, ready_limit=5.0, silence_limit=0.1, hold_sec=0.2) as served:
        yield served


@pytest.fixture
def window(service):
    return TestClient(window_app(service), base_url='http://127.0.0.1:5006')


@pytest.fixture
def waiting(service, socket_path):
    """A session that waits for Ready."""
    connection = registry.client_wire('websocket_unix').dial(UnixSocketAddress(socket_path, SESSION_PATH, ''), None, 5)
    connection.send(serialise({session.KIND: session.Request.READY}))
    while service.phase is not Phase.READY_WAIT:
        time.sleep(0.005)
    yield connection
    connection.close()


def test_the_window_says_when_no_trial_waits(window):
    assert window.get('/status').json() == {
        'phase': 'idle',
        'task': '',
        'control': False,
        'attempt_started': False,
        'cameras': [],
    }


def _ready_from_the_headset(service, waiting) -> None:
    """Press B on the headset, as only the headset or the phone gives Ready."""
    buttons = [0.0] * 6
    buttons[service_module._B] = 1.0
    service.take_sample(None, buttons)
    assert deserialise(waiting.recv(5)) == {session.KIND: session.Reply.READY}


def test_the_window_offers_no_ready(window, waiting):
    assert window.post('/ready').status_code in (404, 405)
    assert 'id="ready"' not in window.get('/').text
    assert window.get('/status').json()['phase'] == 'ready_wait'


def test_cancel_from_the_window_cancels_the_trial(window, waiting):
    assert window.post('/cancel').status_code == 200
    assert deserialise(waiting.recv(5)) == {session.KIND: session.Reply.CANCELLED}
    assert window.get('/status').json()['phase'] == 'idle'


def test_cancel_with_no_trial_that_waits_is_refused(window):
    answer = window.post('/cancel')
    assert answer.status_code == 409
    assert answer.json()['detail'] == 'no trial waits for Ready'


def test_a_page_on_another_site_cannot_cancel_a_trial(window, waiting):
    assert window.post('/cancel', headers={'origin': 'https://example.com'}).status_code == 403
    assert window.get('/status').json()['phase'] == 'ready_wait'


def test_the_window_lists_and_serves_each_camera_of_the_run(window, service, waiting):
    _ready_from_the_headset(service, waiting)
    images = {
        keys.WRIST_IMAGE: np.full((24, 32, 3), 50, np.uint8),
        keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 200, np.uint8),
    }
    encoded = {name: encode_jpeg(image) for name, image in images.items()}
    waiting.send(serialise({session.KIND: session.Request.FRAME, session.DISPLAY: {session.IMAGES: encoded}}))
    waiting.recv(5)
    assert window.get('/status').json()['cameras'] == [keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE]
    for name, image in images.items():
        answer = window.get(f'/camera/{name}.jpg')
        assert answer.headers['content-type'] == 'image/jpeg'
        assert np.asarray(PIL.Image.open(io.BytesIO(answer.content)).convert('RGB')).shape == image.shape
    assert window.get('/camera/image.nowhere.jpg').status_code == 404


def test_the_page_offers_cancel(window):
    assert 'id="cancel"' in window.get('/').text


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
    waiting.send(serialise({session.KIND: session.Request.FRAME, session.DISPLAY: {session.TASK: 'pick up the cube'}}))
    waiting.recv(5)
    assert window.get('/status').json()['task'] == 'pick up the cube'
    waiting.close()
    deadline = time.monotonic() + 5
    while service.phase is not Phase.IDLE:
        assert time.monotonic() < deadline, 'the session never ended'
        time.sleep(0.005)
    assert window.get('/status').json() == {
        'phase': 'idle',
        'task': '',
        'control': False,
        'attempt_started': False,
        'cameras': [],
    }
