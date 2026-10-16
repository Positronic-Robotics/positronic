import io
import time

import httpx
import numpy as np
import PIL.Image
import pytest
from fastapi.testclient import TestClient
from positronic_model_server.serialization import deserialise, serialise
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, UnixSocketAddress

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
    return TestClient(window_app(service))


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
    assert window.get('/status').json() == {'phase': 'IDLE', 'task': '', 'control': False, 'attempt_started': False}


def test_ready_from_the_window_answers_the_ready_request(window, waiting):
    assert window.get('/status').json()['phase'] == 'READY_WAIT'
    assert window.post('/ready').status_code == 200
    assert deserialise(waiting.recv(5)) == {session.KIND: session.Reply.READY}
    assert window.get('/status').json()['phase'] == 'RUNNING'


def test_cancel_from_the_window_cancels_the_trial(window, waiting):
    assert window.post('/cancel').status_code == 200
    assert deserialise(waiting.recv(5)) == {session.KIND: session.Reply.CANCELLED}
    assert window.get('/status').json()['phase'] == 'IDLE'


def test_ready_with_no_trial_that_waits_is_refused(window):
    answer = window.post('/ready')
    assert answer.status_code == 409
    assert answer.json()['detail'] == 'no trial waits for Ready'


def test_a_page_on_another_site_cannot_ready_a_trial(window, waiting):
    assert window.post('/ready', headers={'origin': 'https://example.com'}).status_code == 403
    assert window.get('/status').json()['phase'] == 'READY_WAIT'


def test_the_window_shows_the_view_of_the_headset(window, service):
    answer = window.get('/view.jpg')
    assert answer.headers['content-type'] == 'image/jpeg'
    shown = np.asarray(PIL.Image.open(io.BytesIO(answer.content)).convert('RGB'))
    assert shown.shape == service.view().shape


def test_the_page_offers_ready_and_cancel(window):
    page = window.get('/').text
    assert 'id="ready"' in page and 'id="cancel"' in page


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
