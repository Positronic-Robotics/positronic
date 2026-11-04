import base64
import http.client
import json
import ssl
import time
import types

import cv2
import numpy as np
import pytest
from shared.faceblur import FaceBlurrer, _self_check_frame

from positronic import keys
from positronic.policy.codecs import ACTION
from positronic.vendors import dream
from positronic.vendors.dream import codecs, server
from positronic.vendors.dream.limits import ChunkLimits, YamChunkLimiter

from .fake_gateway import ROWS, FakeGateway, hold

CONFIG = {'checkpoint': 'dreamlabs/dream-1-beta', 'timeout_ms': 5000, 'strict': False, '_record': False}
POSE = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    fake = FakeGateway()
    monkeypatch.setenv('DREAM_GATEWAY_URL', fake.url)
    monkeypatch.setenv('DREAM_API_KEY', 'dl_test_key')
    # The real credentials file of this machine must not reach the tests.
    monkeypatch.setenv('DREAM_CREDENTIALS', str(tmp_path / 'no-credentials'))
    yield fake
    fake.close()


@pytest.fixture(scope='module')
def blurrer() -> FaceBlurrer:
    return FaceBlurrer()


@pytest.fixture
def model(gateway, blurrer, tmp_path):
    limiter = YamChunkLimiter(ChunkLimits(), codecs.FPS)
    built = server.DreamModel(
        CONFIG, limiter, blurrer, tmp_path, retry_deadline_s=30.0, session_idle_s=60.0, meta={'type': 'dream'}
    )
    yield built
    built.close()


def _request(frames: int = 3, view: np.ndarray | None = None) -> dict:
    if view is None:
        view = np.full((176, 320, 3), 120, np.uint8)
    observation = {dream.PROMPT: 'fold the towel', **dream.STATION_ROBOT}
    for slot in dream.ARM_SLOTS.values():
        observation[dream.JOINT_POS.format(slot=slot)] = POSE
        observation[dream.GRIPPER_POS.format(slot=slot)] = [1.0]
    window = np.stack([view] * frames)
    return {
        dream.OBSERVATION: observation,
        dream.FRAMES: {dream.FRAME_WINDOW.format(camera=camera): window for camera in dream.CAMERAS},
    }


def _record(tmp_path, session_id: str) -> dict:
    return json.loads((tmp_path / f'{session_id}.json').read_text())


def test_a_call_answers_one_action_per_row_left_arm_first(model, gateway):
    actions = model(_request(), session_id='a')

    assert len(actions) == ROWS
    np.testing.assert_allclose(actions[0][ACTION], POSE + [1.0] + POSE + [1.0])


def test_a_call_sends_every_frame_of_its_windows_with_the_station_observation(model, gateway):
    model(_request(frames=3), session_id='a')

    observation = gateway.calls[-1]['body']['observation']
    assert len(observation['image_window.top']) == 3
    assert observation['state.arm_l.joint_pos'] == POSE and observation['robot.arm'] == 'yam'
    assert gateway.calls[-1]['headers']['Authorization'] == 'Bearer dl_test_key'


def test_the_window_of_a_later_call_holds_only_the_frames_of_that_call(model, gateway):
    model(_request(frames=3), session_id='a')
    model(_request(frames=codecs.WINDOW_FRAMES), session_id='a')

    assert len(gateway.calls[-1]['body']['observation']['image_window.top']) == codecs.WINDOW_FRAMES


def test_the_first_call_of_a_session_sends_the_canary_in_a_session_of_its_own(model, gateway, tmp_path):
    model(_request(), session_id='a')
    model(_request(), session_id='a')
    model.end_session('a')

    canary, first, second = gateway.calls
    assert canary['body']['session_id'] is None and canary['body']['observation'][dream.PROMPT] != 'fold the towel'
    assert first['body']['session_id'] is None and second['body']['session_id'] is not None
    record = _record(tmp_path, 'a')
    assert len(record['canary']['hash']) == 16 and record['canary']['trace_id'].startswith('trace-')


def test_the_canary_hash_is_the_same_for_the_same_model(model, gateway, tmp_path):
    for session_id in ('a', 'b'):
        model(_request(), session_id=session_id)
        model.end_session(session_id)

    assert _record(tmp_path, 'a')['canary']['hash'] == _record(tmp_path, 'b')['canary']['hash']


def test_the_session_record_holds_the_trace_id_and_timing_of_each_call(model, gateway, tmp_path):
    model(_request(), session_id='a')
    model(_request(), session_id='a')
    model.end_session('a')

    calls = _record(tmp_path, 'a')['calls']
    # The first call is the canary's.
    assert [call['trace_id'] for call in calls] == [call['trace_id'] for call in gateway.calls[1:]]
    assert all(call['rows'] == ROWS and call['gateway_ms'] == 5 for call in calls)
    assert _record(tmp_path, 'a')['type'] == 'dream'


def test_a_retryable_refusal_is_sent_again(model, gateway):
    model(_request(), session_id='a')
    gateway.statuses.append({'code': 'DA-2000', 'reason': 'rate_limited', 'retry_after_s': 0.01})

    actions = model(_request(), session_id='a')

    assert len(actions) == ROWS and gateway.calls[-2]['body']['session_id'] is not None


def test_a_call_in_a_session_that_expired_is_sent_again_in_the_new_session(model, gateway, tmp_path):
    model(_request(), session_id='a')
    gateway.statuses.append({'code': 'DA-0001', 'reason': 'expired'})

    actions = model(_request(), session_id='a')
    model.end_session('a')

    expired, sent_again = gateway.calls[-2:]
    assert len(actions) == ROWS and sent_again['body']['session_id'] != expired['body']['session_id']
    assert _record(tmp_path, 'a')['expired'] == 1


def _close_idle_connections(monkeypatch, times: int) -> None:
    """The next ``times`` requests find their keep-alive connection closed, as TLS reports a gateway that closed it."""
    left = [times]
    request = http.client.HTTPConnection.request

    def closed(self, *args, **kwargs):
        if left[0] > 0:
            left[0] -= 1
            raise ssl.SSLEOFError(8, 'EOF occurred in violation of protocol')
        return request(self, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPConnection, 'request', closed)


def test_a_call_on_a_connection_the_gateway_closed_is_sent_again_on_a_new_one(model, gateway, monkeypatch):
    model(_request(), session_id='a')
    calls = len(gateway.calls)
    _close_idle_connections(monkeypatch, times=1)

    actions = model(_request(), session_id='a')

    assert len(actions) == ROWS and len(gateway.calls) == calls + 1
    assert gateway.calls[-1]['body']['session_id'] is not None


def test_a_call_whose_new_connection_also_fails_raises(model, gateway, monkeypatch):
    model(_request(), session_id='a')
    _close_idle_connections(monkeypatch, times=2)

    with pytest.raises(server.dream_client.DreamProtocolError):
        model(_request(), session_id='a')


def test_a_call_on_a_closed_connection_is_not_sent_again_when_a_whole_call_does_not_fit(model, gateway, monkeypatch):
    model(_request(), session_id='a')
    calls = len(gateway.calls)
    # The call timeout is 5 s, so no call sent now ends inside a 3 s deadline.
    model._retry_deadline_s = 3.0
    _close_idle_connections(monkeypatch, times=1)

    with pytest.raises(server.dream_client.DreamProtocolError):
        model(_request(), session_id='a')
    assert len(gateway.calls) == calls


def test_a_call_may_find_its_connection_closed_again_after_a_wait_that_dreamapi_asked_for(model, gateway, monkeypatch):
    model(_request(), session_id='a')
    calls = len(gateway.calls)
    gateway.statuses.append({'code': 'DA-2000', 'reason': 'rate_limited', 'retry_after_s': 0.01})
    _close_idle_connections(monkeypatch, times=1)
    sleep = server.time.sleep

    def wait_then_close(seconds):
        sleep(seconds)
        _close_idle_connections(monkeypatch, times=1)

    monkeypatch.setattr(server, 'time', types.SimpleNamespace(monotonic=time.monotonic, sleep=wait_then_close))

    actions = model(_request(), session_id='a')

    # The refusal and the answer reach the gateway; the two calls on closed connections do not.
    assert len(actions) == ROWS and len(gateway.calls) == calls + 2


def test_a_call_after_an_idle_gap_starts_a_new_session(model, gateway, tmp_path, monkeypatch):
    model(_request(), session_id='a')
    clock = server.time.monotonic() + 61.0
    monkeypatch.setattr(server.time, 'monotonic', lambda: clock)

    model(_request(), session_id='a')
    model.end_session('a')

    first, second = gateway.calls[-2:]
    assert first['body']['session_id'] is None and second['body']['session_id'] is None
    assert _record(tmp_path, 'a')['renewed'] == 1


def test_the_session_keeps_each_chunk_as_answered_and_as_played(gateway, model, tmp_path):
    def far(observation):
        return {name: (np.asarray(rows) + [1.5, 0, 0, 0, 0, 0, 0]).tolist() for name, rows in hold(observation).items()}

    gateway.chunk = far
    model(_request(), session_id='a')
    model.end_session('a')

    chunks = np.load(tmp_path / 'a.npz')
    assert chunks['answered'].shape == chunks['played'].shape == (1, ROWS, 14)
    assert chunks['answered'][0, 0, 0] == pytest.approx(1.5) and chunks['played'][0, 0, 0] < 1.0


def test_a_refusal_that_is_not_retryable_raises(model, gateway):
    model(_request(), session_id='a')
    gateway.statuses.append({'code': 'DA-4000', 'reason': 'malformed', 'detail': 'bad'})

    with pytest.raises(server.dream_client.DreamServerError):
        model(_request(), session_id='a')


def test_a_retryable_refusal_that_asks_to_wait_past_the_deadline_raises_at_once(model, gateway):
    model(_request(), session_id='a')
    gateway.statuses.append({'code': 'DA-2000', 'reason': 'rate_limited', 'retry_after_s': 60.0})
    calls = len(gateway.calls)

    with pytest.raises(server.dream_client.DreamServerError):
        model(_request(), session_id='a')
    assert len(gateway.calls) == calls + 1


def test_a_call_that_timed_out_goes_again_only_while_a_whole_call_fits_before_the_deadline(model, monkeypatch):
    model(_request(), session_id='a')
    clock = [0.0]
    monkeypatch.setattr(server, 'time', types.SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    model._retry_deadline_s = 12.0
    calls = []

    def timed_out():
        calls.append(clock[0])
        clock[0] += 5.0
        raise server.dream_client.DreamTimeout('no answer')

    monkeypatch.setattr(model._sessions['a'].client, 'get_action', timed_out)

    with pytest.raises(server.dream_client.DreamTimeout):
        model(_request(), session_id='a')
    # The call timeout is 5 s: a third call, sent at 10 s, would end past the deadline at 12 s.
    assert calls == [0.0, 5.0]


def test_a_chunk_past_the_limits_is_limited(gateway, model):
    def far(observation):
        chunks = hold(observation)
        return {name: (np.asarray(rows) + [1.5, 0, 0, 0, 0, 0, -1.0]).tolist() for name, rows in chunks.items()}

    gateway.chunk = far

    actions = np.array([action[ACTION] for action in model(_request(), session_id='a')])

    # The ramp holds each joint to 0.5 rad/s for the first 2 s, and the gripper closes by 0.05 per row.
    np.testing.assert_allclose(actions[0, [0, 6]], [0.5 / codecs.FPS, 0.95])
    assert actions[:, 0].max() < 1.5


def test_every_face_is_blurred_before_a_frame_leaves_the_server(model, gateway):
    face = cv2.cvtColor(_self_check_frame(), cv2.COLOR_BGR2RGB)

    model(_request(frames=1, view=face), session_id='a')

    sent = gateway.calls[-1]['body']['observation']['image_window.top'][0]
    received = cv2.imdecode(np.frombuffer(base64.b64decode(sent), np.uint8), cv2.IMREAD_COLOR)
    original = _self_check_frame()
    # The eyes at (168, 170) and (232, 170) are dark pupils in the original, and blurred in what was sent.
    assert np.abs(received.astype(int) - original.astype(int))[150:190, 150:250].mean() > 20


def test_warm_up_asks_again_until_the_model_answers(gateway):
    gateway.statuses.extend([{'code': 'DA-2000', 'reason': 'pool_warming', 'retry_after_s': 0.01}] * 2)

    server._wait_until_warm(CONFIG, deadline_s=30.0)

    assert len(gateway.calls) == 3


def test_warm_up_raises_when_the_model_asks_to_wait_past_the_deadline(gateway):
    gateway.statuses.append({'code': 'DA-2000', 'reason': 'pool_warming', 'retry_after_s': 60.0})

    with pytest.raises(server.dream_client.DreamServerError):
        server._wait_until_warm(CONFIG, deadline_s=30.0)
    assert len(gateway.calls) == 1


def test_warm_up_asks_again_on_a_new_connection_when_the_gateway_closed_the_idle_one(gateway, monkeypatch):
    gateway.statuses.append({'code': 'DA-2000', 'reason': 'pool_warming', 'retry_after_s': 0.01})
    sleep = server.time.sleep

    def wait_then_close(seconds):
        sleep(seconds)
        _close_idle_connections(monkeypatch, times=1)

    monkeypatch.setattr(server, 'time', types.SimpleNamespace(monotonic=time.monotonic, sleep=wait_then_close))

    server._wait_until_warm(CONFIG, deadline_s=30.0)

    assert len(gateway.calls) == 2


def test_warm_up_asks_again_in_the_new_session_when_its_session_expired(gateway):
    gateway.statuses.extend([
        {'code': 'DA-2000', 'reason': 'pool_warming', 'retry_after_s': 0.01},
        {'code': 'DA-0001', 'reason': 'expired'},
    ])

    server._wait_until_warm(CONFIG, deadline_s=30.0)

    assert len(gateway.calls) == 3
    assert gateway.calls[-1]['body']['session_id'] != gateway.calls[-2]['body']['session_id']


def test_warm_up_asks_again_after_a_call_that_timed_out(gateway, monkeypatch):
    get_action = server.dream_client.DreamClient.get_action
    timed_out = [False]

    def first_times_out(self):
        if not timed_out[0]:
            timed_out[0] = True
            raise server.dream_client.DreamTimeout('no answer')
        return get_action(self)

    monkeypatch.setattr(server.dream_client.DreamClient, 'get_action', first_times_out)

    server._wait_until_warm(CONFIG, deadline_s=30.0)

    assert len(gateway.calls) == 1


def test_a_sessions_dir_under_the_home_directory_is_expanded(gateway, tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)

    built = server.dream_model.override(sessions_dir='~/records', timeout_ms=5000).instantiate()
    built.close()

    assert (tmp_path / 'records').is_dir() and not (tmp_path / '~').exists()


def test_the_pipeline_stacks_nine_frames_of_each_camera_and_plays_whole_chunks():
    deployment = server.pipeline.instantiate()

    spec = deployment.local.to_spec()
    layers = {layer['name']: layer['args'] for layer in spec['seq']}
    assert layers['temporal_stack']['keys'] == [keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE]
    np.testing.assert_allclose(layers['temporal_stack']['offsets_sec'], np.arange(-1.6, 0.01, 0.2), atol=1e-9)
    assert layers['trajectory_player']['replan_on_error'] is True
    # The limiter's ramp counts every row of a chunk as played.
    assert layers['inference_schedule']['merge']['horizon_sec'] is None
