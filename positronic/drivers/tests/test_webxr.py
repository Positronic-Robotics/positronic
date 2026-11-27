import asyncio
import collections
import contextlib
import itertools
import json
import logging
import socket
import time
import types
from collections.abc import AsyncIterator, Iterator
from typing import cast

import av
import httpx
import numpy as np
import psutil
import pytest
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from websockets.asyncio.client import connect as ws_connect
from websockets.sync import client as ws_client

import pimm
from positronic import keys, utils
from positronic.cfg import webxr as cfg_webxr
from positronic.drivers import webrtc, webxr
from positronic.drivers.webxr import Frontend, SessionMode, WebXR
from positronic.geom import Rotation, Transform3D
from positronic.tests.testing_coutils import RecordingEmitter
from positronic.tests.testing_inference import free_port


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


def test_the_headset_video_skips_a_lost_camera_and_resumes_on_its_next_frame(world):
    webxr = WebXR(port=0)
    emitter, receiver = world.local_pipe()
    webxr.frame._bind(receiver)

    emitter.emit('frame 1')
    first = webxr._next_frame(None)
    assert first is not None and first.data == 'frame 1'

    emitter.emit(pimm.SignalError('camera lost'))
    assert webxr._next_frame(first.time) is None

    emitter.emit('frame 2')
    second = webxr._next_frame(first.time)
    assert second is not None and second.data == 'frame 2'


@contextlib.contextmanager
def _served_webxr(frontend: Frontend = Frontend.HEADSET, webxr: WebXR | None = None) -> Iterator[str]:
    """A ``WebXR`` server of ``frontend`` on a free port, or ``webxr`` when given, and its base URL once it answers."""
    if webxr is None:
        webxr = WebXR(port=free_port(), frontend=frontend, use_https=False)
    port = webxr.port
    stop = types.SimpleNamespace(value=False)
    run = webxr.run(cast(pimm.SignalReceiver, stop), cast(pimm.Clock, None))
    next(run)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get(base + '/core.js', timeout=1.0)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline, 'the headset page never answered'
                time.sleep(0.05)
        yield base
    finally:
        stop.value = True
        list(run)


@pytest.mark.parametrize(
    'route',
    [
        '/',
        '/core.js',
        '/three.min.js',
        '/webxr-button.js',
        '/video-player.js',
        '/webrtc-video.js',
        '/quest-trigger.js',
        '/press-count.js',
    ],
)
def test_the_headset_page_is_served_from_any_working_directory(tmp_path, monkeypatch, route):
    monkeypatch.chdir(tmp_path)
    with _served_webxr() as base:
        assert httpx.get(base + route, timeout=1.0).status_code == 200


def test_the_headset_server_logs_through_the_process_handlers():
    root = logging.getLogger()
    handlers = list(root.handlers)
    with _served_webxr():
        assert root.handlers == handlers


def test_the_phone_page_is_served_at_the_root():
    with _served_webxr(Frontend.PHONE) as base:
        assert httpx.get(base + '/', timeout=1.0).status_code == 200


@pytest.mark.parametrize(('preset', 'session'), [(cfg_webxr.oculus, 'immersive-ar'), (cfg_webxr.psvr2, 'immersive-vr')])
def test_the_headset_page_serves_the_session_mode_of_its_preset(preset, session):
    served = preset.override(port=free_port(), use_https=False).instantiate()
    with _served_webxr(webxr=served) as base:
        assert httpx.get(base + '/', timeout=1.0).status_code == 200
        script = httpx.get(base + '/session-mode.js', timeout=1.0)
        assert script.headers['content-type'].startswith('text/javascript')
        assert script.text == f'export const SESSION_MODE = "{session}";\n'


@pytest.mark.parametrize('frontend', list(Frontend))
def test_the_page_drops_a_query(frontend):
    with _served_webxr(frontend) as base:
        stale = httpx.get(f'{base}/?mode=teleop', timeout=1.0)
        assert stale.status_code == 307 and stale.headers['location'] == '/'


def test_a_vr_session_on_the_phone_page_is_refused():
    with pytest.raises(ValueError, match='session_mode VR needs the headset frontend'):
        WebXR(port=free_port(), frontend=Frontend.PHONE, session_mode=SessionMode.VR)


def test_the_banner_lists_each_ipv4_address_that_is_not_a_loopback(monkeypatch):
    address = collections.namedtuple('address', 'family address')
    interfaces = {
        'lo0': [address(socket.AF_INET, '127.0.0.1')],
        'en0': [address(socket.AF_INET6, 'fe80::1'), address(socket.AF_INET, '10.0.0.5')],
        'utun3': [address(socket.AF_INET, '100.64.0.2')],
    }
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: interfaces)
    assert utils.interface_ipv4_addresses() == [('en0', '10.0.0.5'), ('utun3', '100.64.0.2')]


@pytest.mark.parametrize('fps', [0, -1, 30.0])
def test_a_video_rate_that_is_not_a_positive_integer_is_refused(fps):
    with pytest.raises(ValueError, match='video_fps must be a positive integer'):
        WebXR(port=free_port(), video_fps=fps)


def test_the_page_state_is_empty_until_something_sends_one(world):
    served = WebXR(port=free_port(), frontend=Frontend.PHONE, use_https=False)
    emitter, receiver = world.local_pipe()
    served.page_state._bind(receiver)
    with _served_webxr(webxr=served) as base:
        assert httpx.get(base + '/state', timeout=1.0).json() == {}
        emitter.emit({'phase': 'running', 'control': True})
        assert httpx.get(base + '/state', timeout=1.0).json() == {'phase': 'running', 'control': True}


def test_the_page_input_emits_each_object_that_the_page_posts(world):
    """The teleoperator chooses on the page, such as the camera of the headset. A page on another site cannot."""
    served = WebXR(port=free_port(), use_https=False)
    inputs = RecordingEmitter()
    served.page_input._bind(inputs, clock=world.clock)
    with _served_webxr(webxr=served) as base:
        assert httpx.post(base + '/input', json={'headset_camera': keys.WRIST_IMAGE}, timeout=1.0).status_code == 200
        assert httpx.post(base + '/input', json=['not', 'an', 'object'], timeout=1.0).status_code == 422
        assert httpx.post(base + '/input', content=b'not json', timeout=1.0).status_code == 422
        foreign = {'origin': 'https://elsewhere.example'}
        assert httpx.post(base + '/input', json={}, headers=foreign, timeout=1.0).status_code == 403
        other_scheme = {'origin': base.replace('http://', 'https://')}
        assert httpx.post(base + '/input', json={}, headers=other_scheme, timeout=1.0).status_code == 403
        assert httpx.post(base + '/input', json={'x': 1}, headers={'origin': base}, timeout=1.0).status_code == 200
    assert [data for _, data in inputs.emitted] == [{'headset_camera': keys.WRIST_IMAGE}, {'x': 1}]


def test_a_controller_message_parses_back_to_its_poses_and_buttons():
    pose = Transform3D(np.array([0.1, 1.0, -0.3]), Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
    sent = webxr.ControllerButtons((0.5, 0.0, 1.0, 0.0, 1.0, 0.0), (0, 0, 3, 0, 7, 1))
    message = webxr.controller_message({webxr.RIGHT: pose}, {webxr.RIGHT: sent})
    positions, buttons = webxr.parse_controller_message(json.loads(json.dumps(message)))
    assert positions[webxr.LEFT] is None and buttons[webxr.LEFT] is None
    right = positions[webxr.RIGHT]
    assert right is not None
    np.testing.assert_allclose(right.translation, pose.translation)
    np.testing.assert_allclose(right.rotation.as_quat, pose.rotation.as_quat)
    assert buttons[webxr.RIGHT] == sent


@pytest.mark.parametrize(
    ('presses', 'match'),
    [((0,) * 5, 'a press count for each button value'), ((0, 0, -1, 0, 0, 0), 'an integer of at least 0')],
    ids=['too few counts', 'a negative count'],
)
def test_controller_buttons_refuse_press_counts_that_do_not_match_the_values(presses, match):
    with pytest.raises(ValueError, match=match):
        webxr.ControllerButtons((0.0,) * 6, presses)


RELEASED = webxr.ControllerButtons((0.0,) * 6, (0,) * 6)


def _hand_message(values: list[float]) -> dict:
    """A page message of a right hand that the headset tracks with no controller in it, with ``values``."""
    message = webxr.controller_message({webxr.RIGHT: Transform3D()}, {webxr.RIGHT: RELEASED})
    message['controllers'][webxr.RIGHT]['buttons'] = values
    return message


@pytest.mark.parametrize('values', [[], [0.0, 1.0]], ids=['no values', 'too few values'])
def test_a_hand_with_too_few_button_values_parses_as_a_controller_that_is_not_tracked(values):
    positions, buttons = webxr.parse_controller_message(json.loads(json.dumps(_hand_message(values))))
    assert positions[webxr.RIGHT] is None and buttons[webxr.RIGHT] is None


def test_the_controller_stream_emits_a_sample_in_which_no_controller_is_tracked(world):
    served = WebXR(port=free_port(), use_https=False)
    positions, buttons = RecordingEmitter(), RecordingEmitter()
    served.controller_positions._bind(positions, clock=world.clock)
    served.buttons._bind(buttons, clock=world.clock)
    controller = webxr.controller_message({webxr.RIGHT: Transform3D()}, {webxr.RIGHT: RELEASED})
    hand = _hand_message([])
    with (
        _served_webxr(webxr=served) as base,
        ws_client.connect(base.replace('http', 'ws') + webxr.CONTROLLER_PATH) as ws,
    ):
        ws.send(json.dumps(controller))
        ws.send(json.dumps(hand))
        deadline = time.monotonic() + 5
        while len(positions.emitted) < 2 or len(buttons.emitted) < 2:
            assert time.monotonic() < deadline, 'the stream emitted no sample of the hand'
            time.sleep(0.01)
    assert positions.emitted[0][1][webxr.RIGHT] is not None
    assert positions.emitted[1][1] == {webxr.LEFT: None, webxr.RIGHT: None}
    assert buttons.emitted[1][1] == {webxr.LEFT: None, webxr.RIGHT: None}


def _presses_of_a(world, connections: list[list[tuple[int, float]]]) -> list[int]:
    """The presses of A that the controller stream emits for ``connections``, one after the other. Each message of a
    connection is the count of A presses on the page and how many milliseconds before it reaches the stream the page
    sent it. The page clock is far from the clock of the stream."""
    served = WebXR(port=free_port(), use_https=False)
    buttons = RecordingEmitter()
    served.controller_positions._bind(RecordingEmitter(), clock=world.clock)
    served.buttons._bind(buttons, clock=world.clock)
    page_clock_ms = -1e9

    def message(presses_of_a: int, delay_ms: float) -> str:
        state = webxr.ControllerButtons((0.0,) * 6, (0, 0, 0, 0, presses_of_a, 0))
        sent_ms = time.monotonic() * 1000 + page_clock_ms - delay_ms
        return json.dumps(webxr.controller_message({webxr.RIGHT: Transform3D()}, {webxr.RIGHT: state}, sent_ms))

    with _served_webxr(webxr=served) as base:
        for messages in connections:
            with ws_client.connect(base.replace('http', 'ws') + webxr.CONTROLLER_PATH) as ws:
                sent = len(buttons.emitted) + len(messages)
                for presses_of_a, delay_ms in messages:
                    ws.send(message(presses_of_a, delay_ms))
                deadline = time.monotonic() + 5
                while len(buttons.emitted) < sent:
                    assert time.monotonic() < deadline, 'the stream emitted no sample'
                    time.sleep(0.01)
    return [sample[webxr.RIGHT].presses[4] for _, sample in buttons.emitted]


def test_a_press_that_the_page_counted_while_its_socket_was_closed_does_not_count(world):
    # The page counts 3 presses of A while it is not connected
    assert _presses_of_a(world, [[(1, 0), (2, 0)], [(5, 0), (6, 0)]]) == [0, 1, 0, 1]


def test_the_presses_of_a_backlog_after_a_stall_of_the_network_do_not_count(world):
    # The page sends 3 presses during a stall of 3 s, and they all come at once when the stall ends
    backlog = [(2, 3000), (3, 2000), (4, 1000)]
    assert _presses_of_a(world, [[(0, 0), (1, 0), *backlog, (5, 0)]]) == [0, 1, 0, 0, 0, 1]


def test_a_press_right_after_a_slow_stretch_that_is_not_stale_counts(world):
    # Each message comes later than the one before, but never more than the stale limit of 500 ms later than the first
    assert _presses_of_a(world, [[(0, 0), (0, 200), (0, 400), (1, 400)]]) == [0, 0, 0, 1]


def test_a_press_at_the_end_of_a_long_connection_counts_while_the_page_clock_runs_slow():
    # The page sends once each second for 3 hours, and its clock runs 100 ppm slow: the delay grows by 1080 ms
    presses = webxr._ConnectionPresses(stale_limit=0.5)
    received = np.arange(3 * 3600 + 1) * 1000.0
    sent = received * (1 - 100e-6) - 1e9
    counts = [0] * (len(received) - 1) + [1]
    seen = []
    for count, sent_ms, received_ms in zip(counts, sent, received, strict=True):
        state = webxr.ControllerButtons((0.0,) * 6, (0, 0, 0, 0, count, 0))
        right = presses.counted({webxr.RIGHT: state}, sent_ms, received_ms)[webxr.RIGHT]
        assert right is not None
        seen.append(right.presses[4])
    assert seen == counts


@pytest.mark.parametrize('limit', [0.0, -0.5, float('inf'), float('nan')])
def test_a_stale_limit_that_is_not_finite_and_positive_is_refused(limit):
    with pytest.raises(ValueError, match='stale_limit must be finite and positive'):
        WebXR(port=free_port(), stale_limit=limit)


def test_a_controller_message_refuses_a_controller_with_a_pose_and_no_buttons():
    with pytest.raises(ValueError, match='positions and buttons must name the same controllers'):
        webxr.controller_message({webxr.RIGHT: Transform3D()}, {})


def test_a_controller_message_refuses_a_controller_with_fewer_button_values_than_a_controller_has():
    with pytest.raises(ValueError, match='at least 6 button values'):
        webxr.controller_message(
            {webxr.RIGHT: Transform3D()}, {webxr.RIGHT: webxr.ControllerButtons((0.0,) * 5, (0,) * 5)}
        )


@contextlib.asynccontextmanager
async def _page_peer(base: str, offer_id: int = 1) -> AsyncIterator[tuple[MediaStreamTrack, RTCDataChannel]]:
    """A page on ``/ws`` of ``base`` whose peer connection receives the headset video: its video track and its timing
    channel, once the service answers its offer ``offer_id``."""
    async with ws_connect(base.replace('http', 'ws') + webxr.CONTROLLER_PATH) as page:
        connection = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        connection.addTransceiver('video', direction='recvonly')
        timing = connection.createDataChannel(webrtc.TIMING_CHANNEL)
        tracks: asyncio.Queue[MediaStreamTrack] = asyncio.Queue()
        connection.on('track', tracks.put_nowait)
        try:
            await connection.setLocalDescription(await connection.createOffer())
            offer = webrtc.Offer(id=offer_id, sdp=connection.localDescription.sdp)
            await page.send(json.dumps({webxr._SIGNAL: offer.model_dump()}))
            answer = webrtc.Answer.model_validate(json.loads(await asyncio.wait_for(page.recv(), 10))[webxr._SIGNAL])
            assert answer.id == offer_id
            await connection.setRemoteDescription(RTCSessionDescription(answer.sdp, 'answer'))
            yield await asyncio.wait_for(tracks.get(), 10), timing
        finally:
            await connection.close()


async def _emit_frames(emitter: pimm.SignalEmitter, colors: Iterator[int]) -> None:
    """Emit a frame of each of ``colors``, one each 20 ms."""
    for color in colors:
        emitter.emit(np.full((120, 160, 3), color, np.uint8))
        await asyncio.sleep(0.02)


@contextlib.asynccontextmanager
async def _frames_flowing(emitter: pimm.SignalEmitter) -> AsyncIterator[None]:
    """New frames on ``emitter`` until the block ends, as the view of the service gives them."""
    frames = asyncio.ensure_future(_emit_frames(emitter, itertools.cycle([40, 200])))
    try:
        yield
    finally:
        frames.cancel()


async def _received_frame(track: MediaStreamTrack) -> tuple[av.VideoFrame, int]:
    """The next frame of ``track`` and its RTP time from the first frame."""
    frame = await asyncio.wait_for(track.recv(), 10)
    assert isinstance(frame, av.VideoFrame) and frame.pts is not None
    return frame, frame.pts


async def _open(channel: RTCDataChannel) -> None:
    deadline = time.monotonic() + 10
    while channel.readyState != 'open':
        assert time.monotonic() < deadline, 'the data channel never opened'
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_webrtc_peer_of_the_page_receives_the_frames_of_the_headset_video_as_a_video_track(world):
    served = WebXR(port=free_port(), use_https=False)
    emitter, receiver = world.local_pipe()
    served.frame._bind(receiver)
    with _served_webxr(webxr=served) as base:
        async with _page_peer(base) as (track, _timing), _frames_flowing(emitter):
            received = [(await _received_frame(track))[0] for _ in range(5)]
    images = [frame.to_ndarray(format='rgb24') for frame in received]
    assert {image.shape for image in images} == {(120, 160, 3)}
    assert {round(float(image.mean()) / 80) for image in images} <= {0, 2}, 'each frame is one of the emitted frames'


@pytest.mark.asyncio
async def test_the_timing_channel_gives_the_rtp_time_of_each_frame_less_one_offset_and_answers_a_ping(world):
    served = WebXR(port=free_port(), use_https=False)
    emitter, receiver = world.local_pipe()
    served.frame._bind(receiver)
    with _served_webxr(webxr=served) as base:
        async with _page_peer(base) as (track, timing):
            messages: asyncio.Queue[str] = asyncio.Queue()
            timing.on('message', messages.put_nowait)
            await _open(timing)
            timing.send(webrtc.Ping(ping=12.5).model_dump_json())
            async with _frames_flowing(emitter):
                rtp_times = [(await _received_frame(track))[1] for _ in range(5)]
                await asyncio.sleep(0.1)
    parsed = [json.loads(messages.get_nowait()) for _ in range(messages.qsize())]
    pongs = [webrtc.Pong.model_validate(message) for message in parsed if 'pong' in message]
    assert [pong.pong for pong in pongs] == [12.5] and pongs[0].service_ms > 0
    sent = [webrtc.FrameSent.model_validate(message).frame for message in parsed if 'frame' in message]
    # The page finds the offset of the RTP timestamps: the one difference that each shown frame has with a sent time
    # The page finds the offset of the RTP time from the one difference that each shown frame has with a sent time.
    # The decoder of aiortc rounds the RTP time of a frame by up to one tick.
    first, *others = rtp_times
    offsets = [
        first - pts
        for pts in sent
        if all(any(abs(rtp - other - (first - pts)) <= 1 for other in sent) for rtp in others)
    ]
    assert len(offsets) == 1


@pytest.mark.asyncio
async def test_the_service_logs_each_report_of_the_page_on_its_timing_channel(world, caplog):
    caplog.set_level(logging.INFO, logger=webrtc.__name__)
    served = WebXR(port=free_port(), use_https=False)
    emitter, receiver = world.local_pipe()
    served.frame._bind(receiver)
    emitter.emit(np.zeros((120, 160, 3), np.uint8))
    report = webrtc.VideoReport(
        seconds=5.0,
        frames=150,
        ages_ms=[30.0, 40.0, 50.0],
        stalls_ms=[300.0],
        packets_lost=2,
        nacks=2,
        plis=0,
        rtt_ms=4.0,
        codec='video/H264',
    )
    with _served_webxr(webxr=served) as base:
        async with _page_peer(base) as (_track, timing):
            await _open(timing)
            timing.send(json.dumps({'report': report.model_dump()}))
            deadline = time.monotonic() + 10
            while not any('Video over WebRTC' in line for line in caplog.messages):
                assert time.monotonic() < deadline, 'the service never logged the report'
                await asyncio.sleep(0.01)
    assert (
        'Video over WebRTC: 150 frames in 5.0 s, age 40/49/50 ms (median/95th/max), 1 stalls (300 ms); since '
        'the connect 2 packets lost, 2 NACK, 0 PLI; RTT 4 ms; video/H264'
    ) in caplog.messages


@pytest.mark.asyncio
async def test_a_bad_setup_message_logs_an_error_and_the_next_offer_still_gets_the_video(world, caplog):
    served = WebXR(port=free_port(), use_https=False)
    emitter, receiver = world.local_pipe()
    served.frame._bind(receiver)
    with _served_webxr(webxr=served) as base:
        async with ws_connect(base.replace('http', 'ws') + webxr.CONTROLLER_PATH) as page:
            await page.send(json.dumps({webxr._SIGNAL: {'type': 'offer', 'id': 1, 'sdp': 'not an offer'}}))
            deadline = time.monotonic() + 10
            while not any(record.levelno == logging.ERROR for record in caplog.records):
                assert time.monotonic() < deadline, 'the service never logged the bad offer'
                await asyncio.sleep(0.01)
        async with _page_peer(base, offer_id=2) as (track, _timing), _frames_flowing(emitter):
            frame, _ = await _received_frame(track)
    assert frame.width == 160
    assert 'The video peer cannot use a setup message of the page' in caplog.text


@pytest.mark.asyncio
async def test_the_video_connection_closes_with_the_socket_of_the_page(world, caplog):
    caplog.set_level(logging.INFO, logger=webrtc.__name__)
    served = WebXR(port=free_port(), use_https=False)
    emitter, receiver = world.local_pipe()
    served.frame._bind(receiver)
    with _served_webxr(webxr=served) as base:
        async with _page_peer(base) as (track, _timing), _frames_flowing(emitter):
            await _received_frame(track)
        deadline = time.monotonic() + 10
        while 'The video connection 1 of the page is closed' not in caplog.messages:
            assert time.monotonic() < deadline, 'the service never closed the video connection'
            await asyncio.sleep(0.01)
