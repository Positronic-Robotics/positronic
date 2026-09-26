from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pytest
from positronic_wire import websocket, wire

from positronic import keys
from positronic.cfg.embodiment import droid_3cam_fake, droid_fake
from positronic.cfg.policy import bearer_headers
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.offboard import protocol
from positronic.offboard.client import RECV_MS, SEND_MS, InferenceClient
from positronic.offboard.server import AUTH_TOKEN_ENV
from positronic.offboard.serving_cost import InstantChunk, against_server, capture, observations, replay, rig_stack
from positronic.offboard.spec import PolicyDeployment
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.sequential import Sequential

CAMERAS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)

# A stack shaped like the one a video-conditioned server declares: four strided frames per camera.
DECLARED_OFFSETS_SEC = (-23 / 15, -16 / 15, -8 / 15, 0.0)
DECLARED_WIDTH, DECLARED_HEIGHT = 320, 176


def _declared_deployment() -> PolicyDeployment:
    stack = Sequential(
        PauseOnUnavailable(),
        TemporalStack(keys=(*CAMERAS, keys.EE_POSE, keys.GRIP), offsets_sec=DECLARED_OFFSETS_SEC),
        ChunkedSchedule(fps=15.0),
        RestrictImageSize(width=DECLARED_WIDTH, height=DECLARED_HEIGHT),
    )
    return PolicyDeployment(stack, compress_images=True)


def _ticks(count: int, period_ns: int = 66_666_666, size: tuple[int, int] = (48, 64)):
    """A stand-in for the harness: one moving frame per camera per control tick."""
    rng = np.random.default_rng(0)
    for tick in range(count):
        frame = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        yield (
            1_000_000_000 + tick * period_ns,
            {keys.EE_POSE: np.zeros(7), keys.GRIP: 0.0, keys.ROBOT_STATUS: 0, **dict.fromkeys(CAMERAS, frame)},
        )


def _model():
    return partial(InstantChunk(rows=2), session_id='capture')


def test_replay_divides_a_round_trip_into_the_phases_the_server_reports(start_server):
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    payloads = capture(_ticks(12), stack, _model(), requests=2)
    assert payloads, 'the stack sent nothing'

    host, port, *_ = start_server(InstantChunk(rows=2), PolicyDeployment(stack, compress_images=True))
    session = InferenceClient(
        websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    ).new_session()
    try:
        rows = replay(session, payloads, compress_images=True)
    finally:
        session.close()

    assert len(rows) == len(payloads)
    for row in rows:
        assert row['wire_kib'] > 0
        assert row[protocol.TIMING_SERVED] >= row[protocol.TIMING_DECODE]
        assert row['round_trip_ms'] >= row[protocol.TIMING_SERVED]
        assert row[SEND_MS] >= 0.0 and row[RECV_MS] >= 0.0


def test_a_captured_payload_carries_one_stack_per_stacked_key():
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    payloads = capture(_ticks(12), stack, _model(), requests=1)

    sent = payloads[0]
    for camera in CAMERAS:
        assert sent[camera].shape == (3, 48, 64, 3)
    assert sent[keys.EE_POSE].shape == (3, 7)
    assert sent[keys.GRIP].shape == (3,)


def test_a_payload_over_the_server_limit_is_refused_before_it_is_sent():
    session = MagicMock()
    oversized = {'cam': np.zeros((1, 3000, 3000, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match='message limit'):
        replay(session, [oversized], compress_images=False)
    session.infer.assert_not_called()


def test_a_named_server_is_measured_through_the_stack_it_declares(start_server):
    served = start_server(InstantChunk(rows=2), _declared_deployment())

    with against_server('websocket', served.ws()[1]) as measured:
        assert measured.compress_images, 'the wire setting comes from the handshake too'
        payloads = capture(_ticks(40, size=(360, 640)), measured.stack, _model(), requests=2)
        rows = replay(measured.session, payloads, measured.compress_images)

    sent = payloads[0]
    for camera in CAMERAS:
        assert sent[camera].shape == (len(DECLARED_OFFSETS_SEC), DECLARED_HEIGHT, 312, 3)
    assert sent[keys.EE_POSE].shape == (len(DECLARED_OFFSETS_SEC), 7)
    assert len(rows) == len(payloads)
    for row in rows:
        assert row[protocol.TIMING_SERVED] >= 0.0
        assert row[SEND_MS] >= 0.0 and row[RECV_MS] >= 0.0


def test_an_episode_missing_a_key_the_stack_asks_for_says_which(start_server):
    """A key error from inside a layer names nothing a reader can act on; the capture names the key."""
    served = start_server(InstantChunk(rows=2), _declared_deployment())
    gripless = ((ts, {key: value for key, value in obs.items() if key != keys.GRIP}) for ts, obs in _ticks(12))

    with against_server('websocket', served.ws()[1]) as measured:
        with pytest.raises(ValueError, match="asks for 'grip'"):
            capture(gripless, measured.stack, _model(), requests=1)


def _droid_episode(path, cameras):
    """A DROID recording: its observation columns, the commands the rig emitted, and the statics it carries."""
    period_ns = int(1e9 / 15.0)
    with DiskEpisodeWriter(path) as writer:
        for tick in range(3):
            at = tick * period_ns
            writer.append(keys.JOINTS, np.zeros(7), at)
            writer.append(keys.JOINT_VEL, np.zeros(7), at)
            writer.append(keys.EE_POSE, np.zeros(7), at)
            writer.append(keys.ROBOT_STATUS, 0, at)
            writer.append(keys.GRIP, 0.0, at)
            for camera in cameras:
                writer.append(camera, np.zeros((48, 64, 3), np.uint8), at)
            writer.append(keys.TARGET_EE_POSE, np.zeros(7), at)
            writer.append(keys.TARGET_GRIP, 0.0, at)
        writer.set_static(keys.TASK, 'pick the spoon')
        for static in (roboarm_keys.URDF, roboarm_keys.JOINT_NAMES, roboarm_keys.CONTROL_FRAME, roboarm_keys.GRIPPER):
            writer.set_static(static, 'recorded')
        writer.set_static('meshes', 'recorded')
    return DiskEpisode(path)


def _sent(episode, embodiment):
    built = embodiment.instantiate()
    cameras = [name for name in built.observations if name.startswith(keys.IMAGE_PREFIX)]
    stack = rig_stack(cameras, frames=2, rate_hz=15.0, width=64, height=48)
    return capture(observations(episode, built, rate_hz=15.0), stack, _model(), requests=1)[0]


def test_the_wire_carries_what_the_embodiment_observes_and_nothing_else_the_episode_records(tmp_path):
    """A rig sends its declared observations, the task and its descriptor; statics and commands are not sent."""
    episode = _droid_episode(tmp_path / 'episode', (*CAMERAS, keys.EXTERIOR_IMAGE_2))

    sent = _sent(episode, droid_fake)

    assert set(sent) == {
        keys.JOINTS,
        keys.JOINT_VEL,
        keys.EE_POSE,
        keys.ROBOT_STATUS,
        keys.GRIP,
        *CAMERAS,
        keys.TASK,
        keys.DESCRIPTOR,
    }
    assert sent[keys.TASK] == 'pick the spoon'


def test_a_channel_the_embodiment_declares_beyond_two_cameras_reaches_the_stack(tmp_path):
    episode = _droid_episode(tmp_path / 'episode', (*CAMERAS, keys.EXTERIOR_IMAGE_2))

    sent = _sent(episode, droid_3cam_fake)

    assert sent[keys.EXTERIOR_IMAGE_2].shape == (2, 48, 64, 3)


def test_a_channel_the_episode_does_not_record_is_named(tmp_path):
    episode = _droid_episode(tmp_path / 'episode', CAMERAS)

    with pytest.raises(ValueError, match=keys.EXTERIOR_IMAGE_2):
        _sent(episode, droid_3cam_fake)


def test_an_ambient_token_does_not_reach_a_server_the_run_never_named(start_server, monkeypatch):
    """A session opened without `headers` sends no token, even when `AUTH_TOKEN` is set."""
    token = 'a-token-for-another-endpoint'
    monkeypatch.setenv(AUTH_TOKEN_ENV, token)
    served = start_server(InstantChunk(rows=2), _declared_deployment(), auth_token=token)

    with pytest.raises(wire.ConnectRefused), against_server('websocket', served.ws()[1]):
        pass

    with against_server('websocket', served.ws()[1], bearer_headers.instantiate()) as measured:
        assert measured.stack is not None, 'the offered credential opened the session'


def test_a_server_on_a_socket_is_measured_through_the_wire_that_dials_it(start_server, socket_path):
    """A socket wire dials a socket address, so the probe takes the address the chosen wire names."""
    served = start_server(InstantChunk(rows=2), _declared_deployment(), uds=socket_path)

    with against_server('websocket_unix', served.unix()[1]) as measured:
        payloads = capture(_ticks(12), measured.stack, _model(), requests=1)
        rows = replay(measured.session, payloads, measured.compress_images)

    assert len(rows) == len(payloads) == 1
