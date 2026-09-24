from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pytest
from positronic_wire import websocket, wire

from positronic import keys
from positronic.cfg.policy import bearer_headers
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.offboard import protocol
from positronic.offboard.client import RECV_MS, SEND_MS, InferenceClient
from positronic.offboard.server import AUTH_TOKEN_ENV
from positronic.offboard.serving_cost import (
    InstantChunk,
    InstantSource,
    against_server,
    capture,
    observations,
    replay,
    rig_stack,
)
from positronic.offboard.spec import PolicyDeployment
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.sequential import Sequential

CAMERAS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)

# A stack shaped like the one a video-conditioned server declares: four strided frames per camera,
# bounded well under what this probe's own flags default to.
DECLARED_OFFSETS_SEC = (-23 / 15, -16 / 15, -8 / 15, 0.0)
DECLARED_WIDTH, DECLARED_HEIGHT = 320, 176


def _declared_deployment() -> PolicyDeployment:
    stack = Sequential(
        PauseOnUnavailable(),
        TemporalStack(keys=(*CAMERAS, keys.EE_POSE, keys.GRIP), offsets_sec=DECLARED_OFFSETS_SEC),
        ChunkedSchedule(fps=15.0),
        RestrictImageSize(width=DECLARED_WIDTH, height=DECLARED_HEIGHT),
    )
    return PolicyDeployment(InstantSource(2), stack, compress_images=True)


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

    host, port, *_ = start_server(PolicyDeployment(InstantSource(2), stack, compress_images=True))
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
    """A raw stack the server would close the socket on stops the probe with the flags that shrink it."""
    session = MagicMock()
    oversized = {'cam': np.zeros((1, 3000, 3000, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match='message limit'):
        replay(session, [oversized], compress_images=False)
    session.infer.assert_not_called()


def test_a_named_server_is_measured_through_the_stack_it_declares(start_server):
    """The probe sends what the server's handshake declares, not what its own flags would build."""
    served = start_server(_declared_deployment())

    with against_server('websocket', served.host, served.port, '', '') as measured:
        assert measured.compress_images, 'the wire setting comes from the handshake too'
        payloads = capture(_ticks(40, size=(360, 640)), measured.stack, _model(), requests=2)
        rows = replay(measured.session, payloads, measured.compress_images)

    sent = payloads[0]
    for camera in CAMERAS:
        # Four frames, not the 25 the flags default to; 176 high, not the 288 their bound would give.
        assert sent[camera].shape == (len(DECLARED_OFFSETS_SEC), DECLARED_HEIGHT, 312, 3)
    assert sent[keys.EE_POSE].shape == (len(DECLARED_OFFSETS_SEC), 7)
    assert len(rows) == len(payloads)
    for row in rows:
        assert row[protocol.TIMING_SERVED] >= 0.0
        assert row[SEND_MS] >= 0.0 and row[RECV_MS] >= 0.0


def test_an_episode_missing_a_key_the_stack_asks_for_says_which(start_server):
    """A key error from inside a layer names nothing a reader can act on; the capture names the key."""
    served = start_server(_declared_deployment())
    gripless = ((ts, {key: value for key, value in obs.items() if key != keys.GRIP}) for ts, obs in _ticks(12))

    with against_server('websocket', served.host, served.port, '', '') as measured:
        with pytest.raises(ValueError, match="asks for 'grip'"):
            capture(gripless, measured.stack, _model(), requests=1)


def test_every_signal_the_episode_records_reaches_the_stack(tmp_path):
    """A declared stack can ask for a channel no whitelist here knows, and a drop reports a false absence."""
    # A channel `positronic.keys` does not name.
    suffixed = 'robot_state.left.q'
    period_ns = int(1e9 / 15.0)
    with DiskEpisodeWriter(tmp_path / 'episode') as writer:
        for tick in range(3):
            at = tick * period_ns
            writer.append(suffixed, np.zeros(7), at)
            writer.append(keys.GRIP, 0.0, at)
            writer.append(keys.WRIST_IMAGE, np.zeros((48, 64, 3), np.uint8), at)

    handed = [obs for _, obs in observations(DiskEpisode(tmp_path / 'episode'), rate_hz=15.0)]

    assert handed, 'the episode spans three ticks'
    for obs in handed:
        assert suffixed in obs, 'a suffixed state channel was dropped'
        assert keys.GRIP in obs and keys.WRIST_IMAGE in obs


def test_a_camera_the_flags_did_not_name_is_not_sent(tmp_path):
    """A flag-built stack forwards an unstacked camera at full size, so the wire carries what nobody asked for."""
    period_ns = int(1e9 / 15.0)
    with DiskEpisodeWriter(tmp_path / 'episode') as writer:
        for tick in range(3):
            at = tick * period_ns
            for camera in (*CAMERAS, keys.EXTERIOR_IMAGE_2):
                writer.append(camera, np.zeros((48, 64, 3), np.uint8), at)
            writer.append(keys.GRIP, 0.0, at)

    episode = DiskEpisode(tmp_path / 'episode')
    _, by_flags = next(iter(observations(episode, rate_hz=15.0, cameras=CAMERAS)))
    _, declared = next(iter(observations(episode, rate_hz=15.0)))

    assert keys.EXTERIOR_IMAGE_2 not in by_flags, 'a camera the flags did not name rode along to the wire'
    assert all(camera in by_flags for camera in CAMERAS)
    assert keys.EXTERIOR_IMAGE_2 in declared, 'a declared stack picks its own cameras, so every one is handed over'


def test_an_ambient_token_does_not_reach_a_server_the_run_never_named(start_server, monkeypatch):
    """`AUTH_TOKEN` is the operator's credential for one endpoint, not for every host `--server_host` dials."""
    token = 'a-token-for-another-endpoint'
    monkeypatch.setenv(AUTH_TOKEN_ENV, token)
    served = start_server(_declared_deployment(), auth_token=token)

    with pytest.raises(wire.ConnectRefused), against_server('websocket', served.host, served.port, '', ''):
        pass

    with against_server('websocket', served.host, served.port, '', '', bearer_headers.instantiate()) as measured:
        assert measured.stack is not None, 'the offered credential opened the session'
