from unittest.mock import MagicMock

import numpy as np
import pytest

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode, DiskEpisodeWriter
from positronic.offboard import protocol
from positronic.offboard.client import RECV_MS, SEND_MS, InferenceClient
from positronic.offboard.serving_cost import InstantChunk, against_server, capture, observations, replay, rig_stack
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.spec import PolicySource, remote

CAMERAS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)

# A stack shaped like the one a video-conditioned server declares: four strided frames per camera,
# bounded well under what this probe's own flags default to.
DECLARED_OFFSETS_SEC = (-23 / 15, -16 / 15, -8 / 15, 0.0)
DECLARED_WIDTH, DECLARED_HEIGHT = 320, 176


def _declared_stack():
    return (
        StopOnFault()
        | TemporalStack(keys=(*CAMERAS, keys.EE_POSE, keys.GRIP), offsets_sec=DECLARED_OFFSETS_SEC)
        | ChunkedSchedule()
        | RestrictImageSize(width=DECLARED_WIDTH, height=DECLARED_HEIGHT)
    )


def _ticks(count: int, period_ns: int = 66_666_666, size: tuple[int, int] = (48, 64)):
    """A stand-in for the harness: one moving frame per camera per control tick."""
    rng = np.random.default_rng(0)
    for tick in range(count):
        frame = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        yield {
            keys.EE_POSE: np.zeros(7),
            keys.GRIP: 0.0,
            keys.ROBOT_STATUS: 0,
            keys.OBS_TIME_NS: 1_000_000_000 + tick * period_ns,
            **dict.fromkeys(CAMERAS, frame),
        }


def test_replay_divides_a_round_trip_into_the_phases_the_server_reports(start_server):
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    model = InstantChunk(rows=2, period_s=1 / 15.0)
    payloads = capture(_ticks(12), stack, model, requests=2)
    assert payloads, 'the stack sent nothing'

    host, port, *_ = start_server(stack | remote(compress_images=True) | PolicySource(model))
    session = InferenceClient.from_url(f'ws://{host}:{port}').new_session()
    try:
        rows = replay(session, payloads, compress_images=True)
    finally:
        session.close()

    assert len(rows) == len(payloads)
    for row in rows:
        assert row['wire_kib'] > 0
        assert row[protocol.TIMING_SERVED] >= row[protocol.TIMING_DECODE]
        assert row['round_trip_ms'] >= row[protocol.TIMING_SERVED]


def test_a_captured_payload_carries_one_stack_per_stacked_key():
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    payloads = capture(_ticks(12), stack, InstantChunk(rows=2, period_s=1 / 15.0), requests=1)

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
    model = InstantChunk(rows=2, period_s=1 / 15.0)
    host, port, *_ = start_server(_declared_stack() | remote(compress_images=True) | PolicySource(model))

    with against_server(f'ws://{host}:{port}') as measured:
        assert measured.compress_images, 'the wire setting comes from the handshake too'
        payloads = capture(_ticks(40, size=(360, 640)), measured.stack, model, requests=2)
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
    model = InstantChunk(rows=2, period_s=1 / 15.0)
    host, port, *_ = start_server(_declared_stack() | remote(compress_images=True) | PolicySource(model))
    gripless = ({key: value for key, value in obs.items() if key != keys.GRIP} for obs in _ticks(12))

    with against_server(f'ws://{host}:{port}') as measured:
        with pytest.raises(ValueError, match="asks for 'grip'"):
            capture(gripless, measured.stack, model, requests=1)


def test_every_signal_the_episode_records_reaches_the_stack(tmp_path):
    """A declared stack can ask for a channel no whitelist here knows, and a drop reports a false absence."""
    period_ns = int(1e9 / 15.0)
    with DiskEpisodeWriter(tmp_path / 'episode') as writer:
        for tick in range(3):
            at = tick * period_ns
            # A bimanual rig records a suffixed state channel, which no fixed key list here would name.
            writer.append('robot_state.left.q', np.zeros(7), at)
            writer.append(keys.GRIP, 0.0, at)
            writer.append(keys.WRIST_IMAGE, np.zeros((48, 64, 3), np.uint8), at)

    handed = list(observations(DiskEpisode(tmp_path / 'episode'), rate_hz=15.0))

    assert handed, 'the episode spans three ticks'
    for obs in handed:
        assert 'robot_state.left.q' in obs, 'a suffixed state channel was dropped'
        assert keys.GRIP in obs and keys.WRIST_IMAGE in obs
