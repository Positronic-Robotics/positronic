from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pytest
from positronic_wire import websocket, wire

from positronic import keys
from positronic.offboard import protocol
from positronic.offboard.client import InferenceClient
from positronic.offboard.serving_cost import InstantChunk, InstantSource, capture, replay, rig_stack
from positronic.offboard.spec import PolicyDeployment

CAMERAS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)


def _ticks(count: int, period_ns: int = 66_666_666):
    """A stand-in for the harness: one moving frame per camera per control tick."""
    rng = np.random.default_rng(0)
    for tick in range(count):
        frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        yield {
            keys.EE_POSE: np.zeros(7),
            keys.GRIP: 0.0,
            keys.ROBOT_STATUS: 0,
            keys.OBS_TIME_NS: 1_000_000_000 + tick * period_ns,
            **dict.fromkeys(CAMERAS, frame),
        }


def test_replay_divides_a_round_trip_into_the_phases_the_server_reports(start_server):
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    model = InstantChunk(rows=2)
    payloads = capture(_ticks(12), stack, partial(model, session_id='capture'), requests=2)
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


def test_a_captured_payload_carries_one_stack_per_stacked_key():
    stack = rig_stack(CAMERAS, frames=3, rate_hz=15.0, width=64, height=48)
    payloads = capture(_ticks(12), stack, partial(InstantChunk(rows=2), session_id='capture'), requests=1)

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
