"""A server that declares ``stream_frames`` gets the frames of a temporal stack ahead, one per message, and the
observation names them; a server that does not gets the stack as before."""

from typing import Any

import numpy as np
import pytest

from positronic import keys
from positronic.offboard import protocol
from positronic.offboard.client import InferenceSession
from positronic.offboard.tests.conftest import round_trip
from positronic.policy import Policy, RemotePolicy, Session
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import PolicySource, remote


class _Recording(Policy):
    """Answers a chunk that plays out for ``chunk_sec``, and keeps every observation it was called with."""

    def __init__(self, chunk_sec: float):
        self.seen: list[dict[str, Any]] = []
        self._chunk_sec = chunk_sec

    def new_session(self, context=None, rt=None) -> Session:
        policy = self

        class _S(Session):
            def __call__(self, obs, time_ns):
                policy.seen.append(dict(obs))
                return [{keys.ACTION_TIMESTAMP: 0.0}, {keys.ACTION_TIMESTAMP: policy._chunk_sec}]

        return _S()


def _pipeline(policy: Policy):
    return TemporalStack(keys=('x',), offsets_sec=(-0.2, -0.1, 0.0)) | ChunkedSchedule() | remote | PolicySource(policy)


def _obs(tick: int):
    return {keys.OBS_TIME_NS: int(tick * 0.1e9), 'x': np.array([tick, tick])}


@pytest.fixture
def sent(monkeypatch) -> dict[str, list]:
    """Everything the client put on the wire: the observations, and the frames sent ahead."""
    log: dict[str, list] = {'obs': [], 'frames': []}
    infer, push = InferenceSession.infer, InferenceSession.push_frame

    def _infer(self, obs):
        log['obs'].append(obs)
        return infer(self, obs)

    def _push(self, key, obs_time_ns, value):
        log['frames'].append((key, obs_time_ns, np.array(value)))
        push(self, key, obs_time_ns, value)

    monkeypatch.setattr(InferenceSession, 'infer', _infer)
    monkeypatch.setattr(InferenceSession, 'push_frame', _push)
    return log


def _drive(start_server, open_session, stream_frames: bool) -> _Recording:
    served = _Recording(chunk_sec=0.2)
    host, port, *_ = start_server(_pipeline(served), stream_frames=stream_frames)
    session, rt = open_session(RemotePolicy(f'ws://{host}:{port}'))
    try:
        # Tick 0 fires the first request; the chunk plays to 0.2 s, so tick 2 fires the second.
        assert round_trip(session, rt, _obs(0), time_ns=0) is not None
        assert session(_obs(1), int(0.1e9)) is None
        assert round_trip(session, rt, _obs(2), time_ns=int(0.2e9)) is not None
    finally:
        session.close()
    return served


def test_a_server_that_takes_frames_ahead_gets_ids_and_assembles_the_stack(start_server, open_session, sent):
    served = _drive(start_server, open_session, stream_frames=True)

    # The server saw the stack TemporalStack would have built: at tick 2, the frames at 0.0, 0.1 and 0.2 s.
    assert [o['x'].tolist() for o in served.seen] == [[[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 1], [2, 2]]]
    # The wire carried the ids, never the stack.
    assert [o['x'] for o in sent['obs']] == [
        {protocol.FRAME_IDS: [0, 0, 0]},
        {protocol.FRAME_IDS: [0, int(0.1e9), int(0.2e9)]},
    ]
    # Each frame went once, and the middle one went during the chunk, before the request that named it.
    assert [(k, t, v.tolist()) for k, t, v in sent['frames']] == [
        ('x', 0, [0, 0]),
        ('x', int(0.1e9), [1, 1]),
        ('x', int(0.2e9), [2, 2]),
    ]


def test_a_server_that_does_not_take_frames_ahead_gets_the_stack(start_server, open_session, sent):
    served = _drive(start_server, open_session, stream_frames=False)

    assert [o['x'].tolist() for o in served.seen] == [[[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 1], [2, 2]]]
    assert all(isinstance(o['x'], np.ndarray) and o['x'].shape == (3, 2) for o in sent['obs'])
    assert sent['frames'] == []
