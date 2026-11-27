import threading
import time
from typing import Any

import numpy as np
import pytest
from positronic_wire.websocket import WebsocketClientWire

from pimm.time import RECEIVED_WORLD
from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry
from positronic.eval import keys as eval_keys
from positronic.offboard.replay_episode import play
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Policy, PolicyCommandError, RemotePolicy
from positronic.policy.base import Obs, PolicyEnd, PolicyRun, Runtime, Step
from positronic.policy.processors import ChunkedSchedule, TrajectoryPlayer
from positronic.policy.sequential import Sequential

TIMES_NS = [round(i * 0.1e9) for i in range(31)]
CHUNK = 3
FPS = 10


def _episode(timeline: str = HARNESS_WORLD) -> EpisodeContainer:
    """A 3 s episode at 10 Hz: each joint holds ``t``, the pose is at ``t`` in x, and the recorded target is ``-t``."""
    seconds = np.array(TIMES_NS) / 1e9

    def signal(values: list[Any]) -> DummySignal:
        return DummySignal(TIMES_NS, values, timelines=(timeline,))

    return EpisodeContainer({
        keys.JOINTS: signal([np.full(7, t) for t in seconds]),
        keys.EE_POSE: signal([np.array([t, 0, 0, 1, 0, 0, 0]) for t in seconds]),
        keys.GRIP: signal([0.5] * len(TIMES_NS)),
        keys.TARGET_JOINTS: signal([np.full(7, -t) for t in seconds]),
        keys.TARGET_GRIP: signal([1.0] * len(TIMES_NS)),
        keys.TASK: 'Put the cup in the bowl',
        eval_keys.EMBODIMENT: 'test.franka',
        **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE), keys.TARGET_EE_POSE: EEFrame(site=GRIPPER_SITE)}),
    })


class _Hold(Model):
    """Answers a chunk of joint targets equal to the joints it sees, and keeps each observation."""

    def __init__(self, extra: dict[str, Any] | None = None):
        self.seen: list[Obs] = []
        self._extra = extra or {}

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        self.seen.append(obs)
        target = {keys.ROBOT_COMMAND: command.JointPosition(obs[keys.JOINTS]), keys.TARGET_GRIP: 1.0, **self._extra}
        return [target] * CHUNK

    def meta(self) -> dict[str, Any]:
        return {}


def _served(start_server, model: Model) -> RemotePolicy:
    served = start_server(model, PolicyDeployment(Sequential(TrajectoryPlayer(), ChunkedSchedule(fps=FPS))))
    return RemotePolicy(WebsocketClientWire.NAME, served.ws()[1])


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_one_call_gets_the_recorded_observation_at_its_start_and_plays_its_chunk(start_server, timeline):
    model = _Hold()
    lines: list[str] = []

    assert play(_episode(timeline), _served(start_server, model), start_sec=1.0, calls=1, out=lines.append) == 1

    seen = model.seen[0]
    np.testing.assert_allclose(seen[keys.JOINTS], np.full(7, 1.0))
    assert seen[keys.TASK] == 'Put the cup in the bowl'
    assert seen[keys.DESCRIPTOR] == 'test.franka'
    assert keys.EE_POSE not in seen
    assert sum(' robot_command = JointPosition' in line for line in lines) == CHUNK
    assert any(line.startswith('   1.000 s  recorded robot_command.joints = ') for line in lines)


def test_an_answer_with_a_field_a_rig_does_not_take_raises(start_server):
    with pytest.raises(PolicyCommandError, match='unknown'):
        play(_episode(), _served(start_server, _Hold(extra={'unknown': 1})), calls=1, out=lambda _: None)


def test_an_answer_on_a_channel_the_episode_does_not_record_raises(start_server):
    grip_of_left_arm = keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)

    with pytest.raises(PolicyCommandError, match=grip_of_left_arm):
        play(_episode(), _served(start_server, _Hold(extra={grip_of_left_arm: 1.0})), calls=1, out=lambda _: None)


class _Local(Policy):
    """A local policy that states ``frames``, answers ``commands`` and keeps the first three observations it gets.

    With ``call``, each step submits a call that reads, 0.1 s after it starts, whether the run is still open.
    """

    def __init__(
        self, frames: dict[str, EEFrame] | None = None, commands: dict[str, Any] | None = None, call: bool = False
    ):
        self.seen: list[Obs] = []
        self.open_in_calls: list[bool] = []
        self._frames = frames or {}
        self._commands = commands or {}
        self._call = call
        self._call_started = threading.Event()
        self._open = False

    def meta(self) -> dict[str, Any]:
        return ee_frames_entry(self._frames) if self._frames else {}

    def run(self, runtime: Runtime) -> PolicyRun:
        self._open = True
        try:
            obs = yield
            while len(self.seen) < 3:
                self.seen.append(obs)
                if self._call:
                    runtime.submit(self._read_open)
                    assert self._call_started.wait(5)
                obs = yield Step(self._commands, runtime.time_ns + round(1e9 / FPS))
            return PolicyEnd.COMPLETE
        finally:
            self._open = False

    def _read_open(self) -> None:
        self._call_started.set()
        time.sleep(0.1)
        self.open_in_calls.append(self._open)


def test_a_policy_that_states_the_frame_of_the_pose_gets_the_pose():
    policy = _Local(frames={keys.EE_POSE: EEFrame(site=GRIPPER_SITE)})

    play(_episode(), policy, out=lambda _: None)

    np.testing.assert_allclose(policy.seen[2][keys.EE_POSE], [0.2, 0, 0, 1, 0, 0, 0])


def test_a_policy_that_states_another_site_than_the_recording_raises():
    with pytest.raises(ValueError, match='site the rig does not report'):
        play(_episode(), _Local(frames={keys.EE_POSE: EEFrame(site='other_site')}), out=lambda _: None)


def test_a_pose_command_whose_frame_the_policy_does_not_state_raises():
    policy = _Local(commands={keys.ROBOT_COMMAND: command.CartesianPosition(geom.Transform3D())})

    with pytest.raises(PolicyCommandError, match='states no'):
        play(_episode(), policy, out=lambda _: None)


def test_a_call_open_when_the_play_stops_ends_before_the_run_closes():
    policy = _Local(call=True)

    play(_episode(), policy, calls=0, out=lambda _: None)

    assert policy.open_in_calls == [True]


class _FailsToStart(Policy):
    """A policy that submits a call as it starts, and then raises before its first observation."""

    def __init__(self):
        self.call_ended = threading.Event()
        self._call_started = threading.Event()

    def meta(self) -> dict[str, Any]:
        return {}

    def run(self, runtime: Runtime) -> PolicyRun:
        runtime.submit(self._call)
        assert self._call_started.wait(5)
        raise RuntimeError('the policy fails to start')
        yield

    def _call(self) -> None:
        self._call_started.set()
        time.sleep(0.1)
        self.call_ended.set()


def test_a_call_of_a_policy_that_fails_to_start_ends_before_the_play_raises():
    policy = _FailsToStart()

    with pytest.raises(RuntimeError, match='fails to start'):
        play(_episode(), policy, out=lambda _: None)

    assert policy.call_ended.is_set()
