from typing import Any

import numpy as np
import pytest
from positronic_wire.websocket import WebsocketClientWire

from pimm.time import RECEIVED_WORLD
from positronic import keys
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
        keys.TASK: 'Put the cup in the bowl',
        eval_keys.EMBODIMENT: 'test.franka',
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


class _StatesItsPoseFrame(Policy):
    """A policy that states the frame of the pose and keeps the first three observations it gets."""

    def __init__(self):
        self.seen: list[Obs] = []

    def meta(self) -> dict[str, Any]:
        return ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)})

    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        while len(self.seen) < 3:
            self.seen.append(obs)
            obs = yield Step({}, runtime.time_ns + round(1e9 / FPS))
        return PolicyEnd.COMPLETE


def test_a_policy_that_states_the_frame_of_the_pose_gets_the_pose():
    policy = _StatesItsPoseFrame()

    play(_episode(), policy, out=lambda _: None)

    np.testing.assert_allclose(policy.seen[2][keys.EE_POSE], [0.2, 0, 0, 1, 0, 0, 0])
