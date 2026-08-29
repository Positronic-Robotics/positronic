"""Reproducible behavioral golden for the inference pipeline.

Locks the robot-facing behavior of the full chain:
policy -> codec -> Harness -> timestamp-respecting driver -> DsWriterAgent.

The golden is the *state* the episode actually records (``robot_state.ee_pose``,
``robot_state.q``, ``grip``) at the ``DsWriterAgent`` output. State is the
representation-free effect of the pipeline: a deterministic closed-loop fake
robot changes state only when a command is *applied*, so

  * value regression           -> ``value`` differs
  * timing/anchoring regression -> same state at a different ``ts_ns``
  * horizon/gating regression   -> the state trajectory diverges

Everything runs on CPU in a virtual-time world only: no GL/GPU/MuJoCo, no wall
clock in the asserted path (``_SimulatedLatency`` charges a fixed world-time
latency, so inference latency is deterministic).

Regenerate the golden after an intentional behavior change:

    GOLDEN=1 uv run pytest positronic/policy/tests/test_golden_pipeline.py \
        -p no:cacheprovider -o "addopts="
"""

import gzip
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import pimm
from positronic import keys, wire
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, CommandType
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation, Task
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Answer, ChunkSession, DelegatingChunkSession, DelegatingPolicy, Done, Policy
from positronic.policy.codec import ActionTiming
from positronic.policy.harness import Harness
from positronic.policy.layers import ChunkPlayer, StopOnFault
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler

GOLDEN_FILE = Path(__file__).parent / 'golden_pipeline.json.gz'

INITIAL_POS = np.array([0.30, 0.00, 0.40], dtype=np.float32)
INITIAL_Q = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70], dtype=np.float32)
TARGET_POS = np.array([0.50, 0.00, 0.45], dtype=np.float32)

# Fixed deterministic inference latency in world time. Spans >1 control tick (harness loop is
# 0.01 s) so the shift ``_SimulatedLatency`` adds is observable in recorded ts.
INFERENCE_LATENCY_S = 0.05
ACTION_FPS = 15.0
ACTION_HORIZON_S = 0.5  # 8 of every 10-action chunk survives truncation
CONTROL_PERIOD_S = 0.005  # fake robot/gripper sampling cadence (200 Hz)

# State signals captured at the DsWriterAgent output and locked by the golden.
CAPTURED_SIGNALS = (keys.EE_POSE, keys.JOINTS, keys.GRIP)


class _ScriptedSession(ChunkSession):
    def __call__(self, obs, time_ns):
        current = np.asarray(obs[keys.EE_POSE][:3], dtype=np.float32)
        delta = TARGET_POS - current
        chunk = []
        for i in range(10):
            step = current + delta * 0.5 * ((i + 1) / 10.0)
            pose = Transform3D(translation=step.astype(np.float32), rotation=Rotation.identity)
            chunk.append({keys.ROBOT_COMMAND: CartesianPosition(pose=pose), 'target_grip': round(0.50 + 0.01 * i, 4)})
        return Done(chunk)


class ScriptedProportionalPolicy(Policy):
    """Pure proportional controller toward ``TARGET_POS``.

    Reads ``robot_state.ee_pose`` only; returns a 10-action chunk. No RNG, no
    clock, no images. The codec stamps and truncates; ``ChunkPlayer`` anchors the chunk and plays it.
    """

    def new_session(self, context=None, rt=None):
        return _ScriptedSession()


class _SimulatedLatency(DelegatingPolicy):
    """A fixed inference latency in world time: every chunk is stamped for, and answers at, ``latency_sec``
    after the call that produced it, whatever the machine took. Sits under the player, so nothing below it
    sees an observation while a chunk is in flight."""

    def __init__(self, inner: Policy, latency_sec: float, clock: pimm.Clock):
        super().__init__(inner)
        self._latency_ns = round(latency_sec * 1e9)
        self._clock = clock

    class _Held(Answer):
        """The chunk under it, held back until the world reaches ``release_at_ns``."""

        def __init__(self, inner: Answer, release_at_ns: int, clock: pimm.Clock):
            self._inner = inner
            self._release_at_ns = release_at_ns
            self._clock = clock

        def done(self) -> bool:
            return self._clock.now_ns() >= self._release_at_ns and self._inner.done()

        def result(self):
            return self._inner.result()

    class _Session(DelegatingChunkSession):
        def __init__(self, inner: ChunkSession, latency_ns: int, clock: pimm.Clock):
            super().__init__(inner)
            self._latency_ns = latency_ns
            self._clock = clock

        def __call__(self, obs, time_ns):
            # The chunk answers at ``latency_ns``, so the sessions below time it from that instant.
            release_at_ns = time_ns + self._latency_ns
            return _SimulatedLatency._Held(self._inner(obs, release_at_ns), release_at_ns, self._clock)

    def new_session(self, context=None, rt=None) -> ChunkSession:
        inner = self._inner.new_session(context, rt)
        assert isinstance(inner, ChunkSession)
        return _SimulatedLatency._Session(inner, self._latency_ns, self._clock)


class FakeRobot(pimm.ControlSystem):
    """Deterministic closed-loop arm: applies each command as it arrives.

    Mirrors ``MujocoSim``'s arm loop (execute on updated, emit state). ``ee_pose`` becomes the applied
    ``CartesianPosition`` target and the first three joints track it, so recorded state is a lossless
    re-expression of applied commands. Closed loop: the policy's next chunk evolves with this feedback.
    """

    def __init__(self):
        self._pos = INITIAL_POS.copy()
        self._q = INITIAL_Q.copy()
        self._status = RobotStatus.AVAILABLE
        self._error_pending = False
        self.commands = pimm.ControlSystemReceiver[CommandType](self)
        self.state = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def inject_error(self):
        self._error_pending = True

    def _apply(self, cmd):
        match cmd:
            case CartesianPosition(pose=pose):
                self._pos = np.asarray(pose.translation, dtype=np.float32)
                self._q = self._q.copy()
                self._q[:3] = self._pos

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        self.robot_meta.emit({})
        while not should_stop.value:
            cmd_msg = self.commands.read()
            if self._status == RobotStatus.ERROR:
                self._status = RobotStatus.AVAILABLE
            elif cmd_msg is not None and cmd_msg.updated:
                self._apply(cmd_msg.data)
            if self._error_pending:
                self._status = RobotStatus.ERROR
                self._error_pending = False
            self.state.emit(make_robot_state(self._pos, self._q, self._status))
            yield pimm.Sleep(CONTROL_PERIOD_S)


class FakeGripper(pimm.ControlSystem):
    """Identity gripper: reported grip equals the last applied target."""

    def __init__(self):
        self._grip = 0.0
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.grip = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            msg = self.target_grip.read()
            if msg is not None and msg.updated:
                self._grip = float(msg.data)
            self.grip.emit(self._grip)
            yield pimm.Sleep(CONTROL_PERIOD_S)


def _run_pipeline(tmp_path: Path) -> dict:
    """Run the full pipeline once; return per-signal recorded state."""
    policy = ActionTiming(fps=ACTION_FPS, horizon_sec=ACTION_HORIZON_S).wrap(ScriptedProportionalPolicy())
    robot = FakeRobot()
    gripper = FakeGripper()

    with LocalDatasetWriter(tmp_path) as ds_writer, pimm.World(virtual_time=True) as world:
        embodiment = Embodiment(
            descriptor='',
            observations={
                keys.ROBOT_STATE: Observation(robot.state, Serializers.robot_state),
                keys.GRIP: Observation(gripper.grip, None),
            },
            commands={
                keys.ROBOT_COMMAND: Command(robot.commands, Serializers.robot_command),
                'target_grip': Command(gripper.target_grip, None),
            },
            prepare_handlers={},
            static_meta=dict(ROBOT_STATIC_META),
            meta_source=robot.robot_meta,
            # The fake robot's control-period sleep is this world's sole time-master — the shape a sim eval
            # runs in.
            simulated=True,
        )
        harness = Harness(
            (StopOnFault() | ChunkPlayer()).wrap(_SimulatedLatency(policy, INFERENCE_LATENCY_S, world.clock)),
            embodiment,
        )
        ds_agent = wire.wire_embodiment(world, harness, embodiment, ds_writer, TimeMode.MESSAGE)
        world.connect(harness.ds_command, ds_agent.command)
        perform_task = world.pair(harness.perform_task)
        done_em = world.pair(harness.done)

        # Robot/gripper emit state every tick, so the script only drives the
        # episode lifecycle and the one-shot error injection.
        script = [
            (partial(perform_task, Task(instruction_source='golden', timeout_sec=None)), 0.0),
            (None, 1.5),  # several reactive inference + chunk/horizon cycles
            (robot.inject_error, 0.0),  # one-shot error: StopOnFault stops the arm for that frame
            (None, 0.5),
            (None, 1.5),  # more cycles after recovery
            (partial(done_em.emit, {keys.EVAL_ENDED_BY: keys.ENDED_BY_OPERATOR}), 0.0),
            (None, 0.5),  # let DsWriterAgent commit before world exit
        ]
        scheduler = world.start([harness, ManualDriver(script), robot, gripper, ds_agent])
        drive_scheduler(scheduler, steps=8000)

    episode = LocalDataset(tmp_path)[0]
    out: dict[str, dict] = {}
    for name in CAPTURED_SIGNALS:
        sig = episode[name]
        ts_ns = [int(t) for t in sig.keys()]
        values = []
        for v in sig.values():
            arr = np.asarray(v)
            values.append(arr.tolist() if arr.ndim else float(arr))
        out[name] = {'ts_ns': ts_ns, 'value': values}
    return out


def test_golden_pipeline(tmp_path):
    recorded = _run_pipeline(tmp_path)
    assert all(recorded[s]['ts_ns'] for s in CAPTURED_SIGNALS), 'Pipeline recorded no state'

    if os.environ.get('GOLDEN'):
        with gzip.open(GOLDEN_FILE, 'wt') as f:
            json.dump(recorded, f, separators=(',', ':'))
        pytest.skip(f'Golden written to {GOLDEN_FILE}')

    assert GOLDEN_FILE.exists(), f'{GOLDEN_FILE} missing; regenerate with GOLDEN=1'
    with gzip.open(GOLDEN_FILE, 'rt') as f:
        golden = json.load(f)

    assert set(recorded) == set(golden), f'Signal set mismatch: {set(recorded)} vs {set(golden)}'
    for name in CAPTURED_SIGNALS:
        got, exp = recorded[name], golden[name]
        assert len(got['ts_ns']) == len(exp['ts_ns']), (
            f'{name}: sample count {len(got["ts_ns"])} != {len(exp["ts_ns"])} (golden)'
        )
        assert got['ts_ns'] == exp['ts_ns'], f'{name}: ts_ns diverged (timing/anchoring regression)'
        np.testing.assert_allclose(got['value'], exp['value'], atol=1e-6, err_msg=f'{name}: value diverged')
