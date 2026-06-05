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
clock in the asserted path (a fixed-float ``simulate_inference`` is used so
inference latency is deterministic).

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
from positronic import wire
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, Recover, Reset, TrajectoryPlayer
from positronic.embodiment import franka
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.codec import ActionTiming
from positronic.policy.harness import Directive, Harness
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler

GOLDEN_FILE = Path(__file__).parent / 'golden_pipeline.json.gz'

INITIAL_POS = np.array([0.30, 0.00, 0.40], dtype=np.float32)
INITIAL_Q = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70], dtype=np.float32)
TARGET_POS = np.array([0.50, 0.00, 0.45], dtype=np.float32)

# Fixed deterministic inference latency. Spans >1 control tick (harness loop is
# 0.01 s) so the post-inference anchoring effect is observable in recorded ts.
SIMULATE_INFERENCE_S = 0.05
ACTION_FPS = 15.0
ACTION_HORIZON_S = 0.5  # 8 of every 10-action chunk survives truncation
CONTROL_PERIOD_S = 0.005  # fake robot/gripper sampling cadence (200 Hz)

# State signals captured at the DsWriterAgent output and locked by the golden.
CAPTURED_SIGNALS = ('robot_state.ee_pose', 'robot_state.q', 'grip')


class _ScriptedSession(Session):
    def __call__(self, obs):
        current = np.asarray(obs['robot_state.ee_pose'][:3], dtype=np.float32)
        delta = TARGET_POS - current
        chunk = []
        for i in range(10):
            step = current + delta * 0.5 * ((i + 1) / 10.0)
            pose = Transform3D(translation=step.astype(np.float32), rotation=Rotation.identity)
            chunk.append({'robot_command': CartesianPosition(pose=pose), 'target_grip': round(0.50 + 0.01 * i, 4)})
        return chunk


class ScriptedProportionalPolicy(Policy):
    """Pure proportional controller toward ``TARGET_POS``.

    Reads ``robot_state.ee_pose`` only; returns a 10-action chunk. No RNG, no
    clock, no images. Codec stamps/truncates; the harness anchors/schedules.
    """

    def new_session(self, context=None):
        return _ScriptedSession()


class _FakeRobotState:
    """Lossless re-expression of the last applied command over sim-time."""

    def __init__(self, pos: np.ndarray, q: np.ndarray, status: RobotStatus):
        self._pos = pos
        self._q = q
        self.status = status

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def dq(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    @property
    def ee_pose(self) -> Transform3D:
        return Transform3D(translation=self._pos.copy(), rotation=Rotation.identity)


class FakeRobot(pimm.ControlSystem):
    """Deterministic closed-loop arm: applies the latest command immediately.

    Mirrors ``MujocoFranka``'s structure (read latest command, apply, emit
    state). ``ee_pose`` becomes the applied ``CartesianPosition`` target and the
    first three joints track it, so recorded state is a lossless re-expression
    of applied commands. Closed loop: the policy's next chunk evolves with this
    feedback.
    """

    def __init__(self):
        self._pos = INITIAL_POS.copy()
        self._q = INITIAL_Q.copy()
        self._status = RobotStatus.AVAILABLE
        self._error_pending = False
        self.commands = pimm.ControlSystemReceiver(self, default=None)
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
            case Reset():
                self._pos = INITIAL_POS.copy()
                self._q = INITIAL_Q.copy()
                self._status = RobotStatus.AVAILABLE
            case Recover():
                self._status = RobotStatus.AVAILABLE

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        self.robot_meta.emit({})
        player = TrajectoryPlayer()
        while not should_stop.value:
            cmd_msg = self.commands.read()
            if cmd_msg.updated and cmd_msg.data is not None:
                player.set(cmd_msg.data)
            for cmd in player.advance(clock.now_ns()):
                self._apply(cmd)
            if self._error_pending:
                self._status = RobotStatus.ERROR
                self._error_pending = False
            self.state.emit(_FakeRobotState(self._pos, self._q, self._status))
            yield pimm.Sleep(CONTROL_PERIOD_S)


class FakeGripper(pimm.ControlSystem):
    """Identity gripper: reported grip equals the last applied target."""

    def __init__(self):
        self._grip = 0.0
        self.target_grip = pimm.ControlSystemReceiver(self, default=0.0)
        self.grip = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        player = TrajectoryPlayer()
        while not should_stop.value:
            msg = self.target_grip.read()
            if msg.updated and msg.data is not None:
                player.set(msg.data)
            for grip in player.advance(clock.now_ns()):
                self._grip = float(grip)
            self.grip.emit(self._grip)
            yield pimm.Sleep(CONTROL_PERIOD_S)


def _run_pipeline(tmp_path: Path) -> dict:
    """Run the full pipeline once; return per-signal recorded state."""
    policy = ActionTiming(fps=ACTION_FPS, horizon_sec=ACTION_HORIZON_S).wrap(ScriptedProportionalPolicy())
    robot = FakeRobot()
    gripper = FakeGripper()

    with LocalDatasetWriter(tmp_path) as ds_writer, pimm.World(virtual_time=True) as world:
        embodiment = franka(robot, gripper, descriptor='')
        harness = Harness(policy, embodiment, simulate_inference=SIMULATE_INFERENCE_S)
        ds_agent = wire.wire_embodiment(world, harness, embodiment, ds_writer, TimeMode.MESSAGE)
        world.connect(harness.ds_command, ds_agent.command)
        directive_em = world.pair(harness.directive)

        # Robot/gripper emit state every tick, so the script only drives the
        # episode lifecycle and the one-shot error injection.
        script = [
            (partial(directive_em.emit, Directive.RUN(task='golden')), 0.0),
            (None, 1.5),  # several reactive inference + chunk/horizon cycles
            (robot.inject_error, 0.0),  # -> ERROR -> harness Recover -> resume
            (None, 0.5),
            (None, 1.5),  # more cycles after recovery
            (partial(directive_em.emit, Directive.FINISH()), 0.0),
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
