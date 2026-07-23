import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco as mj
import numpy as np
import pos3
import pytest
import tqdm

import pimm
import positronic.cfg.simulator
from positronic import keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Eval, Observation, Task
from positronic.inference import main
from positronic.policy.tests.test_harness import StubPolicy
from positronic.policy.wrappers import ChunkedSchedule
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.simulator.mujoco.transforms import AddBox, SetBodyPosition
from positronic.utils import package_assets_path


# This integration test exercises the unified `main` end-to-end on the sim embodiment.
@pytest.mark.timeout(30.0)
def test_sim_emits_commands_and_records_dataset(tmp_path, monkeypatch):
    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0.0

        def refresh(self):
            pass

        def close(self):
            pass

        def update(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(tqdm, 'tqdm', lambda *args, **kwargs: DummyTqdm(*args, **kwargs))
    monkeypatch.setenv('MUJOCO_GL', 'egl')

    class FakeRenderer:
        def __init__(self, _model, *, height, width, max_geom=10000, font_scale=None):
            self.height = height
            self.width = width

        def update_scene(self, _data, camera=None):
            pass

        def render(self, out=None):
            if out is not None:
                out[:] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return None
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr('positronic.simulator.mujoco.sim.mj.Renderer', FakeRenderer)

    policy = StubPolicy()

    camera_dict = {keys.WRIST_IMAGE: 'handcam_left_ph'}

    with pos3.mirror():
        ev = stack_cubes(
            mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
            loaders=positronic.cfg.simulator.stack_cubes_loaders(),
            camera_fps=10,
            camera_dict=camera_dict,
            instruction='integration-test',
            timeout=0.4,
        )
        main(
            policy=policy,
            evals=[replace(ev, trials=[{'eval.trial_index': i, 'eval.seed': 100 + i} for i in range(2)])],
            output_dir=str(tmp_path),
            wrap=ChunkedSchedule(),
        )

    ds = LocalDataset(tmp_path)
    # Two trials: the harness runs the plan itself, self-terminating each trial at the task's timeout.
    assert len(ds) == 2

    episode = ds[0]
    assert episode.static['eval.terminated'] is False
    assert episode.static['eval.trial_index'] == 0
    assert episode.static['eval.seed'] == 100
    assert episode.static['eval.universe'] == 'sim'
    assert episode.static['eval.embodiment'] == 'mujoco.franka'
    assert episode.static['eval.timeout'] == 0.4
    # The post-loader scene description rides robot_meta into static: with eval.seed it makes
    # the episode self-contained for downstream scoring and faithful replay.
    assert episode.static['scene_xml'].startswith('<mujoco')
    signals = episode.signals
    assert 'robot_command.pose' in signals
    assert 'target_grip' in signals
    assert keys.WRIST_IMAGE in signals
    # Privileged ground truth: the full sim state is recorded as a time-series signal.
    assert 'sim_state.mjSTATE_INTEGRATION' in signals

    camera_samples = list(signals[keys.WRIST_IMAGE])
    assert camera_samples, 'Camera signal for handcam_left is empty'
    first_image, _ = camera_samples[0]
    assert isinstance(first_image, np.ndarray)

    # Frame-0 is recorded for every trial, not just the first: each episode's first sim-state sample is its
    # own post-reset scene (seeds 100 and 101), bit-reproducible from a fresh reset on the same seed. A
    # dropped frame-0 — or a stale step from the prior trial's run loop bleeding in — would record a
    # post-step state instead.
    for i, seed in enumerate((100, 101)):
        reference = MujocoSim(
            'positronic/assets/mujoco/franka_table.xml', positronic.cfg.simulator.stack_cubes_loaders()
        )
        reference.reset(seed=seed)
        first_state, _ = list(ds[i].signals['sim_state.mjSTATE_INTEGRATION'])[0]
        np.testing.assert_allclose(first_state, reference.save_state()['mjSTATE_INTEGRATION'], rtol=1e-6)

    pose_signal = signals['robot_command.pose']
    pose_samples = list(pose_signal)
    assert pose_samples, 'robot_command.pose signal is empty'
    first_pose, _first_pose_ts = pose_samples[0]
    np.testing.assert_allclose(first_pose[:3], np.array([0.4, 0.5, 0.6], dtype=np.float32))
    np.testing.assert_allclose(first_pose[3:], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.all(np.diff([ts for _, ts in pose_samples]) > 0) or len(pose_samples) == 1

    grip_signal = signals['target_grip']
    grip_samples = list(grip_signal)
    assert grip_samples, 'target_grip signal is empty'
    grip_values = [value for value, _ts in grip_samples]
    # The inter-episode home grip is drained by ``reset``, so the command stream starts with the policy's
    # first commanded grip (0.33) — consistent with ``robot_command.pose`` above.
    assert grip_values[0] == pytest.approx(0.33, rel=1e-2, abs=1e-2)
    assert np.all(np.diff([ts for _, ts in grip_samples]) > 0) or len(grip_samples) == 1

    assert policy.observations, 'Policy did not receive any observations'
    last_obs = policy.observations[-1]
    assert isinstance(last_obs[keys.WRIST_IMAGE], np.ndarray)
    assert keys.EE_POSE in last_obs
    # The task's instruction is injected by the harness.
    assert last_obs[keys.TASK] == 'integration-test'
    assert last_obs['descriptor'] == 'mujoco.franka'


class _CountdownProducer(pimm.ControlSystem):
    """A local deterministic producer standing in for the simulator, so the harness+recorder control loop
    is exercised end to end without MuJoCo. Obs encodes the env step count — ``reset`` is step 0, each step
    adds 1 — so a recorded episode's first ``value`` sample is all-zeros iff the recorder logged the
    post-reset frame-0. ``done`` fires after ``done_after`` steps (``None`` → never). The producer is the
    eval's sole time-master: it sleeps one ``control_dt`` every turn, publishing frame-0 in its own turn
    (in sequence, after the recorder's open-turn drain) and free-running — it advances each tick regardless
    of the commands the policy emits.
    """

    def __init__(self, done_after: int | None = None, control_dt: float = 0.01):
        self._done_after = done_after
        self._control_dt = control_dt
        self._steps = 0
        self._active = False
        self._reset_pending = False
        self.observations = pimm.EmitterDict(self)
        self.commands = pimm.ReceiverDict(self, default=None)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        self.done = pimm.ControlSystemEmitter(self)

    def reset(self, _context: dict | None = None) -> None:
        self._steps = 0
        self._reset_pending = True
        self._active = True

    def _emit_obs(self) -> None:
        self.observations['value'].emit(np.full(7, self._steps, dtype=np.float64))

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            yield pimm.Sleep(self._control_dt)
            if self._reset_pending:
                self._reset_pending = False
                self.robot_meta.emit({})
                self._emit_obs()  # frame-0 (step 0), before any step advances the env
                self.done.emit({})
            elif self._active:
                self._steps += 1
                self._emit_obs()
                if self._done_after is not None and self._steps >= self._done_after:
                    self.done.emit({'eval.success': True})
                    self._active = False


def _countdown_eval(producer: _CountdownProducer, timeout: float) -> Eval:
    embodiment = Embodiment(
        descriptor='test.countdown',
        observations={'value': Observation(producer.observations['value'], None)},
        commands={
            'robot_command': Command(
                producer.commands['robot_command'], roboarm_command.Reset(), Serializers.robot_command
            )
        },
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=producer.robot_meta,
        control_systems=(producer,),
        simulated=True,
    )
    task = Task(instruction='count', timeout=timeout, privileged={}, reset=producer.reset, done=producer.done)
    return Eval(embodiment, task)


@pytest.mark.timeout(30.0)
def test_countdown_records_frame0_every_trial(tmp_path):
    """[harness + recorder + sim] with no MuJoCo: every recorded episode's first ``value`` sample is the
    post-reset frame-0 (all-zeros), trials after the first included. Proves the recorder's open-turn drain
    drops the pre-reset frame and the producer publishes frame-0 in sequence. The small ``control_dt`` wakes
    the producer quickly between trials, so a stray step would overwrite frame-0 if it weren't published in
    the producer's own turn."""
    ev = _countdown_eval(_CountdownProducer(control_dt=0.01), timeout=0.35)
    with pos3.mirror():
        main(
            policy=StubPolicy(command=ev.embodiment.commands['robot_command'].home, target_grip=0.0),
            evals=[replace(ev, trials=[{'eval.trial_index': i, 'eval.seed': i} for i in range(2)])],
            output_dir=str(tmp_path),
            wrap=None,  # the degenerate obs is not Franka-shaped, so run the policy unwrapped
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 2
    for i in range(2):
        first_value, _ts = ds[i].signals['value'][0]
        np.testing.assert_array_equal(first_value, np.zeros(7))


@pytest.mark.timeout(30.0)
def test_countdown_terminates_on_done_records_payload(tmp_path):
    """[harness + recorder + sim] with no MuJoCo: a self-driven trial ends early when the producer's ``done``
    fires, recording ``eval.terminated`` True and the delivered payload into the episode's static data."""
    ev = _countdown_eval(_CountdownProducer(done_after=4), timeout=15.0)
    with pos3.mirror():
        main(
            policy=StubPolicy(command=ev.embodiment.commands['robot_command'].home, target_grip=0.0),
            evals=[replace(ev, trials=[{'eval.trial_index': 0, 'eval.seed': 100}])],
            output_dir=str(tmp_path),
            wrap=None,
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    episode = ds[0]
    assert episode.static['eval.terminated'] is True
    assert episode.static['eval.success'] is True  # the delivered ``done`` payload lands in static data
    assert episode.static['eval.embodiment'] == 'test.countdown'
    # The terminal frame (the step where ``done`` fired) is recorded, not dropped by STOP closing the writer.
    values = [v for v, _ in episode.signals['value']]
    assert values[-1][0] == 4.0


@pytest.mark.timeout(30.0)
def test_sim_reset_seed_reproduces_scene():
    """Same seed → identical post-reset scene, state- and model-level; different seed → a different scene.

    The fixed (non-freejointed) marker box randomizes at the model level, so it only
    re-randomizes because reset rebuilds the model wholesale.
    """
    loaders = [
        *positronic.cfg.simulator.stack_cubes_loaders(),
        AddBox(name='marker', size=[0.01, 0.01, 0.01], pos=[0.0, 0.0, 0.05]),
        SetBodyPosition(body_name='marker_body', random_position=[[0.9, 0.5, 0.05], [1.0, 0.6, 0.05]]),
    ]
    sim = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders)
    sim.reset(seed=123)
    first = sim.save_state()
    first_xml = sim.scene_xml
    first_marker = sim.model.body('marker_body').pos.copy()
    sim.reset(seed=99)
    second = sim.save_state()
    second_marker = sim.model.body('marker_body').pos.copy()
    sim.reset(seed=123)
    third = sim.save_state()

    for name, array in first.items():
        np.testing.assert_array_equal(third[name], array)
    assert sim.scene_xml == first_xml
    np.testing.assert_array_equal(sim.model.body('marker_body').pos, first_marker)
    assert any(not np.array_equal(second[name], array) for name, array in first.items())
    assert not np.array_equal(second_marker, first_marker)

    # A fresh sim (its own random draw) restores the recorded scene wholesale: load_state
    # rebuilds the model from scene_xml, so model-level randomization replays faithfully.
    # Model fields pass through XML text, so they round-trip to float-printing precision;
    # the state arrays are set verbatim and stay exact.
    replayed = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders)
    replayed.load_state({**third, 'scene_xml': first_xml}, reset_time=False)
    np.testing.assert_allclose(replayed.model.body('marker_body').pos, first_marker, rtol=1e-6)
    for name, array in replayed.save_state().items():
        np.testing.assert_array_equal(array, third[name])


def test_sim_state_reconstructs_dynamics():
    """A recorded sim state reconstructs the simulation, not just the instant: restore it and the
    continued trajectory matches the original step-for-step.

    This is the contract that lets ``STATE_SPECS`` be a minimal subset — it must carry everything a
    forward step reads (the solver warm-start, the actuator ``ctrl``), or contact-rich motion
    diverges and the assertions below break.
    """
    sim = MujocoSim('positronic/assets/mujoco/franka_table.xml', positronic.cfg.simulator.stack_cubes_loaders())
    sim.reset(seed=123)
    # Drive the arm off equilibrium so the saved state carries live velocity and warm-started contact
    # forces; a settled scene would not exercise the parts a thinned subset drops.
    sim.data.ctrl[:7] += 0.3
    for _ in range(200):
        sim.step()

    saved = sim.save_state()
    trajectory = []
    for _ in range(300):
        sim.step()
        trajectory.append((sim.data.qpos.copy(), sim.data.qvel.copy()))

    # Restore onto the same model (no ``scene_xml`` → no recompile) so the comparison isolates state
    # sufficiency from the model's float-printed XML round-trip.
    sim.load_state(saved, reset_time=False)
    for qpos, qvel in trajectory:
        sim.step()
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        np.testing.assert_array_equal(sim.data.qvel, qvel)


def test_recorded_urdf_matches_sim_kinematics():
    """The model the sim records for the viewer and IK reproduces the MuJoCo model the sim runs:
    the arm link frames and the ``end_effector`` control frame agree with ``panda.xml`` across joint
    configurations, and the joint limits match. The control frame is the contract that keeps
    ``ik_joints_from_episode`` inverting ``robot_state.ee_pose`` against the grasp site the sim
    measured; the limits are what ``DLSIKSolverWithLimits`` constrains each solve to.
    """
    joints = [f'joint{i}' for i in range(1, 8)]

    def compiled(spec):
        # A URDF carries ``end_effector`` as a frame-only body; resolve it to a readable site.
        if 'end_effector' not in {site.name for body in spec.bodies for site in body.sites}:
            spec.body('end_effector').add_site().name = 'end_effector'
        model = spec.compile()
        return model, mj.MjData(model)

    root = ET.fromstring(bundled_panda_model()['urdf'])  # drop meshes so the URDF compiles file-free
    for link in root.findall('.//link'):
        for el in link.findall('visual') + link.findall('collision'):
            link.remove(el)
    m_urdf, d_urdf = compiled(mj.MjSpec.from_string(ET.tostring(root, encoding='unicode')))
    m_mjcf, d_mjcf = compiled(mj.MjSpec.from_file(str(package_assets_path('assets/mujoco/panda.xml'))))

    qadr_urdf = [m_urdf.joint(n).qposadr.item() for n in joints]
    qadr_mjcf = [m_mjcf.joint(n).qposadr.item() for n in joints]
    ee_urdf = mj.mj_name2id(m_urdf, mj.mjtObj.mjOBJ_SITE, 'end_effector')
    ee_mjcf = mj.mj_name2id(m_mjcf, mj.mjtObj.mjOBJ_SITE, 'end_effector')
    for n in joints:
        np.testing.assert_allclose(m_urdf.joint(n).range, m_mjcf.joint(n).range, atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, len(joints))
        d_urdf.qpos[qadr_urdf] = q
        d_mjcf.qpos[qadr_mjcf] = q
        mj.mj_forward(m_urdf, d_urdf)
        mj.mj_forward(m_mjcf, d_mjcf)
        for link in (f'link{i}' for i in range(1, 8)):
            bu = mj.mj_name2id(m_urdf, mj.mjtObj.mjOBJ_BODY, link)
            bm = mj.mj_name2id(m_mjcf, mj.mjtObj.mjOBJ_BODY, link)
            np.testing.assert_allclose(d_urdf.xpos[bu], d_mjcf.xpos[bm], atol=1e-6)
            assert abs(float(np.dot(d_urdf.xquat[bu], d_mjcf.xquat[bm]))) > 1 - 1e-6
        np.testing.assert_allclose(d_urdf.site_xpos[ee_urdf], d_mjcf.site_xpos[ee_mjcf], atol=1e-6)
        np.testing.assert_allclose(d_urdf.site_xmat[ee_urdf], d_mjcf.site_xmat[ee_mjcf], atol=1e-6)
