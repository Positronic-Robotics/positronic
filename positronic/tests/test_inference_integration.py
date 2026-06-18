import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import pos3
import pytest
import tqdm

import positronic.cfg.simulator
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm import bundled_panda_model
from positronic.inference import main
from positronic.policy.tests.test_harness import StubPolicy
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

    camera_dict = {'image.wrist': 'handcam_left_ph'}

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
            embodiment=ev.embodiment,
            task=ev.task,
            policy=policy,
            trials=[{'eval.trial_index': i, 'eval.seed': 100 + i} for i in range(2)],
            output_dir=str(tmp_path),
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
    assert 'image.wrist' in signals
    # Privileged ground truth: the full sim state is recorded as a time-series signal.
    assert 'sim_state.mjSTATE_INTEGRATION' in signals

    camera_samples = list(signals['image.wrist'])
    assert camera_samples, 'Camera signal for handcam_left is empty'
    first_image, _ = camera_samples[0]
    assert isinstance(first_image, np.ndarray)

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
    assert grip_values[0] == pytest.approx(0.33, rel=1e-2, abs=1e-2)
    assert np.all(np.diff([ts for _, ts in grip_samples]) > 0) or len(grip_samples) == 1

    assert policy.observations, 'Policy did not receive any observations'
    last_obs = policy.observations[-1]
    assert isinstance(last_obs['image.wrist'], np.ndarray)
    assert 'robot_state.ee_pose' in last_obs
    # The task's instruction is injected by the harness (no longer carried by the driver).
    assert last_obs['task'] == 'integration-test'
    assert last_obs['descriptor'] == 'mujoco.franka'


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
