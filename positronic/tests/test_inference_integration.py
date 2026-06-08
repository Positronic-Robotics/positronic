import numpy as np
import pos3
import pytest
import tqdm

import positronic.cfg.embodiment
import positronic.cfg.simulator
from positronic.dataset.local_dataset import LocalDataset
from positronic.inference import main, timed
from positronic.policy.tests.test_harness import StubPolicy
from positronic.simulator.mujoco.sim import FullSimState


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
    loaders = positronic.cfg.simulator.stack_cubes_loaders()
    for idx, loader in enumerate(loaders):
        if idx in (2, 4):
            loader.seed = idx

    with pos3.mirror():
        embodiment = positronic.cfg.embodiment.sim.override(
            mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
            loaders=loaders,
            camera_fps=10,
            camera_dict=camera_dict,
            observers={'sim_state': FullSimState()},
        )()
        main(
            embodiment=embodiment,
            policy=policy,
            driver=timed.override(simulation_time=0.4, task='integration-test', show_gui=False, num_iterations=1)(),
            output_dir=str(tmp_path),
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1

    episode = ds[0]
    signals = episode.signals
    assert 'robot_commands.pose' in signals
    assert 'target_grip' in signals
    assert 'image.wrist' in signals
    # Privileged ground truth: the full sim state is recorded as a time-series signal.
    assert 'sim_state.mjSTATE_FULLPHYSICS' in signals

    camera_samples = list(signals['image.wrist'])
    assert camera_samples, 'Camera signal for handcam_left is empty'
    first_image, _ = camera_samples[0]
    assert isinstance(first_image, np.ndarray)

    pose_signal = signals['robot_commands.pose']
    pose_samples = list(pose_signal)
    assert pose_samples, 'robot_commands.pose signal is empty'
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
    assert 'task' in last_obs
    assert last_obs['descriptor'] == 'mujoco.franka'


def test_sim_embodiment_records_full_sim_state():
    """The stack_cubes eval records the privileged full sim state, not live success observers.

    Guards the production observer default directly (the heavyweight e2e test above wires its own
    observers, so it cannot catch a regression of the ``sim`` embodiment config).
    """
    observers = positronic.cfg.embodiment.sim.kwargs['observers']
    assert set(observers) == {'sim_state'}
    assert isinstance(observers['sim_state'], FullSimState)
