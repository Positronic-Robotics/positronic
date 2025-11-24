import numpy as np
import pytest
import tqdm

import positronic.cfg.simulator
import positronic.utils.s3 as pos3
from positronic.dataset.local_dataset import LocalDataset
from positronic.inference import main_sim
from positronic.policy.tests.test_inference import StubPolicy, make_stub_action_decoder, make_stub_observation_encoder


# This integration test intentionally exercises the current `main_sim` wiring end-to-end.
@pytest.mark.timeout(30.0)
def test_main_sim_emits_commands_and_records_dataset(tmp_path, monkeypatch):
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

    observation_encoder = make_stub_observation_encoder()
    action_decoder = make_stub_action_decoder()
    policy = StubPolicy()

    camera_dict = {'image.handcam_left': 'handcam_left_ph'}
    loaders = [
        cfg(seed=idx) if idx in (2, 4) else cfg()
        for idx, cfg in enumerate(positronic.cfg.simulator.stack_cubes_loaders)
    ]

    with pos3.mirror():
        main_sim(
            mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
            observation_encoder=observation_encoder,
            action_decoder=action_decoder,
            policy=policy,
            loaders=loaders,
            camera_fps=10,
            policy_fps=10,
            simulation_time=0.4,
            camera_dict=camera_dict,
            task='integration-test',
            output_dir=str(tmp_path),
            show_gui=False,
            num_iterations=1,
        )

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1

    episode = ds[0]
    signals = episode.signals
    assert 'robot_commands.pose' in signals
    assert 'target_grip' in signals

    camera_signals = [name for name in signals if 'handcam_left' in name]
    assert camera_signals, f'Expected camera signal for handcam_left, found keys: {list(signals)}'
    camera_signal = signals[camera_signals[0]]
    camera_samples = list(camera_signal)
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
    assert isinstance(last_obs['vision'], np.ndarray)
    assert last_obs['vision'].shape[0] == 1
    assert observation_encoder.last_inputs is not None
    assert any('handcam_left' in key for key in observation_encoder.last_image_keys)
