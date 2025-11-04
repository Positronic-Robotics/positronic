from functools import partial

import numpy as np
import pytest
import tqdm

import pimm
import positronic.cfg.simulator
import positronic.utils.s3 as pos3
from pimm.tests.testing import MockClock
from positronic import geom
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers import roboarm
from positronic.run_inference import Inference, InferenceCommand, main_sim
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler


class StubStateEncoder:
    def __init__(self) -> None:
        self.last_inputs: dict[str, object] | None = None
        self.last_image_keys: list[str] = []

    def encode(self, inputs: dict[str, object]) -> dict[str, object]:
        self.last_inputs = inputs
        self.last_image_keys = sorted([key for key in inputs if key.startswith('image.')])
        pose = inputs['robot_state.ee_pose']
        translation = pose[:3]
        quaternion = pose[3:7]
        return {
            'vision': np.array([[1.0, 2.0]], dtype=np.float32),
            'robot_translation': translation,
            'robot_quaternion': quaternion,
            'grip_value': inputs['grip'],
        }


class StubActionDecoder:
    def __init__(self) -> None:
        self.last_action: np.ndarray | None = None
        self.last_inputs: dict[str, object] | None = None
        self.pose = geom.Transform3D(
            translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=geom.Rotation.identity
        )
        self.grip = np.array(0.33, dtype=np.float32)

    def decode(self, action: np.ndarray, inputs: dict[str, object]) -> tuple[roboarm.command.CommandType, float]:
        self.last_action = np.copy(action)
        self.last_inputs = inputs
        decoded = np.asarray(action, dtype=np.float32).reshape(-1)

        translation = self.pose.translation
        if decoded.size >= 3:
            translation = decoded[:3]

        rotation = self.pose.rotation
        if decoded.size >= 7:
            quaternion = decoded[3:7]
            norm = np.linalg.norm(quaternion)
            if norm > 0:
                rotation = geom.Rotation.from_quat(quaternion / norm)

        if decoded.size >= 8:
            self.grip = np.array(decoded[7], dtype=np.float32)
        elif decoded.size >= 4:
            self.grip = np.array(decoded[3], dtype=np.float32)

        self.pose = geom.Transform3D(translation=np.array(translation, dtype=np.float32), rotation=rotation)
        return (roboarm.command.CartesianPosition(pose=self.pose), float(self.grip))


class SpyPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.last_obs: dict[str, object] | None = None

    def select_action(self, obs: dict[str, object]) -> np.ndarray:
        self.last_obs = obs
        return self.action


DEFAULT_STUB_POLICY_ACTION = np.array([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.33]], dtype=np.float32)


class StubPolicy:
    def __init__(self, action: np.ndarray | None = None) -> None:
        if action is None:
            action = DEFAULT_STUB_POLICY_ACTION.copy()
        if action.ndim == 1:
            action = action[np.newaxis, :]
        self.action = action
        self.last_obs: dict[str, object] | None = None
        self.observations: list[dict[str, object]] = []
        self.reset_calls = 0

    def select_action(self, obs: dict[str, object]) -> np.ndarray:
        self.last_obs = obs
        self.observations.append(obs)
        return self.action

    def reset(self) -> None:
        self.reset_calls += 1


def make_stub_observation_encoder() -> StubStateEncoder:
    return StubStateEncoder()


def make_stub_action_decoder() -> StubActionDecoder:
    return StubActionDecoder()


class FakeRobotState:
    def __init__(self, translation: np.ndarray, joints: np.ndarray, status: roboarm.RobotStatus) -> None:
        self.ee_pose = geom.Transform3D(translation=translation, rotation=geom.Rotation.identity)
        self.q = joints
        self.dq = np.zeros_like(joints)
        self.status = status


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
        yield w


def make_robot_state(translation, joints, status=roboarm.RobotStatus.AVAILABLE) -> FakeRobotState:
    translation = np.asarray(translation, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    return FakeRobotState(translation, joints, status)


def emit_ready_payload(frame_emitter, robot_emitter, grip_emitter, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.uint8)
    frame_adapter.array[:] = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_emitter.emit(frame_adapter)
    robot_emitter.emit(robot_state)
    grip_emitter.emit(0.25)


@pytest.mark.timeout(3.0)
def test_inference_emits_cartesian_move(world, clock):
    encoder = StubStateEncoder()
    decoder = StubActionDecoder()
    policy = SpyPolicy(action=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))

    inference = Inference(encoder, decoder, policy, inference_fps=15, task='stack-blocks')

    # Wire inference interfaces so we can inspect the produced commands and grip targets.
    frame_em = world.pair(inference.frames['image.cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)
    grip_rx = world.pair(inference.target_grip)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    # Provide a single coherent observation bundle then stop the world loop.
    driver = ManualDriver([
        (partial(command_em.emit, InferenceCommand.START()), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    assert policy.last_obs is not None
    obs = policy.last_obs
    np.testing.assert_allclose(obs['vision'], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(obs['robot_translation'], robot_state.ee_pose.translation)
    np.testing.assert_allclose(obs['robot_quaternion'], robot_state.ee_pose.rotation.as_quat)
    assert obs['grip_value'] == pytest.approx(0.25)
    assert obs['task'] == 'stack-blocks'

    assert encoder.last_inputs is not None
    inputs = encoder.last_inputs
    assert 'image.cam' in inputs
    expected_pose = np.concatenate([robot_state.ee_pose.translation, robot_state.ee_pose.rotation.as_quat])
    np.testing.assert_allclose(inputs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(inputs['robot_state.q'], robot_state.q)
    np.testing.assert_allclose(inputs['robot_state.dq'], np.zeros_like(robot_state.q))

    assert decoder.last_action is not None
    assert decoder.last_inputs is not None
    np.testing.assert_allclose(decoder.last_action, np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    np.testing.assert_allclose(decoder.last_inputs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(decoder.last_inputs['robot_state.q'], robot_state.q)

    command_msg = command_rx.read()
    assert command_msg is not None
    command = command_msg.data
    assert isinstance(command, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, decoder.pose.translation)
    np.testing.assert_allclose(command.pose.rotation.as_quat, decoder.pose.rotation.as_quat)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(float(decoder.grip))


@pytest.mark.timeout(3.0)
def test_inference_waits_for_complete_inputs(world, clock):
    encoder = StubStateEncoder()
    decoder = StubActionDecoder()
    policy = SpyPolicy(action=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32))

    inference = Inference(encoder, decoder, policy, inference_fps=15)

    # Keep handles to the control system IO so we can drip feed partial data.
    frame_em = world.pair(inference.frames['cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)
    grip_rx = world.pair(inference.target_grip)

    assert len(inference.frames) == 1

    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])

    def assert_no_outputs():
        assert command_rx.read() is None
        assert grip_rx.read() is None
        assert policy.last_obs is None

    # First send an empty frame, then the full payload and ensure only the latter produces actions.
    driver = ManualDriver([
        (partial(command_em.emit, InferenceCommand.START()), 0.0),
        (partial(frame_em.emit, pimm.shared_memory.NumpySMAdapter((0, 0, 0), np.uint8)), 0.01),
        (assert_no_outputs, 0.005),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=30)

    assert policy.last_obs is not None

    # Successful path should generate consistent Cartesian moves and grip commands.
    command_msg = command_rx.read()
    assert command_msg is not None
    assert isinstance(command_msg.data, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(command_msg.data.pose.translation, decoder.pose.translation)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(float(decoder.grip))

    # Subsequent reads return the last value, so identity equality signals no new messages.
    assert command_rx.read() is command_msg
    assert grip_rx.read() is grip_msg


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
