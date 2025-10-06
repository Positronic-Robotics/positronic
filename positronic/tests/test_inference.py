from functools import partial

import numpy as np
import pytest
import torch

import pimm
from pimm.tests.testing import MockClock
from positronic import geom
from positronic.drivers import roboarm
from positronic.run_inference import Inference, InferenceCommand
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler


class StubStateEncoder:
    def __init__(self) -> None:
        self.last_inputs: dict[str, object] | None = None

    def encode(self, inputs: dict[str, object]) -> dict[str, object]:
        self.last_inputs = inputs
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

    def decode(self, action: np.ndarray, inputs: dict[str, object]) -> dict[str, object]:
        self.last_action = np.copy(action)
        self.last_inputs = inputs
        return {
            'target_robot_position': self.pose,
            'target_grip': self.grip,
        }


class SpyPolicy:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action
        self.device: str | None = None
        self.last_obs: dict[str, torch.Tensor | object] | None = None

    def to(self, device: str):
        self.device = device
        return self

    def select_action(self, obs: dict[str, torch.Tensor | object]) -> torch.Tensor:
        self.last_obs = obs
        return self.action


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
    frame_emitter.emit({'image': np.zeros((2, 2, 3), dtype=np.uint8)})
    robot_emitter.emit(robot_state)
    grip_emitter.emit(0.25)


@pytest.mark.timeout(3.0)
def test_inference_emits_cartesian_move(world, clock):
    encoder = StubStateEncoder()
    decoder = StubActionDecoder()
    policy = SpyPolicy(action=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32))

    inference = Inference(encoder, decoder, 'cpu', policy, inference_fps=15, task='stack-blocks')

    # Wire inference interfaces so we can inspect the produced commands and grip targets.
    frame_em = world.pair(inference.frames['cam'])
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

    assert policy.device == 'cpu'
    assert policy.last_obs is not None
    obs = policy.last_obs
    assert obs['vision'].device.type == 'cpu'
    np.testing.assert_allclose(obs['vision'].cpu().numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(obs['robot_translation'].cpu().numpy(), robot_state.ee_pose.translation)
    np.testing.assert_allclose(obs['robot_quaternion'].cpu().numpy(), robot_state.ee_pose.rotation.as_quat)
    assert obs['grip_value'].item() == pytest.approx(0.25)
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
    np.testing.assert_allclose(decoder.last_action, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    np.testing.assert_allclose(decoder.last_inputs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(decoder.last_inputs['robot_state.q'], robot_state.q)

    command_msg = command_rx.read()
    assert command_msg is not None
    command = command_msg.data
    assert isinstance(command, roboarm.command.CartesianMove)
    np.testing.assert_allclose(command.pose.translation, decoder.pose.translation)
    np.testing.assert_allclose(command.pose.rotation.as_quat, decoder.pose.rotation.as_quat)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(float(decoder.grip))


@pytest.mark.timeout(3.0)
def test_inference_skips_when_robot_is_moving(world, clock):
    encoder = StubStateEncoder()
    decoder = StubActionDecoder()
    policy = SpyPolicy(action=torch.tensor([[0.9]], dtype=torch.float32))

    inference = Inference(encoder, decoder, 'cpu', policy, inference_fps=15)

    # Capture IO to verify that nothing is emitted while the robot is busy.
    frame_em = world.pair(inference.frames['cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)
    grip_rx = world.pair(inference.target_grip)

    robot_state = make_robot_state([0.5, 0.1, 0.2], [0.4, 0.2, 0.1], status=roboarm.RobotStatus.MOVING)

    # Feed an otherwise valid payload but keep status at MOVING so inference must wait.
    driver = ManualDriver([
        (partial(command_em.emit, InferenceCommand.START()), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    assert policy.last_obs is None
    assert decoder.last_action is None

    assert command_rx.read() is None
    assert grip_rx.read() is None


@pytest.mark.timeout(3.0)
def test_inference_waits_for_complete_inputs(world, clock):
    encoder = StubStateEncoder()
    decoder = StubActionDecoder()
    policy = SpyPolicy(action=torch.tensor([[0.5, 0.6, 0.7, 0.8]], dtype=torch.float32))

    inference = Inference(encoder, decoder, 'cpu', policy, inference_fps=15)

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
        (partial(frame_em.emit, {}), 0.01),
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
    assert isinstance(command_msg.data, roboarm.command.CartesianMove)
    np.testing.assert_allclose(command_msg.data.pose.translation, decoder.pose.translation)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(float(decoder.grip))

    # Subsequent reads return the last value, so identity equality signals no new messages.
    assert command_rx.read() is command_msg
    assert grip_rx.read() is grip_msg
