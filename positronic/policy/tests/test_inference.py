from functools import partial

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.dataset.ds_writer_agent import Serializers
from positronic.drivers import roboarm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, Recover, Reset, to_wire
from positronic.drivers.roboarm.command import from_wire as cmd_from_wire
from positronic.geom import Rotation, Transform3D
from positronic.policy.inference import Inference, InferenceCommand
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler


class SpyPolicy:
    def __init__(self, command: roboarm.command.CommandType | None = None, target_grip: float = 0.33) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)
        self.last_obs: dict[str, object] | None = None

    def select_action(self, obs: dict[str, object]) -> dict[str, object]:
        self.last_obs = obs
        return {'robot_command': to_wire(self.command), 'target_grip': self.target_grip}

    def close(self) -> None:
        """Tests rely on Inference calling policy.close(); provide no-op."""
        return None


class StubPolicy:
    """Reusable policy stub for tests.

    Compatible with `positronic.policy.inference.Inference`: returns wire-format robot commands + target grip.
    """

    def __init__(self, command: roboarm.command.CommandType | None = None, target_grip: float = 0.33) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)
        self.last_obs: dict[str, object] | None = None
        self.observations: list[dict[str, object]] = []
        self.reset_calls = 0
        self.meta: dict[str, object] = {}

    def select_action(self, obs: dict[str, object]) -> dict[str, object]:
        self.last_obs = obs
        self.observations.append(obs)
        return {'robot_command': to_wire(self.command), 'target_grip': self.target_grip}

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        """Tests rely on Inference calling policy.close(); provide no-op."""
        return None


class FakeRobotState:
    def __init__(self, translation: np.ndarray, joints: np.ndarray, status: RobotStatus) -> None:
        self.ee_pose = Transform3D(translation=translation, rotation=Rotation.identity)
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


def make_robot_state(translation, joints, status=RobotStatus.AVAILABLE) -> FakeRobotState:
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
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    inference = Inference(policy)

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
        (partial(command_em.emit, InferenceCommand.START(task='stack-blocks')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    assert policy.last_obs is not None
    obs = policy.last_obs
    assert 'image.cam' in obs
    expected_pose = np.concatenate([robot_state.ee_pose.translation, robot_state.ee_pose.rotation.as_quat])
    np.testing.assert_allclose(obs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(obs['robot_state.q'], robot_state.q)
    np.testing.assert_allclose(obs['robot_state.dq'], np.zeros_like(robot_state.q))
    assert obs['grip'] == pytest.approx(0.25)
    assert obs['task'] == 'stack-blocks'

    command_msg = command_rx.read()
    assert command_msg is not None
    command = command_msg.data
    assert isinstance(command, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, pose.translation)
    np.testing.assert_allclose(command.pose.rotation.as_quat, pose.rotation.as_quat)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(0.33)


@pytest.mark.timeout(3.0)
def test_inference_waits_for_complete_inputs(world, clock):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    inference = Inference(policy)

    # Keep handles to the control system IO so we can drip feed partial data.
    frame_em = world.pair(inference.frames['image.cam'])
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

    # First send robot/grip without a frame (inference should block), then the full payload.
    # All sleeps are 0.0 so the priority-queue scheduler alternates fairly between
    # Inference (which yields Pass) and ManualDriver.
    driver = ManualDriver([
        (partial(command_em.emit, InferenceCommand.START(task='dummy-task')), 0.0),
        (partial(robot_em.emit, robot_state), 0.0),
        (partial(grip_em.emit, 0.25), 0.0),
        (assert_no_outputs, 0.0),  # still missing a frame
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.0),
        (None, 0.0),
    ])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=30)

    assert policy.last_obs is not None

    # Successful path should generate consistent Cartesian moves and grip commands.
    command_msg = command_rx.read()
    assert command_msg is not None
    assert isinstance(command_msg.data, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(command_msg.data.pose.translation, pose.translation)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(0.33)

    # Subsequent reads return the last value, so identity equality signals no new messages.
    assert command_rx.read() is command_msg
    assert grip_rx.read() is grip_msg


@pytest.mark.timeout(3.0)
def test_inference_reset_emits_reset_and_calls_policy_reset(world, clock):
    policy = StubPolicy()
    inference = Inference(policy)

    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)
    grip_rx = world.pair(inference.target_grip)

    driver = ManualDriver([(partial(command_em.emit, InferenceCommand.RESET()), 0.0), (None, 0.01)])

    scheduler = world.start([inference, driver])
    drive_scheduler(scheduler, clock=clock, steps=10)

    cmd_msg = command_rx.read()
    assert cmd_msg is not None
    assert isinstance(cmd_msg.data, Reset)

    grip_msg = grip_rx.read()
    assert grip_msg is not None
    assert grip_msg.data == pytest.approx(0.0)

    assert policy.reset_calls == 1


@pytest.mark.timeout(3.0)
def test_inference_clears_queue_on_reset_stepwise(world, clock):
    """Verify that RESET clears any pending actions in the queue (stepwise check)."""
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)

    class ChunkPolicy(StubPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter = 0

        def select_action(self, obs):
            self.counter += 1
            # Return chunk of 10 actions to ensure queue is not drained
            # Grips will be: 100, 101, ... 109 then 200, 201 ...
            return [
                {'robot_command': to_wire(self.command), 'target_grip': self.counter * 100.0 + i} for i in range(10)
            ]

    policy = ChunkPolicy(command=CartesianPosition(pose=pose))
    inference = Inference(policy)

    frame_em = world.pair(inference.frames['image.cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    grip_rx = world.pair(inference.target_grip)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    scheduler = world.start([inference])

    # 1. START
    command_em.emit(InferenceCommand.START(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    # 2. Emit Inputs -> Policy returns chunk 1 [100 ... 109]
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state)

    # Run enough steps to consume SOME but NOT ALL of current chunk
    # e.g. 3 steps -> emits 100, 101, 102. Queue has 103..109
    drive_scheduler(scheduler, clock=clock, steps=3)

    # Verify we are consuming the chunk
    val = grip_rx.read().data
    assert 100.0 <= val < 109.0, f'Expected value in chunk 1 range, got {val}'
    assert val > 100.0, 'Should have advanced past first element'

    # 3. RESET
    command_em.emit(InferenceCommand.RESET())
    drive_scheduler(scheduler, clock=clock, steps=2)

    # Verify Reset emitted (grip 0.0)
    val = grip_rx.read().data
    assert val == 0.0, f'Expected 0.0 (Reset), got {val}'

    # 4. START again
    command_em.emit(InferenceCommand.START(task='test'))
    # 5. Emit Inputs -> Policy returns chunk 2 [200 ... 209]
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state)

    # Run 2 steps: 1 to process inputs, 1 to emit first action of NEW chunk
    drive_scheduler(scheduler, clock=clock, steps=2)

    # 6. Check emission.
    # If queue cleared: should be 200.0 (first of new chunk)
    # If queue persisted: would be ~103.0 (next of unused old chunk)
    val = grip_rx.read().data
    assert val >= 200.0, f'Expected >= 200.0 (Start of New Chunk), got {val}. Queue clearing failed!'


@pytest.mark.timeout(3.0)
def test_inference_clears_queue_on_start_stepwise(world, clock):
    """Verify that START clears any pending actions in the queue."""
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)

    class ChunkPolicy(StubPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter = 0

        def select_action(self, obs):
            self.counter += 1
            # Return chunk of 10 actions
            return [
                {'robot_command': to_wire(self.command), 'target_grip': self.counter * 100.0 + i} for i in range(10)
            ]

    policy = ChunkPolicy(command=CartesianPosition(pose=pose))
    inference = Inference(policy)

    frame_em = world.pair(inference.frames['image.cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    grip_rx = world.pair(inference.target_grip)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    scheduler = world.start([inference])

    # 1. START
    command_em.emit(InferenceCommand.START(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    # 2. Emit Inputs -> Policy returns chunk 1 [100 ... 109]
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state)

    # Run enough steps to consume SOME but NOT ALL of current chunk
    # e.g. 3 steps -> emits 100, 101, 102. Queue has 103..109
    drive_scheduler(scheduler, clock=clock, steps=3)

    # Verify recent emission is from chunk 1
    val = grip_rx.read().data
    assert 100.0 <= val < 109.0

    # 3. START again (Restarting while running)
    command_em.emit(InferenceCommand.START(task='test-restart'))
    # This should clear the queue.
    # We drive 1 step to process the command.
    drive_scheduler(scheduler, clock=clock, steps=1)

    # 4. Emit Inputs -> Policy returns chunk 2 [200 ... 209]
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state)

    # Run 2 steps: 1 to select action (replenish queue), 1 to emit
    drive_scheduler(scheduler, clock=clock, steps=2)

    # 5. Check emission.
    # If queue cleared: should be 200.0 (first of new chunk)
    # If queue persisted: would be ~103.0 (remnant of chunk 1)
    val = grip_rx.read().data
    assert val >= 200.0, f'Expected >= 200.0 (Start of New Chunk), got {val}. Queue clearing on START failed!'


@pytest.mark.timeout(3.0)
def test_inference_recovers_from_error(world, clock):
    """Verify that inference detects ERROR, clears queue, sends Recover, and resumes after recovery."""
    policy = StubPolicy()
    inference = Inference(policy)

    frame_em = world.pair(inference.frames['image.cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)

    robot_state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    robot_state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)

    scheduler = world.start([inference])

    # 1. START inference
    command_em.emit(InferenceCommand.START(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    # 2. Emit ERROR state
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state_err)
    drive_scheduler(scheduler, clock=clock, steps=2)

    # Should have emitted a Recover command
    cmd_msg = command_rx.read()
    assert cmd_msg is not None
    assert isinstance(cmd_msg.data, Recover)

    # Policy should NOT have been called (only obs count = 0)
    assert len(policy.observations) == 0

    # 3. Now emit AVAILABLE state — inference should resume and call policy
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state_ok)
    drive_scheduler(scheduler, clock=clock, steps=3)

    assert len(policy.observations) > 0


@pytest.mark.timeout(3.0)
def test_inference_error_clears_pending_actions(world, clock):
    """Verify that ERROR clears any pending actions in the queue."""
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)

    class ChunkPolicy(StubPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter = 0

        def select_action(self, obs):
            self.counter += 1
            return [
                {'robot_command': to_wire(self.command), 'target_grip': self.counter * 100.0 + i} for i in range(10)
            ]

    policy = ChunkPolicy(command=CartesianPosition(pose=pose))
    inference = Inference(policy)

    frame_em = world.pair(inference.frames['image.cam'])
    robot_em = world.pair(inference.robot_state)
    grip_em = world.pair(inference.gripper_state)
    command_em = world.pair(inference.command)
    command_rx = world.pair(inference.robot_commands)
    grip_rx = world.pair(inference.target_grip)

    robot_state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    robot_state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)

    scheduler = world.start([inference])

    # 1. START + emit inputs -> policy returns chunk 1 [100..109]
    command_em.emit(InferenceCommand.START(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state_ok)
    drive_scheduler(scheduler, clock=clock, steps=3)

    # Verify consuming chunk 1
    val = grip_rx.read().data
    assert 100.0 <= val < 110.0

    # 2. Switch to ERROR state
    robot_em.emit(robot_state_err)
    drive_scheduler(scheduler, clock=clock, steps=2)

    # Should have sent Recover
    cmd_msg = command_rx.read()
    assert isinstance(cmd_msg.data, Recover)

    # 3. Switch back to AVAILABLE -> policy called again with chunk 2 [200..209]
    emit_ready_payload(frame_em, robot_em, grip_em, robot_state_ok)
    drive_scheduler(scheduler, clock=clock, steps=3)

    val = grip_rx.read().data
    assert val >= 200.0, f'Expected >= 200.0 (fresh chunk after error recovery), got {val}'


def test_recover_command_wire_roundtrip():
    """Verify Recover command serializes and deserializes correctly."""
    original = Recover()
    wire = to_wire(original)
    assert wire == {'type': 'recover'}
    restored = cmd_from_wire(wire)
    assert isinstance(restored, Recover)


def test_robot_state_serializer_records_error():
    """Verify Serializers.robot_state() includes .error field."""
    state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    result = Serializers.robot_state(state_ok)
    assert result is not None
    assert result['.error'] == 0

    state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)
    result = Serializers.robot_state(state_err)
    assert result is not None
    assert result['.error'] == 1

    state_reset = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.RESETTING)
    result = Serializers.robot_state(state_reset)
    assert result is None
