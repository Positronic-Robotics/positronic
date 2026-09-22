"""Delta targets use control observations, including after inference waits and between chunk actions."""

from threading import Event

import numpy as np
import pytest

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command
from positronic.policy.action import DeltaToAbsolute
from positronic.policy.base import Step
from positronic.policy.codec import ChangeEEFrame
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec
from positronic.simulator.env_server import protocol
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.simulator.mujoco.sim import MujocoSim

QUAT = geom.Rotation.Representation.QUAT


def test_joint_conversion_preserves_other_commands_and_does_not_mutate_input():
    codec = from_spec(DeltaToAbsolute().to_spec())
    assert isinstance(codec, DeltaToAbsolute)
    joints = np.array([0.1, 0.2, 0.3])
    delta = np.array([0.02, -0.03, 0.04])
    mode = command.PositionControl()
    action = {keys.ROBOT_COMMAND: command.JointDelta(delta, mode), keys.TARGET_GRIP: 0.7}
    result = codec.decode(action, obs={keys.JOINTS: joints})
    assert isinstance(result[keys.ROBOT_COMMAND], command.JointPosition)
    np.testing.assert_allclose(result[keys.ROBOT_COMMAND].positions, [0.12, 0.17, 0.34])
    assert result[keys.ROBOT_COMMAND].mode is mode
    assert result[keys.TARGET_GRIP] == 0.7
    assert isinstance(action[keys.ROBOT_COMMAND], command.JointDelta)
    np.testing.assert_array_equal(joints, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(delta, [0.02, -0.03, 0.04])


@pytest.mark.parametrize('composed', [False, True])
def test_cartesian_conversion_preserves_world_rotation_and_tool_frame(composed):
    pose = geom.Transform3D([0.3, 0.1, 0.4], geom.Rotation.from_euler([0.2, -0.3, 0.5]))
    frame = geom.Transform3D([0.0, 0.0, 0.08], geom.Rotation.from_euler([0.4, 0.2, -0.1]))
    delta = geom.Transform3D([0.01, -0.02, 0.03], geom.Rotation.from_euler([-0.1, 0.3, 0.2]))
    mode = command.PositionControl()
    outer, inner = DeltaToAbsolute(), ChangeEEFrame(frame)
    conversion = outer | inner if composed else Sequential(outer, inner)
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)

    def infer(obs):
        np.testing.assert_allclose(obs[keys.EE_POSE], (pose * frame).as_vector(QUAT))
        return {keys.ROBOT_COMMAND: command.CartesianDelta(delta, mode=mode)}

    run = runtime.start(Sequential(conversion), infer)
    try:
        output = run.send({keys.EE_POSE: pose.as_vector(QUAT)})
        assert isinstance(output, dict)
        result = output[keys.ROBOT_COMMAND]
        actual = (result.pose * frame).as_matrix
        expected = (pose * frame).as_matrix.copy()
        expected[:3, 3] += delta.translation
        expected[:3, :3] = delta.as_matrix[:3, :3] @ expected[:3, :3]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert result.mode is mode
    finally:
        runtime.close()
        run.close()


@pytest.mark.parametrize('kind', ['joint', 'cartesian'])
def test_scheduler_anchors_each_action_when_emitted_and_does_not_reapply_during_gaps(kind):
    now_ns = 0
    release = Event()
    delta = (
        command.JointDelta(np.array([0.1]))
        if kind == 'joint'
        else command.CartesianDelta(geom.Transform3D([0.1, 0, 0], geom.Rotation.identity))
    )

    def obs(position):
        return {
            keys.JOINTS: np.array([position]),
            keys.EE_POSE: geom.Transform3D([position, 0, 0], geom.Rotation.identity).as_vector(QUAT),
        }

    def infer(_obs):
        assert release.wait(5), 'test did not release inference'
        return [{keys.ROBOT_COMMAND: delta}, {keys.ROBOT_COMMAND: delta}]

    runtime = Executor(lambda: now_ns, simulated=True, charge_inference_time=False)
    run = runtime.start(Sequential(DeltaToAbsolute(), ChunkedSchedule(10)), infer)

    def resume(position):
        result = run.send(obs(position))
        assert isinstance(result, Step)
        return result

    try:
        assert resume(0).commands == {}
        release.set()
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        first = resume(1)
        now_ns = 50_000_000
        assert resume(1.05).commands == {}
        now_ns = 100_000_000
        second = resume(2)
        now_ns = 150_000_000
        assert resume(2.05).commands == {}
        for step, expected in ((first, 1.1), (second, 2.1)):
            target = step.commands[keys.ROBOT_COMMAND]
            position = target.positions[0] if kind == 'joint' else target.pose.translation[0]
            assert position == pytest.approx(expected)
        assert first.resume_at_ns == 100_000_000
        assert second.resume_at_ns == 200_000_000
        release.clear()
        now_ns = 200_000_000
        assert resume(3).commands == {}
        now_ns = 300_000_000
        assert resume(3.5).commands == {}
        release.set()
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        target = resume(4).commands[keys.ROBOT_COMMAND]
        position = target.positions[0] if kind == 'joint' else target.pose.translation[0]
        assert position == pytest.approx(4.1)
    finally:
        release.set()
        runtime.close()
        run.close()


def test_shared_codec_uses_each_calls_observation_and_preserves_absolute_steps():
    codec = DeltaToAbsolute(command_key='arm', joints_key='joints')
    delta = command.JointDelta(np.array([0.2]))
    inner = codec.wrap(lambda obs: {'arm': delta})

    def nested(obs):
        np.testing.assert_allclose(inner({'joints': np.array([10.0])})['arm'].positions, [10.2])
        return Step({'arm': delta}, 100)

    result = codec.wrap(nested)({'joints': np.array([1.0])})
    np.testing.assert_allclose(result.commands['arm'].positions, [1.2])
    assert result.resume_at_ns == 100
    absolute = {'arm': command.JointPosition(np.array([3.0]))}
    assert codec.decode(absolute) is absolute
    assert codec.decode({}) == {}


def test_conversion_rejects_missing_state_wrong_joint_count_and_whole_chunks():
    codec = DeltaToAbsolute()
    data = {keys.ROBOT_COMMAND: command.JointDelta(np.ones(7))}
    with pytest.raises(ValueError, match='observation'):
        codec.decode(data)
    with pytest.raises(KeyError, match=keys.JOINTS):
        codec.decode(data, obs={})
    with pytest.raises(ValueError, match='shape'):
        codec.decode(data, obs={keys.JOINTS: np.ones(1)})
    with pytest.raises(ValueError, match='scheduler'):
        codec.decode([data], obs={keys.JOINTS: np.ones(7)})


def test_remote_adapter_keeps_the_absolute_target_during_gaps():
    adapter = LiberoAdapter(camera_dict={})
    codec = DeltaToAbsolute()
    target = codec.decode({keys.ROBOT_COMMAND: command.JointDelta(np.array([0.1]))}, obs={keys.JOINTS: np.array([1.0])})
    first = adapter.action({keys.ROBOT_COMMAND: pimm.Message(target[keys.ROBOT_COMMAND])})
    for _ in range(10):
        held = adapter.action({keys.ROBOT_COMMAND: None})
        np.testing.assert_array_equal(
            held[keys.ROBOT_COMMAND][protocol.COMMAND_JOINT_POS], first[keys.ROBOT_COMMAND][protocol.COMMAND_JOINT_POS]
        )


@pytest.mark.parametrize('kind', ['joint', 'cartesian'])
def test_mujoco_motion_matches_driver_conversion_with_fresh_observations(kind):
    """Compare physics trajectories, including a gap after the last command, without a model or renderer."""
    reference = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders=())
    candidate = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders=())
    for sim in (reference, candidate):
        sim.step(0.2)
    codec = DeltaToAbsolute()
    reference_joints, candidate_joints = [], []
    for tick in range(300):
        if tick in (0, 50, 100):
            delta = (
                command.JointDelta(np.array([0.03, 0, 0, 0, 0, 0, 0]))
                if kind == 'joint'
                else command.CartesianDelta(geom.Transform3D([0.0, 0, 0.003], geom.Rotation.identity))
            )
            obs = {keys.JOINTS: candidate._q.copy(), keys.EE_POSE: candidate._ee_pose.as_vector(QUAT)}
            absolute = codec.decode({keys.ROBOT_COMMAND: delta}, obs=obs)[keys.ROBOT_COMMAND]
            reference._apply_command(delta)
            candidate._apply_command(absolute)
        reference.step()
        candidate.step()
        reference_joints.append(reference._q.copy())
        candidate_joints.append(candidate._q.copy())
    assert not reference._error and not candidate._error
    np.testing.assert_allclose(candidate_joints, reference_joints, rtol=0, atol=1e-10)
    assert np.max(np.abs(candidate_joints[-1] - candidate_joints[100])) > 1e-4
