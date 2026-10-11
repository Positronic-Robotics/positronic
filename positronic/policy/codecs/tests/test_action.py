import json

import numpy as np
import pytest
from positronic_model_server import serialization
from positronic_model_server.spec import ARGS, component

import positronic.drivers.roboarm.command as cmd_module
from positronic import keys as obs_keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.command import Impedance, JointDelta
from positronic.geom import Rotation
from positronic.policy.codecs import ACTION, JointPositionAction, SetControlMode, UnpackActionChunk
from positronic.policy.codecs.action import AbsoluteJointsAction, AbsolutePositionAction, IKJointsAction
from positronic.policy.spec import from_spec


@pytest.mark.parametrize('horizon', [0, 1, 40])
@pytest.mark.parametrize('wire_round_trip', [False, True])
def test_unpack_action_chunk_selects_fields_and_preserves_every_timestep(horizon, wire_round_trip):
    description = component(
        'unpack_action_chunk', fields={'joints': [0, 'robot.q'], 'grip': [0, 'robot/grip']}, squeeze_dims=1
    )
    codec = from_spec(json.loads(json.dumps(description)))
    assert isinstance(codec, UnpackActionChunk)
    joints = np.arange(horizon * 7, dtype=np.float32).reshape(1, horizon, 7)
    grip = np.arange(horizon, dtype=np.float64).reshape(1, horizon, 1)
    result = ({'robot.q': joints, 'robot/grip': grip, 'unused': np.zeros((2, 9))}, {'timing': 0.1})
    if wire_round_trip:
        result = serialization.deserialise(serialization.serialise(result))

    decoded = codec.decode(result)

    assert len(decoded) == horizon
    for index, action in enumerate(decoded):
        assert set(action) == {'joints', 'grip'}
        np.testing.assert_array_equal(action['joints'], joints[0, index])
        np.testing.assert_array_equal(action['grip'], grip[0, index])
        assert action['joints'].dtype == np.float32
        assert action['grip'].dtype == np.float64
    assert codec.to_spec() == description


@pytest.mark.parametrize('squeeze_dims', [0, 2])
def test_unpack_action_chunk_accepts_a_root_array_and_keeps_trailing_dimensions(squeeze_dims):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
    batched = values.reshape((1,) * squeeze_dims + values.shape)
    decoded = UnpackActionChunk({ACTION: []}, squeeze_dims=squeeze_dims).decode(batched)

    assert len(decoded) == 3
    for action, expected in zip(decoded, values, strict=True):
        np.testing.assert_array_equal(action[ACTION], expected)
    scalar_steps = UnpackActionChunk({'value': ['values']}).decode({'values': np.array([1, 2])})
    assert scalar_steps == [{'value': 1}, {'value': 2}]


@pytest.mark.parametrize('values', [np.zeros((2, 3, 7)), np.zeros((0, 3, 7)), np.zeros(1), [[1, 2, 3]]])
def test_unpack_action_chunk_requires_singleton_batch_dimensions_and_a_time_axis(values):
    with pytest.raises(ValueError, match='leading size-one dimensions and a time axis'):
        UnpackActionChunk({'value': []}, squeeze_dims=1).decode(values)


def test_unpack_action_chunk_requires_matching_horizons():
    codec = UnpackActionChunk({'joints': ['q'], 'grip': ['grip']})
    with pytest.raises(ValueError, match='share one horizon'):
        codec.decode({'q': np.zeros((3, 7)), 'grip': np.zeros((2, 1))})


def test_unpack_action_chunk_owns_its_paths_and_requires_selected_fields():
    fields = {'value': [0, 'input']}
    codec = UnpackActionChunk(fields)
    fields['value'][1] = 'changed'
    description = codec.to_spec()
    description[ARGS]['fields']['value'][1] = 'changed'

    assert codec.decode([{'input': np.array([5])}]) == [{'value': 5}]
    with pytest.raises(KeyError, match='input'):
        codec.decode([{'unselected': 1}])


@pytest.mark.parametrize('squeeze_dims', [-1, 1.5, True])
def test_unpack_action_chunk_rejects_invalid_dimensions(squeeze_dims):
    with pytest.raises(ValueError, match='squeeze_dims'):
        UnpackActionChunk({'value': []}, squeeze_dims=squeeze_dims)


@pytest.mark.parametrize(
    'fields', [{}, {'value': 'q'}, {'value': {'q': 0}}, {'value': [False]}, {'value': [0.5]}, {1: ['q']}]
)
def test_unpack_action_chunk_rejects_invalid_field_paths(fields):
    with pytest.raises(ValueError, match='field'):
        UnpackActionChunk(fields)


def test_unpack_action_chunk_preserves_observations_and_training_columns():
    codec = UnpackActionChunk({'value': ['prediction']})
    observation = {'image': np.zeros((2, 3, 3), dtype=np.uint8), 'state': np.zeros(7)}
    episode = EpisodeContainer({'value': DummySignal([0], [1])}, meta={'label': 'test'})

    assert codec.encode(observation) is observation
    assert codec.training_encoder(episode) is episode
    assert codec.meta == {}


@pytest.mark.parametrize('num_joints', [6, 7])
def test_joint_position_action_preserves_values_and_does_not_threshold_grip(num_joints):
    codec = JointPositionAction('prediction.joints', 'prediction.grip', num_joints)
    restored = from_spec(json.loads(json.dumps(codec.to_spec())))
    assert isinstance(restored, JointPositionAction)
    positions = np.arange(num_joints, dtype=np.float32)
    decoded = restored.decode({'prediction.joints': positions, 'prediction.grip': np.array([0.25])})
    np.testing.assert_array_equal(decoded[obs_keys.ROBOT_COMMAND].positions, positions)
    assert decoded[obs_keys.ROBOT_COMMAND].positions.dtype == positions.dtype
    assert decoded[obs_keys.ROBOT_COMMAND].mode is None
    assert decoded[obs_keys.TARGET_GRIP] == 0.25


@pytest.mark.parametrize('joints, grip', [(np.zeros(6), [0]), (np.zeros(7), [0, 1])])
def test_joint_position_action_requires_the_configured_joint_count_and_one_grip(joints, grip):
    with pytest.raises(ValueError):
        JointPositionAction('q', 'g').decode({'q': joints, 'g': grip})


def test_absolute_position_action_encode_decode_quat():
    ts = [1000, 2000]
    q = [Rotation.identity for _ in ts]
    t = [np.array([0.1, -0.2, 0.3], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    pose = [np.concatenate([t[i], q[i].as_quat]).astype(np.float32) for i in range(len(ts))]

    ep = EpisodeContainer({obs_keys.TARGET_EE_POSE: DummySignal(ts, pose), obs_keys.TARGET_GRIP: DummySignal(ts, g)})

    act = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, obs_keys.TARGET_GRIP, Rotation.Representation.QUAT)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({ACTION: vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded[obs_keys.TARGET_GRIP]
    assert isinstance(command, cmd_module.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, t[0], atol=1e-6)
    np.testing.assert_allclose(command.pose.rotation.as_quat, q[0].as_quat, atol=1e-6)
    assert np.isclose(target_grip, g[0])


def test_absolute_joints_action_encode_decode():
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    ep = EpisodeContainer({obs_keys.TARGET_JOINTS: DummySignal(ts, joints), obs_keys.TARGET_GRIP: DummySignal(ts, g)})

    act = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 7 joints + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({ACTION: vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded[obs_keys.TARGET_GRIP]
    assert isinstance(command, cmd_module.JointPosition)
    np.testing.assert_allclose(command.positions, joints[0], atol=1e-6)
    assert np.isclose(target_grip, g[0])


IMPEDANCE = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)


class TestSetControlMode:
    def test_wire_description_restores_the_control_mode(self):
        restored = from_spec(json.loads(json.dumps(SetControlMode(IMPEDANCE).to_spec())))
        assert isinstance(restored, SetControlMode)
        decoded = restored.decode({obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))})
        assert decoded[obs_keys.ROBOT_COMMAND].mode == IMPEDANCE

    def test_rejects_a_command_as_the_control_mode(self):
        with pytest.raises(ValueError, match='control mode'):
            SetControlMode(cmd_module.to_wire(cmd_module.JointPosition(np.zeros(7))))

    def test_every_command_in_a_chunk_carries_the_mode(self):
        chunk = [
            {obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))},
            {obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.ones(7))},
            {obs_keys.TARGET_GRIP: 0.5},
        ]
        decoded = SetControlMode(IMPEDANCE).decode(chunk)
        assert isinstance(decoded, list)
        for action in decoded[:2]:
            assert isinstance(action, dict)
            assert action[obs_keys.ROBOT_COMMAND].mode == IMPEDANCE
        assert obs_keys.ROBOT_COMMAND not in decoded[2]

    def test_a_single_action_carries_the_mode(self):
        decoded = SetControlMode(IMPEDANCE).decode({obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))})
        assert isinstance(decoded, dict)
        assert decoded[obs_keys.ROBOT_COMMAND].mode == IMPEDANCE

    def test_every_arm_channel_is_stamped(self):
        """A bimanual action names a channel per arm, and both execute under the mode."""
        action = {
            f'{obs_keys.ROBOT_COMMAND}.left': JointDelta(velocities=np.zeros(7)),
            f'{obs_keys.ROBOT_COMMAND}.right': JointDelta(velocities=np.ones(7)),
            obs_keys.TARGET_JOINTS: np.zeros(7),  # in the command family by name, but a vector
            obs_keys.TARGET_GRIP: 0.5,
        }
        decoded = SetControlMode(IMPEDANCE).decode(action)
        assert isinstance(decoded, dict)
        assert decoded[f'{obs_keys.ROBOT_COMMAND}.left'].mode == IMPEDANCE
        assert decoded[f'{obs_keys.ROBOT_COMMAND}.right'].mode == IMPEDANCE
        np.testing.assert_array_equal(decoded[obs_keys.TARGET_JOINTS], np.zeros(7))


def test_non_deliverable_codec_is_rejected():
    with pytest.raises(NotImplementedError, match='IKJointsAction'):
        IKJointsAction(solver_cls=None).to_spec()
