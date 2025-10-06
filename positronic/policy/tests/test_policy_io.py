import numpy as np
import pytest

from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.geom import Rotation, Transform3D
from positronic.policy.action import AbsolutePositionAction, RelativeTargetPositionAction
from positronic.policy.observation import ObservationEncoder


def test_observation_encode_images_and_state_shapes():
    # Image matches target size; ensures no resampling artifacts in assertions
    h, w = 6, 8
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    enc = ObservationEncoder(state_features=['a', 'b'], left=('left.image', (w, h)))
    obs = enc.encode({'left.image': img, 'a': [1, 2], 'b': 3.0})

    assert 'observation.images.left' in obs and 'observation.state' in obs
    left = obs['observation.images.left']
    state = obs['observation.state']

    assert left.shape == (1, 3, h, w)
    assert left.dtype == np.float32
    assert np.isclose(left.max(), 1.0) and np.isclose(left.min(), 1.0)

    assert state.shape == (1, 3)
    np.testing.assert_allclose(state[0], np.array([1, 2, 3], dtype=np.float32))


def test_observation_encode_missing_or_bad_images_raise():
    enc = ObservationEncoder(state_features=[], left=('left.image', (8, 6)))
    with pytest.raises(KeyError):  # Missing key
        enc.encode({})

    with pytest.raises(ValueError):  # Wrong shape
        enc.encode({'left.image': np.zeros((8, 8), dtype=np.uint8)})


def test_absolute_position_action_encode_decode_quat():
    # Identity rotation, known translation/grip
    ts = [1000, 2000]
    q = [Rotation.identity for _ in ts]
    t = [np.array([0.1, -0.2, 0.3], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    pose = [np.concatenate([t[i], q[i].as_quat]).astype(np.float32) for i in range(len(ts))]

    ep = EpisodeContainer(
        signals={
            'robot_commands.pose': DummySignal(ts, pose),
            'target_grip': DummySignal(ts, g),
        }
    )

    act = AbsolutePositionAction('robot_commands.pose', 'target_grip', Rotation.Representation.QUAT)
    sig = act.encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    out = act.decode(vec, inputs={})
    assert isinstance(out['target_robot_position'], Transform3D)
    np.testing.assert_allclose(out['target_robot_position'].translation, t[0], atol=1e-6)
    np.testing.assert_allclose(out['target_robot_position'].rotation.as_quat, q[0].as_quat, atol=1e-6)
    assert np.isclose(out['target_grip'], g[0])


def test_relative_target_position_action_encode_decode_quat():
    ts = [1000]
    q_cur = [Rotation.identity]
    # 90 deg about X: w=cos(45)=sqrt(2)/2, x=sin(45)=sqrt(2)/2
    q_tgt = [Rotation(np.sqrt(0.5), np.sqrt(0.5), 0, 0)]
    t_cur = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    t_tgt = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
    g_tgt = [0.7]

    cur_pose = [np.concatenate([t_cur[0], q_cur[0].as_quat]).astype(np.float32)]
    tgt_pose = [np.concatenate([t_tgt[0], q_tgt[0].as_quat]).astype(np.float32)]

    ep = EpisodeContainer(
        signals={
            'robot_state.ee_pose': DummySignal(ts, cur_pose),
            'robot_commands.pose': DummySignal(ts, tgt_pose),
            'target_grip': DummySignal(ts, g_tgt),
        }
    )

    act = RelativeTargetPositionAction(Rotation.Representation.QUAT)
    vec = list(act.encode_episode(ep))[0][0]
    # First 4 should match target quaternion since current is identity
    np.testing.assert_allclose(vec[:4], q_tgt[0].as_quat, atol=1e-6)
    assert vec.dtype == np.float32
    # Current implementation encodes translation as (current - target)
    np.testing.assert_allclose(vec[4:7], t_cur[0] - t_tgt[0], atol=1e-6)
    assert np.isclose(vec[7], g_tgt[0])

    out = act.decode(
        vec,
        inputs={'robot_state.ee_pose': np.concatenate([t_cur[0], q_cur[0].as_quat])},
    )
    assert isinstance(out['target_robot_position'], Transform3D)
    # Decode applies diff to current translation
    np.testing.assert_allclose(out['target_robot_position'].translation, t_cur[0] + vec[4:7], atol=1e-6)
