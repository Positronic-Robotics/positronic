"""What ``PackedState`` carries, and what it must carry across a process boundary."""

import numpy as np
import pytest

from positronic import geom
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.state import PackedState


@pytest.mark.parametrize('n_joints', [5, 6, 7])
def test_what_goes_in_comes_back_out(n_joints):
    state = PackedState(n_joints)
    q = np.arange(n_joints, dtype=np.float64) * 0.1
    dq = np.arange(n_joints, dtype=np.float64) * -0.01
    pose = geom.Transform3D(np.array([0.3, -0.2, 0.5]), geom.Rotation.from_rotvec(np.array([0.1, 0.2, 0.3])))

    state.encode(q, dq, pose, RobotStatus.BUSY)

    np.testing.assert_allclose(state.q, q, atol=1e-6)
    np.testing.assert_allclose(state.dq, dq, atol=1e-6)
    np.testing.assert_allclose(state.ee_pose.translation, pose.translation, atol=1e-6)
    np.testing.assert_allclose(state.ee_pose.rotation.as_quat, pose.rotation.as_quat, atol=1e-6)
    assert state.status is RobotStatus.BUSY


@pytest.mark.parametrize('n_joints', [5, 6, 7])
def test_the_state_says_how_to_build_it_again(n_joints):
    """Shared memory rebuilds the payload on the far side from ``instantiation_params``, so a state that
    does not name its joint count comes back the wrong size and reads another arm's numbers."""
    state = PackedState(n_joints)

    rebuilt = PackedState(*state.instantiation_params())

    assert rebuilt.n_joints == n_joints
    assert rebuilt.array.shape == state.array.shape


def test_a_reading_is_a_copy_of_what_the_buffer_holds():
    """The buffer is written again every tick, so a reader that kept a view would watch its own reading
    change under it."""
    state = PackedState(6)
    state.encode(np.zeros(6), np.zeros(6), geom.Transform3D(), RobotStatus.AVAILABLE)
    q = state.q

    state.encode(np.ones(6), np.zeros(6), geom.Transform3D(), RobotStatus.AVAILABLE)

    np.testing.assert_allclose(q, np.zeros(6))
