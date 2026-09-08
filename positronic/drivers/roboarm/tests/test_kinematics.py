"""What ``MjcfKinematics`` answers, pinned to what the YAM driver answered before it shared the class.

``yam_kinematics_goldens.npz`` was generated from the YAM's own ``_Kinematics`` at `cab986af`: ten random
joint vectors inside the model's range, the pose each puts ``DEFAULT_FRAME`` at, the joints IK finds for
that pose from a seed 0.15 rad away, and two targets outside the arm's reach. Regenerating it is a change
of behaviour, not of test data.
"""

from pathlib import Path

import numpy as np
import pytest

from positronic import geom
from positronic.drivers.roboarm.kinematics import MjcfKinematics
from positronic.drivers.roboarm.models import DEFAULT_FRAME

_YAM_MJCF = 'assets/mujoco/i2rt_yam/yam.xml'
_YAM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
_GOLDENS = Path(__file__).with_name('yam_kinematics_goldens.npz')


def _yam_reach_postures(x: float, y: float) -> list[np.ndarray]:
    """The YAM's own warm starts, copied from its driver so the goldens meet the seeds that made them."""
    az = np.arctan2(y, x)
    return [np.array([az, 1.8, 2.2, 0.0, -0.9, 0.0]), np.array([az, 1.2, 1.2, 0.0, 0.6, 0.0])]


@pytest.fixture(scope='module')
def yam() -> MjcfKinematics:
    return MjcfKinematics(_YAM_MJCF, DEFAULT_FRAME, _YAM_JOINTS, _yam_reach_postures)


@pytest.fixture(scope='module')
def goldens() -> dict[str, np.ndarray]:
    return dict(np.load(_GOLDENS))


def _pose(row: np.ndarray) -> geom.Transform3D:
    return geom.Transform3D(row[:3], geom.Rotation.from_quat(row[3:]))


def test_forward_kinematics_answer_what_the_yam_driver_answered(yam, goldens):
    for q, want in zip(goldens['q'], goldens['fk'], strict=True):
        got = yam.fk(q)
        np.testing.assert_allclose(got.translation, want[:3], atol=1e-9)
        assert geom.quat_closest(got.rotation, geom.Rotation.from_quat(want[3:])) == geom.Rotation.from_quat(want[3:])


def test_inverse_kinematics_answer_what_the_yam_driver_answered(yam, goldens):
    for target, seed, want, solved in zip(
        goldens['ik_target'], goldens['ik_seed'], goldens['ik_q'], goldens['ik_solved'], strict=True
    ):
        got = yam.ik(_pose(target), seed)
        if not solved:
            assert got is None
            continue
        assert got is not None
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_a_capped_search_keeps_the_shape_the_arm_stands_in(yam, goldens):
    """``max_jump`` is what a streamed setpoint is solved under: no seed but the live posture, and no
    solution further from it than the cap."""
    target, seed = _pose(goldens['ik_target'][0]), goldens['ik_seed'][0]

    assert yam.ik(target, seed, max_jump=1e-6) is None

    got = yam.ik(target, seed, max_jump=1.0)
    assert got is not None and np.all(np.abs(got - seed) <= 1.0)


def test_a_target_beyond_the_arm_is_refused_rather_than_approached(yam):
    """The solver stops wherever it got to, so an unverified answer would be a pose the arm cannot hold."""
    assert yam.ik(geom.Transform3D(np.array([2.0, 0.0, 0.3])), np.zeros(6)) is None
