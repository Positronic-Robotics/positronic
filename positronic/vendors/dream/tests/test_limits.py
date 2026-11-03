import numpy as np
import pytest

from positronic.drivers.roboarm.models import YAM_JOINT_NAMES
from positronic.vendors.dream.limits import ChunkLimits, YamChunkLimiter

FPS = 30.0
# A raised arm, well inside the joint range of the YAM.
POSE = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
# A session time past the ramp.
AFTER_RAMP_S = 10.0


def _limiter(**limits) -> YamChunkLimiter:
    return YamChunkLimiter(ChunkLimits(**limits), FPS)


def _hand_steps(limiter: YamChunkLimiter, start: np.ndarray, rows: np.ndarray) -> np.ndarray:
    hands = [limiter._hand(row[:6]) for row in (start, *rows)]
    return np.linalg.norm(np.diff(hands, axis=0), axis=1)


def test_a_slow_chunk_passes_unchanged():
    rows = POSE + np.linspace(0.0, 0.01, 10)[:, None] * [1, 1, 1, 1, 1, 1, 0]

    out, report = _limiter().limit(rows, POSE, AFTER_RAMP_S)

    np.testing.assert_allclose(out, rows)
    assert (report.joint_range, report.ee_speed, report.grip_step, report.ramp) == (0, 0, 0, 0)


def test_a_fast_chunk_moves_the_hand_at_the_speed_limit():
    limiter = _limiter()
    rows = np.tile(POSE + [0.8, 0, 0, 0, 0, 0, 0], (5, 1))

    out, report = limiter.limit(rows, POSE, AFTER_RAMP_S)

    steps = _hand_steps(limiter, POSE, out)
    assert steps.max() == pytest.approx(0.6 / FPS, rel=0.05)
    assert report.ee_speed == 5
    # Each row moves the arm towards the target, and none goes past it.
    assert np.all(np.diff(out[:, 0]) > 0) and out[-1, 0] < 0.8


def test_the_ramp_holds_each_joint_to_its_speed_cap_at_the_start_of_a_session():
    rows = np.tile(POSE + [0.2, 0, 0, 0, 0, 0, 0], (4, 1))

    out, report = _limiter().limit(rows, POSE, elapsed_s=0.0)

    np.testing.assert_allclose(np.diff(np.concatenate([POSE[None], out])[:, 0]), 0.5 / FPS)
    assert report.ramp == 4


def test_the_ramp_ends_after_ramp_s():
    rows = np.tile(POSE + [0.02, 0, 0, 0, 0, 0, 0], (2, 1))

    _, report = _limiter().limit(rows, POSE, elapsed_s=2.0)

    assert report.ramp == 0


def test_the_gripper_closes_by_at_most_max_grip_step_per_row():
    rows = np.tile(POSE * [1, 1, 1, 1, 1, 1, 0], (3, 1))

    out, report = _limiter().limit(rows, POSE, AFTER_RAMP_S)

    np.testing.assert_allclose(out[:, 6], [0.95, 0.9, 0.85])
    assert report.grip_step == 3


def _model_range(limiter: YamChunkLimiter, joint: int) -> np.ndarray:
    return limiter._model.joint(YAM_JOINT_NAMES[joint]).range


@pytest.mark.parametrize('commanded', [100.0, -100.0])
def test_a_joint_outside_its_range_holds_the_margin_inside_its_stop(commanded):
    limiter = _limiter(max_ee_speed_m_s=100.0)
    rows = POSE[None].copy()
    rows[0, 3] = commanded

    out, report = limiter.limit(rows, POSE, AFTER_RAMP_S)

    lower, upper = _model_range(limiter, 3)
    assert out[0, 3] == pytest.approx(upper - 0.03 if commanded > 0 else lower + 0.03)
    assert report.joint_range == 1


def test_a_joint_inside_the_margin_of_its_stop_is_moved_out_of_it():
    limiter = _limiter(max_ee_speed_m_s=100.0)
    lower, _ = _model_range(limiter, 2)
    rows = POSE[None].copy()
    rows[0, 2] = lower + 0.01

    out, report = limiter.limit(rows, POSE, AFTER_RAMP_S)

    assert out[0, 2] == pytest.approx(lower + 0.03)
    assert report.joint_range == 1


def test_no_margin_clips_to_the_range_itself():
    limiter = _limiter(max_ee_speed_m_s=100.0, joint_margin_rad=0.0)
    rows = POSE[None].copy()
    rows[0, 3] = 100.0

    out, _ = limiter.limit(rows, POSE, AFTER_RAMP_S)

    assert out[0, 3] == _model_range(limiter, 3)[1]


def test_a_margin_that_leaves_no_room_in_a_joint_range_is_refused():
    with pytest.raises(ValueError, match='leaves no room'):
        _limiter(joint_margin_rad=10.0)


@pytest.mark.parametrize('rows', [np.full((2, 7), np.nan), np.zeros((2, 6))])
def test_a_chunk_of_another_shape_or_with_a_non_finite_value_is_refused(rows):
    with pytest.raises(ValueError, match='finite rows of 7'):
        _limiter().limit(rows, POSE, AFTER_RAMP_S)


def test_a_negative_limit_is_refused():
    with pytest.raises(ValueError, match='max_grip_step'):
        ChunkLimits(max_grip_step=-0.1)
