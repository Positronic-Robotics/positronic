import numpy as np

from positronic.vendors.lerobot import cosmos3_droid as droid


def _rows(n: int = 5, start: int = 0):
    joints = np.arange(n * 7, dtype=np.float64).reshape(n, 7)
    grip = np.linspace(0.0, 0.9, n)
    target_grip = np.linspace(1.0, 0.1, n)
    ee_pose = np.tile([0.5, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0], (n, 1))
    return joints, grip, target_grip, droid.droid_rows(joints, grip, target_grip, ee_pose, start)


def test_the_joint_action_of_a_frame_is_the_joint_state_of_the_next_frame():
    joints, _, _, rows = _rows()
    np.testing.assert_array_equal(rows[droid.ACTION_JOINT_POSITION], joints[1:].astype(np.float32))
    np.testing.assert_array_equal(rows[droid.JOINT_POSITIONS], joints[:-1].astype(np.float32))


def test_the_gripper_state_is_the_raw_closed_fraction_and_keeps_values_between_0_and_1():
    _, grip, target_grip, rows = _rows()
    np.testing.assert_array_equal(rows[droid.GRIPPER_POSITION][:, 0], grip[:-1].astype(np.float32))
    np.testing.assert_array_equal(rows[droid.ACTION_GRIPPER_POSITION][:, 0], target_grip[:-1].astype(np.float32))
    assert np.any((rows[droid.GRIPPER_POSITION] > 0) & (rows[droid.GRIPPER_POSITION] < 1))


def test_rows_start_at_the_cut_frame():
    joints, _, _, rows = _rows(n=6, start=2)
    np.testing.assert_array_equal(rows[droid.JOINT_POSITIONS][0], joints[2].astype(np.float32))
    assert len(rows[droid.ACTION_JOINT_POSITION]) == 3


def test_an_episode_whose_arm_drops_14_9_cm_is_left_out():
    heights = 0.6 - np.array([0.0, 0.05, 0.149, 0.12])
    assert droid.start_cut_frame(heights) is None


# The arm waits at the start height (frames 0-3, within 1 cm), descends from frame 4 and is 15 cm down at frame 6.
_WAIT_THEN_DESCENT = 0.6 - np.array([0.0, 0.004, -0.003, 0.009, 0.03, 0.09, 0.15, 0.25])


def test_the_cut_keeps_the_descent_that_reaches_15_cm():
    assert droid.start_cut_frame(_WAIT_THEN_DESCENT) <= 3


def test_the_cut_drops_the_wait_before_the_descent():
    assert droid.start_cut_frame(_WAIT_THEN_DESCENT) >= 3


def test_the_gripper_pointing_down_has_droid_euler_angles():
    # A rotation of pi about x: the gripper points down, as at the DROID start pose.
    cartesian = droid.cartesian_position(np.array([[0.4, 0.1, 0.5, 0.0, 1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(cartesian[0, :3], [0.4, 0.1, 0.5])
    np.testing.assert_allclose(np.abs(cartesian[0, 3]), np.pi, atol=1e-6)
    np.testing.assert_allclose(cartesian[0, 4:], [0.0, 0.0], atol=1e-6)


def test_timestamps_are_the_float32_15_hz_grid_bit_for_bit():
    expected = np.arange(1000, dtype=np.float32) / np.float32(15)
    assert np.array_equal(droid.frame_timestamps(1000).view(np.uint32), expected.view(np.uint32))


def test_the_keep_ranges_key_is_the_one_that_flux_index_droid_looks_up():
    # `flux_action.data.droid.cosmos.filter_key('ts-1')` at the serving pin of flux-action.
    expected = (
        'gs://xembodiment_data/r2d2/r2d2-data-full/ts-1/recordings/'
        'MP4--gs://xembodiment_data/r2d2/r2d2-data-full/ts-1/trajectory.h5'
    )
    assert droid.keep_ranges_key('ts-1') == expected


def test_a_prompt_joins_its_distinct_instructions_with_the_droid_separator():
    assert droid.prompt(['Put the cup on the plate', 'Put the cup on the plate', 'Place the cup']) == (
        'Put the cup on the plate | Place the cup'
    )


def _ids(prefix: str, count: int) -> list[str]:
    return [f'{prefix}{i}' for i in range(count)]


def test_the_held_out_sessions_are_the_ones_closest_to_the_share():
    # A draw in session order takes the 90-episode day; the closest set to 10 of 100 is the 10-episode day.
    sessions = {'towels': {'2025-10-14': _ids('a', 90), '2025-10-15': _ids('b', 10)}}
    assert droid.heldout_sessions(sessions, share=0.1, seed=0) == sorted(_ids('b', 10))


def test_the_held_out_sessions_can_join_small_days_in_each_group():
    sessions = {
        'towels': {'d1': _ids('a', 6), 'd2': _ids('b', 6), 'd3': _ids('c', 100)},
        'spoons': {'d1': _ids('s', 4), 'd2': _ids('t', 36)},
    }
    assert droid.heldout_sessions(sessions, share=0.1, seed=0) == sorted(_ids('a', 6) + _ids('b', 6) + _ids('s', 4))
