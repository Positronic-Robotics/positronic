import numpy as np

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import ACTION
from positronic.vendors import dream
from positronic.vendors.dream import codecs

TASK = 'put the plastic bottles in the bin'


def _view(red: int, height: int = 260, width: int = 416) -> np.ndarray:
    """A view of one colour, with green and blue fixed, so a swap of channels or of cameras shows."""
    view = np.empty((height, width, 3), dtype=np.uint8)
    view[...] = (red, 100, 200)
    return view


def _obs(**changes) -> dict:
    obs = {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float64),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float64),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 0.75,
        keys.EXTERIOR_IMAGE: _view(10),
        keys.WRIST_LEFT_IMAGE: _view(20),
        keys.WRIST_RIGHT_IMAGE: _view(30),
        keys.TASK: TASK,
    }
    return {**obs, **changes}


def _encode(obs: dict, **overrides) -> dict:
    return codecs.yam_bimanual.override(**overrides).instantiate().encode(obs)


def test_each_arm_goes_to_its_station_slot_with_the_gripper_opening():
    observation = _encode(_obs())[dream.OBSERVATION]

    assert observation['state.arm_l.joint_pos'] == [0, 1, 2, 3, 4, 5]
    assert observation['state.arm_l.gripper_pos'] == [1.0]
    assert observation['state.arm_r.joint_pos'] == [6, 7, 8, 9, 10, 11]
    assert observation['state.arm_r.gripper_pos'] == [0.25]


def test_the_observation_carries_the_task_and_the_station_robot():
    observation = _encode(_obs())[dream.OBSERVATION]

    assert observation[dream.PROMPT] == TASK
    assert {name: observation[name] for name in dream.STATION_ROBOT} == dream.STATION_ROBOT


def test_a_replan_of_the_client_passes_through():
    assert _encode(_obs(**{policy_keys.REPLAN_NOW: True}))[policy_keys.REPLAN_NOW] is True
    assert _encode(_obs())[policy_keys.REPLAN_NOW] is False


def _prefix(rows: int, offset_s: float) -> dict:
    actions = [{'row': index} for index in range(rows)]
    prefix = {policy_keys.RTC.PREFIX_ACTIONS: actions, policy_keys.RTC.OBSERVATION_OFFSET_SEC: offset_s}
    return {policy_keys.RTC.ACTION_PREFIX: prefix}


def test_the_rows_ahead_are_the_rows_of_the_plan_that_play_after_the_observation():
    # The row that plays now became due 0.02 s before the observation: 12 rows of 0.05 s end 0.58 s after it.
    assert _encode(_obs(**_prefix(12, -0.02)))[dream.ROWS_AHEAD] == 12
    assert _encode(_obs(**_prefix(12, -0.04)))[dream.ROWS_AHEAD] == 11
    # A plan that starts 0.1 s after the observation plays 2 rows of hold first.
    assert _encode(_obs(**_prefix(12, 0.1)))[dream.ROWS_AHEAD] == 14


def test_no_rows_are_ahead_when_the_arm_holds_its_last_row_or_no_plan_comes():
    assert _encode(_obs(**_prefix(0, -0.3)))[dream.ROWS_AHEAD] == 0
    assert _encode(_obs())[dream.ROWS_AHEAD] == 0


def test_the_queued_rows_leave_out_the_row_that_plays_at_the_observation_however_far_into_it():
    # 0.4 and 0.8 of a period into the row that plays now: the rows ahead round two ways, the queued rows do not.
    assert _encode(_obs(**_prefix(12, -0.02)))[dream.ROWS_QUEUED] == 11
    assert _encode(_obs(**_prefix(12, -0.04)))[dream.ROWS_QUEUED] == 11
    assert _encode(_obs(**_prefix(12, 0.0)))[dream.ROWS_QUEUED] == 11


def test_every_row_is_queued_when_the_plan_starts_after_the_observation():
    assert _encode(_obs(**_prefix(12, 0.1)))[dream.ROWS_QUEUED] == 12


def test_no_rows_are_queued_when_the_arm_holds_its_last_row_or_no_plan_comes():
    assert _encode(_obs(**_prefix(0, -0.3)))[dream.ROWS_QUEUED] == 0
    assert _encode(_obs())[dream.ROWS_QUEUED] == 0


def test_each_camera_goes_at_dreams_frame_size_under_its_station_window():
    frames = _encode(_obs())[dream.FRAMES]

    assert set(frames) == {'image_window.top', 'image_window.left', 'image_window.right'}
    for window in frames.values():
        assert window.dtype == np.uint8 and window.shape == (1, 176, 320, 3)
    assert {key: window[0, 88, 160].tolist() for key, window in frames.items()} == {
        'image_window.top': [10, 100, 200],
        'image_window.left': [20, 100, 200],
        'image_window.right': [30, 100, 200],
    }


def test_a_frame_stack_keeps_its_last_nine_frames_oldest_first():
    stack = np.stack([_view(red) for red in range(12)])

    window = _encode(_obs(**{keys.EXTERIOR_IMAGE: stack}))[dream.FRAMES]['image_window.top']

    assert window.shape == (9, 176, 320, 3)
    assert window[:, 88, 160, 0].tolist() == list(range(3, 12))


def test_a_frame_is_cut_to_dreams_aspect_ratio_before_it_is_scaled():
    view = _view(10)
    view[:15] = 0  # 416x260 cuts to 416x229: 15 rows go above and 16 below
    view[-16:] = 0
    obs = _obs(**{keys.EXTERIOR_IMAGE: view})

    cut = _encode(obs)[dream.FRAMES]['image_window.top'][0]
    stretched = _encode(obs, crop_to_aspect=False)[dream.FRAMES]['image_window.top'][0]

    assert cut.all()
    assert not stretched[:8].any()


def test_a_row_decodes_to_a_joint_command_and_a_closed_fraction_per_arm():
    row = np.array([0, 1, 2, 3, 4, 5, 0.25, 6, 7, 8, 9, 10, 11, 1.0])

    decoded = codecs.yam_bimanual.instantiate().decode([{ACTION: row}])[0]

    left = decoded[keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)]
    right = decoded[keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)]
    assert isinstance(left, command.JointPosition) and isinstance(right, command.JointPosition)
    np.testing.assert_array_equal(left.positions, np.arange(6))
    np.testing.assert_array_equal(right.positions, np.arange(6, 12))
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == 0.75
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == 0.0
