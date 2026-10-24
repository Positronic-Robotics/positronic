import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.policy.codecs import ACTION
from positronic.vendors import abc_dit
from positronic.vendors.abc_dit import codecs

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


def _encode(obs: dict) -> dict:
    return codecs.multitask_200k.instantiate().encode(obs)


def test_the_state_is_abcs_14_d_layout_left_arm_first_with_gripper_widths():
    state = _encode(_obs())[abc_dit.STATE]

    assert state.dtype == np.float32 and state.shape == (14,)
    np.testing.assert_array_equal(state[:7], [0, 1, 2, 3, 4, 5, 1.0])
    np.testing.assert_array_equal(state[7:], [6, 7, 8, 9, 10, 11, 0.25])


def test_a_swap_of_the_arms_reaches_the_state():
    swapped = _obs(**{
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12.0),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6.0),
    })

    np.testing.assert_array_equal(_encode(swapped)[abc_dit.STATE][:6], np.arange(6, 12))


def test_each_view_goes_channels_first_at_abcs_render_size_under_its_abc_camera_name():
    images = _encode(_obs())[abc_dit.IMAGES]

    assert set(images) == {abc_dit.TOP_CAMERA, abc_dit.LEFT_CAMERA, abc_dit.RIGHT_CAMERA}
    for view in images.values():
        assert view.dtype == np.uint8 and view.shape == (3, 168, 224)
    centre = {name: view[:, 84, 112].tolist() for name, view in images.items()}
    assert centre == {
        abc_dit.TOP_CAMERA: [10, 100, 200],
        abc_dit.LEFT_CAMERA: [20, 100, 200],
        abc_dit.RIGHT_CAMERA: [30, 100, 200],
    }


def test_a_view_of_another_aspect_ratio_is_padded_and_not_stretched():
    # A 416x260 view scales to 224x140, so 14 black rows pad it above and below.
    top = _encode(_obs())[abc_dit.IMAGES][abc_dit.TOP_CAMERA]

    assert not top[:, :13].any() and not top[:, -13:].any()
    assert top[:, 15:-15].all()


def test_a_task_gets_the_prompt_the_checkpoint_trained_under():
    assert _encode(_obs())[abc_dit.PROMPT] == 'sim throw plastic bottles in bin'
    assert _encode(_obs(**{keys.TASK: 'hang mug on mug rack'}))[abc_dit.PROMPT] == 'sim hang the mug on the mug rack'


def test_a_task_the_checkpoint_did_not_train_on_is_refused():
    with pytest.raises(ValueError, match='no prompt for the task'):
        _encode(_obs(**{keys.TASK: 'fold the towel'}))


def test_an_action_decodes_left_arm_first_into_joint_commands_and_positronic_grips():
    vector = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.25]]).astype(np.float32)

    decoded = codecs.multitask_200k.instantiate().decode({ACTION: vector})

    left, right = (
        decoded[keys.arm_channel(keys.ROBOT_COMMAND, keys.LEFT_ARM)],
        decoded[keys.arm_channel(keys.ROBOT_COMMAND, keys.RIGHT_ARM)],
    )
    assert isinstance(left, command.JointPosition) and isinstance(right, command.JointPosition)
    np.testing.assert_array_equal(left.positions, np.arange(6))
    np.testing.assert_array_equal(right.positions, np.arange(6, 12))
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.LEFT_ARM)] == pytest.approx(0.0)
    assert decoded[keys.arm_channel(keys.TARGET_GRIP, keys.RIGHT_ARM)] == pytest.approx(0.75)
