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


def test_an_empty_prompt_prefix_sends_the_trained_prompt_without_the_sim_prefix():
    codec = codecs.multitask_200k.override(prompt_prefix='').instantiate()

    assert codec.encode(_obs(**{keys.TASK: 'hang mug on mug rack'}))[abc_dit.PROMPT] == 'hang the mug on the mug rack'


def _framed_view(box: codecs.CropBox, height: int = 600, width: int = 960) -> np.ndarray:
    """A black frame with a white rectangle that fills `box`."""
    view = np.zeros((height, width, 3), dtype=np.uint8)
    view[round(box.top * height) : round(box.bottom * height), round(box.left * width) : round(box.right * width)] = 255
    return view


@pytest.mark.parametrize('size', [(600, 960), (280, 448)])
def test_each_yambox_view_is_cropped_to_its_box_and_fills_abcs_view(size):
    obs = _obs(**{key: _framed_view(box, *size) for key, box in codecs.YAMBOX_CROPS.items()})

    images = codecs.multitask_200k.override(crops=codecs.YAMBOX_CROPS).instantiate().encode(obs)[abc_dit.IMAGES]

    for view in images.values():
        assert view.shape == (3, 168, 224)
        # The 4:3 box fills the 4:3 view, so no black frame or pad survives, up to one row of rounding.
        assert (view[:, 1:-1, :] == 255).all()


def test_the_wide_top_crop_sends_the_whole_exterior_frame_and_crops_the_wrist_views():
    frames = {key: _framed_view(box) for key, box in codecs.YAMBOX_CROPS.items()}
    codec = codecs.AbcObservationCodec({TASK: 'p'}, crops=codecs.CROPS[codecs.Crop.YAMBOX_WIDE_TOP])

    images = codec.encode(_obs(**frames))[abc_dit.IMAGES]

    # The 16:10 frame pads to 224x140 inside 224x168, so 14 black rows lie above and below it.
    top = images[abc_dit.TOP_CAMERA]
    assert top[:, :13, :].max() == 0 and top[:, -13:, :].max() == 0
    # The whole frame reaches the view: its black left part and its white yambox box both show.
    assert top[:, 84, :50].max() == 0 and top[:, 140, 120].min() == 255
    for camera in (abc_dit.LEFT_CAMERA, abc_dit.RIGHT_CAMERA):
        assert (images[camera][:, 1:-1, :] == 255).all()


def test_a_codec_with_an_action_prefix_tells_the_server_the_prefix_and_the_played_actions():
    prefix = codecs.ActionPrefix(length=4, executed=16)
    codec = codecs.AbcObservationCodec({TASK: 'p'}, action_prefix=prefix)

    request = codec.encode(_obs())

    assert request[abc_dit.PREFIX_LENGTH] == 4
    assert request[abc_dit.EXECUTED_ACTIONS] == 16
    assert codec.meta[codecs.PREFIX_LENGTH_META] == 4
    without = codecs.AbcObservationCodec({TASK: 'p'})
    assert abc_dit.PREFIX_LENGTH not in without.encode(_obs()) and without.meta[codecs.PREFIX_LENGTH_META] == 0


@pytest.mark.parametrize(('length', 'executed'), [(0, 16), (8, 16), (4, 3), (7, 24)])
def test_an_action_prefix_the_checkpoint_did_not_train_on_or_that_overflows_a_chunk_is_refused(length, executed):
    with pytest.raises(ValueError):
        codecs.ActionPrefix(length=length, executed=executed)


def test_the_codec_metadata_names_its_crop_boxes_and_prompt_prefix():
    meta = codecs.multitask_200k.override(crops=codecs.YAMBOX_CROPS, prompt_prefix='').instantiate().meta

    assert meta[codecs.CROPS_META][keys.EXTERIOR_IMAGE] == pytest.approx([4 / 15, 0.32, 5 / 6, 1.0])
    assert meta[codecs.PROMPT_PREFIX_META] == ''


def test_a_crop_box_keeps_only_the_part_of_the_view_inside_it():
    box = codecs.CropBox(0.5, 0.0, 1.0, 1.0)
    view = np.zeros((300, 400, 3), dtype=np.uint8)
    view[:, 200:] = 200
    codec = codecs.AbcObservationCodec({TASK: 'p'}, crops={keys.EXTERIOR_IMAGE: box})

    images = codec.encode(_obs(**{keys.EXTERIOR_IMAGE: view}))[abc_dit.IMAGES]

    # The 200x300 crop pads to 224x168 with black columns; its content is the right half only.
    assert images[abc_dit.TOP_CAMERA][:, :, 60:-60].min() == 200
    # A view with no box goes whole: the uniform wrist view keeps its colour after the pad.
    assert images[abc_dit.LEFT_CAMERA][:, 84, 112].tolist() == [20, 100, 200]


@pytest.mark.parametrize('box', [(0.5, 0.0, 0.5, 1.0), (-0.1, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.2), (0.6, 0, 0.4, 1)])
def test_a_crop_box_outside_the_frame_or_without_area_is_refused(box):
    with pytest.raises(ValueError, match='inside the frame'):
        codecs.CropBox(*box)


def test_a_crop_box_for_a_camera_the_codec_does_not_read_is_refused():
    with pytest.raises(ValueError, match='name no camera'):
        codecs.AbcObservationCodec({TASK: 'p'}, crops={'image.side': codecs.FULL_FRAME})


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
