import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import ACTION
from positronic.vendors import pi05_abc
from positronic.vendors.pi05_abc import codecs


def _obs(views: dict[str, np.ndarray]) -> dict:
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.zeros(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.zeros(6, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 0.0,
        **views,
        keys.TASK: codecs.TASK,
    }


def test_the_yambox_crop_sends_each_cropped_view_without_pad():
    # A 960x600 frame whose crop box is white and whose rest is black: a view that misses the box shows black.
    views = {}
    for key, box in codecs.YAMBOX_CROPS.items():
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        frame[round(box.top * 600) : round(box.bottom * 600), round(box.left * 960) : round(box.right * 960)] = 255
        views[key] = frame

    request = codecs.AbcObservationCodec(codecs.BOTTLES_PROMPTS, crops=codecs.YAMBOX_CROPS).encode(_obs(views))

    images = request[pi05_abc.IMAGES]
    assert {role: view.shape for role, view in images.items()} == dict.fromkeys(
        (pi05_abc.TOP_CAMERA, pi05_abc.LEFT_CAMERA, pi05_abc.RIGHT_CAMERA), (3, 168, 224)
    )
    assert all(view.min() == 255 for view in images.values())


def test_a_crop_box_for_an_unknown_camera_is_refused():
    with pytest.raises(ValueError, match='name no camera'):
        codecs.AbcObservationCodec(codecs.BOTTLES_PROMPTS, crops={'image.side': codecs.FULL_FRAME})


def _views() -> dict[str, np.ndarray]:
    return dict.fromkeys(codecs.AbcObservationCodec.CAMERAS.values(), np.zeros((300, 400, 3), dtype=np.uint8))


def _planned(left: float, right: float) -> dict:
    """A planned command inside the YAM joint limits: each arm's joints at `left` or `right` plus 0.0..0.5, the left
    grip at 0.25, the right open."""
    return _planned_joints(left + np.arange(6) / 10, right + np.arange(6) / 10)


def _planned_joints(left: np.ndarray, right: np.ndarray) -> dict:
    planned = {}
    for arm, joints, grip in ((keys.LEFT_ARM, left, 0.25), (keys.RIGHT_ARM, right, 0.0)):
        planned[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = command.JointPosition(positions=np.asarray(joints))
        planned[keys.arm_channel(keys.TARGET_GRIP, arm)] = grip
    return planned


def _with_plan(obs: dict, plan: list[dict]) -> dict:
    prefix = {
        policy_keys.RTC.PREFIX_ACTIONS: plan,
        policy_keys.RTC.CHUNK_OFFSET_SEC: 0.5,
        policy_keys.RTC.OBSERVATION_OFFSET_SEC: 0.0,
    }
    return {**obs, policy_keys.RTC.ACTION_PREFIX: prefix}


def test_the_last_command_is_the_command_that_the_arm_holds_in_the_layout_of_the_state():
    request = codecs.abc(last_command=True).encode(_with_plan(_obs(_views()), [_planned(0.1, 0.2), _planned(0.3, 0.4)]))

    # Per arm the joints, then the gripper width, as in the state: 1 - 0.25 for the left grip, 1 - 0 for the right.
    expected = [*(0.1 + np.arange(6) / 10), 0.75, *(0.2 + np.arange(6) / 10), 1.0]
    np.testing.assert_allclose(request[pi05_abc.LAST_COMMAND], expected)
    assert set(request) == {pi05_abc.STATE, pi05_abc.IMAGES, pi05_abc.PROMPT, pi05_abc.LAST_COMMAND}


def test_a_held_command_past_the_yam_joint_limits_reaches_the_model_at_the_limits():
    # The right wrist roll at 5.33 rad (its limit is 2.0944 rad) and the left base yaw at -4 (its limit is -2.61799).
    left, right = np.full(6, 0.5), np.full(6, 0.5)
    left[0], right[5] = -4.0, 5.33
    request = codecs.abc(last_command=True).encode(_with_plan(_obs(_views()), [_planned_joints(left, right)]))

    expected = [-2.61799, *[0.5] * 5, 0.75, *[0.5] * 5, 2.0944, 1.0]
    np.testing.assert_allclose(request[pi05_abc.LAST_COMMAND], expected, atol=1e-6)


@pytest.mark.parametrize('plan', [None, []], ids=['no plan', 'an ended plan'])
def test_with_no_command_in_the_plan_the_last_command_is_the_state(plan):
    obs = _obs(_views()) if plan is None else _with_plan(_obs(_views()), plan)
    request = codecs.abc(last_command=True).encode(obs)

    np.testing.assert_array_equal(request[pi05_abc.LAST_COMMAND], request[pi05_abc.STATE])


def test_a_codec_without_the_last_command_sends_neither_it_nor_the_plan():
    request = codecs.abc().encode(_with_plan(_obs(_views()), [_planned(10, 20)]))

    assert set(request) == {pi05_abc.STATE, pi05_abc.IMAGES, pi05_abc.PROMPT}


def test_the_last_command_does_not_change_how_a_chunk_decodes():
    chunk = [{ACTION: np.arange(14, dtype=np.float32) / 100 + i} for i in range(3)]

    plain, with_last = codecs.abc().decode(chunk), codecs.abc(last_command=True).decode(chunk)

    for a, b in zip(plain, with_last, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(getattr(a[key], 'positions', a[key]), getattr(b[key], 'positions', b[key]))
