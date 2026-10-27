import numpy as np
import pytest

from pimm.time import RECEIVED_WORLD
from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.drivers.roboarm import command, models
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.policy import spec
from positronic.policy.codecs import ACTION, Codec
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings


@pytest.mark.parametrize('config', [model_settings.droid, model_settings.droid_three_cameras])
def test_pose_conversion_preserves_droid_convention_and_tool_frame_metadata(config):
    settings = config()
    settings[model_settings.IMAGE_MAPPINGS] = {}
    codec = spec.from_spec(recipe.inference(settings))
    assert isinstance(codec, Codec)
    rng = np.random.default_rng(3)
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    for _ in range(20):
        pose = geom.Transform3D(rng.normal(size=3), geom.Rotation.from_quat(rng.normal(size=4)))
        inputs = {
            keys.EE_POSE: pose.as_vector(geom.Rotation.Representation.QUAT),
            keys.JOINTS: np.zeros(7),
            keys.GRIP: 0.2,
            keys.TASK: 'pick',
        }
        tool_pose = pose * models.DROID_EE_FRAME
        expected = np.concatenate([
            tool_pose.translation,
            (tool_pose.rotation.as_rotation_matrix @ correction)[:2].reshape(6),
        ]).astype(np.float32)
        encoded = codec.encode(inputs)[gr00t.STATE][gr00t.EE_POSE][0, 0]
        np.testing.assert_allclose(encoded, expected, atol=2e-7)

    for meta in (codec.meta, recipes.droid(settings=settings).meta):
        frame = models.parse_ee_frames(meta[roboarm_keys.EE_FRAMES])[keys.EE_POSE]
        assert frame.site == models.GRIPPER_SITE
        np.testing.assert_allclose(frame.offset.as_matrix, models.DROID_EE_FRAME.as_matrix, atol=1e-9)


@pytest.mark.parametrize('config', [model_settings.droid, model_settings.droid_three_cameras])
@pytest.mark.parametrize('horizon', [0, 1, 40])
def test_droid_decodes_native_chunk_and_binarizes_grip(config, horizon):
    codec = spec.from_spec(recipe.inference(config()))
    assert isinstance(codec, Codec)
    targets = np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7) / 100
    output = (
        {
            gr00t.JOINT_POSITION: targets[np.newaxis],
            gr00t.GRIP: np.array([0.5 if i % 2 else 0.51 for i in range(horizon)]).reshape(1, horizon, 1),
        },
        {},
    )
    decoded = codec.decode(output)
    assert len(decoded) == horizon
    for i, item in enumerate(decoded):
        np.testing.assert_array_equal(item[keys.ROBOT_COMMAND].positions, targets[i])
        assert item[keys.ROBOT_COMMAND].positions.dtype == np.float32
        assert item[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert item[keys.TARGET_GRIP] == (0.0 if i % 2 else 1.0)
        assert 'timestamp' not in item


def test_droid_control_mode_is_the_droid_impedance_on_the_wire():
    """The serving image cannot import positronic, so it spells out the mode the rig runs."""
    assert model_settings.droid()[model_settings.CONTROL_MODE] == command.to_wire(DROID_IMPEDANCE)


@pytest.mark.parametrize('config', [recipes.droid, recipes.droid_three_cameras])
def test_training_cadence_metadata(config):
    training = config(**{f'settings.{model_settings.ACTION_FPS}': 20})
    assert training.meta[ACTION_FPS] == 20
    assert config().meta[ACTION_FPS] == 15


def _arm_training(**overrides):
    return recipes.droid(**{
        f'settings.{model_settings.IMAGE_MAPPINGS}': {},
        f'settings.{model_settings.EE_FRAMES}': {
            keys.EE_POSE: {
                model_settings.SITE: model_settings.GRIPPER_SITE,
                model_settings.OFFSET: [0, 0, 0, 1, 0, 0, 0],
            }
        },
        **overrides,
    })


def _arm_episode(ts, joints, grip, target_grip, poses=([0, 0, 0, 1, 0, 0, 0],)):
    return EpisodeContainer({
        keys.EE_POSE: DummySignal(ts[: len(poses)], poses),
        keys.JOINTS: DummySignal(ts, joints),
        keys.GRIP: DummySignal(ts, grip),
        keys.TARGET_GRIP: DummySignal(ts, target_grip),
        **models.ee_frames_entry({keys.EE_POSE: models.EEFrame(site=models.GRIPPER_SITE)}),
    })


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_training_actions_align_on_the_recording_clock(timeline):
    training = _arm_training(action_lead_sec=0)
    episode = EpisodeContainer({
        keys.EE_POSE: DummySignal([100], [[0, 0, 0, 1, 0, 0, 0]], timelines=(timeline,)),
        keys.JOINTS: DummySignal([100, 300], [np.zeros(7), np.ones(7)], timelines=(timeline,)),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0], timelines=(timeline,)),
        keys.TARGET_GRIP: DummySignal([100, 200], [0.0, 1.0], timelines=(timeline,)),
        **models.ee_frames_entry({keys.EE_POSE: models.EEFrame(site=models.GRIPPER_SITE)}),
    })
    encoded = training(episode)
    action = encoded[ACTION]
    assert list(action.timestamps(timeline)) == [100, 200, 300]
    values = np.asarray(action.values())
    np.testing.assert_array_equal(values[:, :9], np.repeat(encoded[gr00t.EE_POSE].values(), 3, axis=0))
    np.testing.assert_array_equal(values[:, 9], [0, 1, 1])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.zeros(7), np.ones(7)])


@pytest.mark.parametrize('step_ns', [66_000_000, 66_666_667])
def test_training_arm_actions_are_the_state_two_steps_ahead_and_hold_the_last_state(step_ns):
    ts = [step * step_ns for step in range(5)]
    joints = [np.full(7, step, dtype=float) for step in range(5)]
    poses = [[step / 10, 0, 0, 1, 0, 0, 0] for step in range(5)]
    encoded = _arm_training()(_arm_episode(ts, joints, [0.0] * 5, [0.0] * 5, poses))
    values = np.asarray(encoded[ACTION].values())
    ahead = [2, 3, 4, 4, 4]
    np.testing.assert_array_equal(values[:, :9], np.asarray(encoded[gr00t.EE_POSE].values())[ahead])
    np.testing.assert_array_equal(values[:, 10:], np.asarray(encoded[gr00t.JOINT_POSITION].values())[ahead])
    np.testing.assert_array_equal(np.asarray(encoded[gr00t.JOINT_POSITION].values())[:, 0], [0, 1, 2, 3, 4])


@pytest.mark.parametrize(('binarize_grip', 'expected'), [(True, [0, 0, 1, 1]), (False, [0.2, 0.5, 0.7, 1.0])])
def test_training_gripper_action_is_the_commanded_grip(binarize_grip, expected):
    ts = [0, 1, 2, 3]
    training = _arm_training(binarize_grip=binarize_grip)
    encoded = training(_arm_episode(ts, [np.zeros(7)] * 4, [0.0, 0.1, 0.3, 0.3], [0.2, 0.5, 0.7, 1.0]))
    np.testing.assert_allclose(np.asarray(encoded[ACTION].values())[:, 9], expected)
    np.testing.assert_allclose(np.asarray(encoded[gr00t.GRIP].values())[:, 0], [0.0, 0.1, 0.3, 0.3])
