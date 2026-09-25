import numpy as np

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.policy import keys as policy_keys
from positronic.vendors import gr00t
from positronic.vendors.gr00t.codecs import DroidCodec, droid


def test_droid_decodes_full_chunk_and_binarizes_grip():
    codec = droid()
    targets = np.arange(40 * 7, dtype=np.float32).reshape(40, 7) / 100
    output = [
        {gr00t.JOINT_POSITION: q, gr00t.GRIP: [0.5 if i % 2 else 0.51], gr00t.EE_POSE: np.zeros(9)}
        for i, q in enumerate(targets)
    ]
    decoded = codec.decode(output)
    assert len(decoded) == 40
    for i, item in enumerate(decoded):
        np.testing.assert_array_equal(item[keys.ROBOT_COMMAND].positions, targets[i])
        assert item[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert item[keys.TARGET_GRIP] == (0.0 if i % 2 else 1.0)
        assert 'timestamp' not in item


def test_training_cadence_is_preserved_without_timestamp_commands():
    codec = droid(training_fps=20)
    assert codec.training_encoder.meta[policy_keys.ACTION_FPS] == 20


def test_observation_warms_through_the_pose_it_recodes():
    """A zero-filled pose would raise here: ``_encode_pose`` recodes the quaternion into rot6d."""
    codec = DroidCodec(image_mappings={gr00t.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE: keys.WRIST_IMAGE})

    obs = codec.warm_observation('stack the cubes')

    assert obs is not None
    assert obs[gr00t.LANGUAGE][gr00t.TASK] == [['stack the cubes']]
    width, height = gr00t.IMAGE_SIZE
    assert obs[gr00t.VIDEO][gr00t.WRIST_IMAGE].shape == (1, 1, height, width, 3)
    assert obs[gr00t.STATE][gr00t.EE_POSE].shape == (1, 1, gr00t.STATE_DIMS[gr00t.EE_POSE])
    assert obs[gr00t.STATE][gr00t.JOINT_POSITION].shape == (1, 1, gr00t.STATE_DIMS[gr00t.JOINT_POSITION])


def test_the_joint_width_comes_from_the_state_dims_both_ways(monkeypatch):
    monkeypatch.setitem(gr00t.STATE_DIMS, gr00t.JOINT_POSITION, 6)
    codec = DroidCodec(image_mappings={gr00t.WRIST_IMAGE: keys.WRIST_IMAGE})

    obs = codec.warm_observation('stack the cubes')
    decoded = codec.decode({gr00t.JOINT_POSITION: np.zeros(6), gr00t.GRIP: [0.0]})

    assert obs is not None
    assert obs[gr00t.STATE][gr00t.JOINT_POSITION].shape == (1, 1, 6)
    assert decoded[keys.ROBOT_COMMAND].positions.shape == (6,)
