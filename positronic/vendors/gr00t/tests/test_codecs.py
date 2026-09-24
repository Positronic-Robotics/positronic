import numpy as np

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.policy import keys as policy_keys
from positronic.vendors import gr00t
from positronic.vendors.gr00t.codecs import droid


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
