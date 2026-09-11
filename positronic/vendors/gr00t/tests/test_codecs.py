import numpy as np
import pytest

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.vendors import gr00t
from positronic.vendors.gr00t.codecs import droid


def test_droid_executes_fifteen_joint_targets_at_15hz_and_binarizes_grip():
    codec = droid()
    targets = np.arange(40 * 7, dtype=np.float32).reshape(40, 7) / 100
    output = [
        {gr00t.JOINT_POSITION: q, gr00t.GRIP: [0.5 if i % 2 else 0.51], gr00t.EE_POSE: np.zeros(9)}
        for i, q in enumerate(targets)
    ]
    decoded = codec.decode(output)
    assert len(decoded) == 16
    for i, item in enumerate(decoded[:-1]):
        np.testing.assert_array_equal(item[keys.ROBOT_COMMAND].positions, targets[i])
        assert item[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert item[keys.TARGET_GRIP] == (0.0 if i % 2 else 1.0)
        assert item[keys.ACTION_TIMESTAMP] == pytest.approx(i / 15)
    assert decoded[-1] == {keys.ACTION_TIMESTAMP: 1.0}
