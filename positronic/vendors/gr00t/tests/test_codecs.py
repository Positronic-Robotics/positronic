import numpy as np

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.policy import keys as policy_keys
from positronic.policy.codec import ACTION
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


def test_training_actions_align_recorded_samples():
    codec = DroidCodec(image_mappings={})
    episode = EpisodeContainer({
        keys.EE_POSE: DummySignal([100], [[0, 0, 0, 1, 0, 0, 0]]),
        keys.JOINTS: DummySignal([100, 300], [np.zeros(7), np.ones(7)]),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0]),
    })
    encoded = codec.training_encoder(episode)
    action = encoded[ACTION]
    assert list(action.timestamps(RECORDED_TIME)) == [100, 200, 300]
    values = np.asarray(action.values())
    np.testing.assert_array_equal(values[:, :9], np.repeat(encoded[gr00t.EE_POSE].values(), 3, axis=0))
    np.testing.assert_array_equal(values[:, 9], [0, 1, 1])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.zeros(7), np.ones(7)])
