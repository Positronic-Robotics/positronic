import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.ds.internal import REAL_ROBOT_TRANSFORM, SIM_ROBOT_TRANSFORM
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.models import DEFAULT_FRAME, EE_LINK, EEFrame, bundled_franka_model, ee_frames_entry
from positronic.policy.training import ChangeEEFrame

QUAT = geom.Rotation.Representation.QUAT
OFFSET = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, 0.0, 0.3]))
# rules-allow: hardcoded-keys — the keys a recording wrote before ``ee_frames``, which no code writes now
CONTROL_FRAME, EE_FRAME = 'control_frame', 'ee_frame'


def _recording(**statics):
    poses = DummySignal([1000, 2000], np.tile([0.3, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0], (2, 1)))
    return EpisodeContainer(data={keys.EE_POSE: poses, keys.TARGET_EE_POSE: poses, **statics})


def _sites(episode):
    return {name: frame.site for name, frame in ee_frames(episode).items()}


@pytest.mark.parametrize(
    'statics, site',
    [({CONTROL_FRAME: EE_LINK}, EE_LINK), ({CONTROL_FRAME: DEFAULT_FRAME}, DEFAULT_FRAME), ({}, DEFAULT_FRAME)],
    ids=['before_485', 'default', 'no_frame'],
)
def test_a_real_recording_states_the_site_its_control_frame_named(statics, site):
    relabelled = REAL_ROBOT_TRANSFORM(_recording(**statics))
    assert _sites(relabelled) == dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), site)
    assert CONTROL_FRAME not in relabelled


def test_a_real_recording_states_the_offset_its_ee_frame_named():
    relabelled = REAL_ROBOT_TRANSFORM(_recording(**{CONTROL_FRAME: DEFAULT_FRAME, EE_FRAME: OFFSET.as_vector(QUAT)}))
    for frame in ee_frames(relabelled).values():
        assert frame.site == DEFAULT_FRAME
        np.testing.assert_allclose(frame.offset.as_matrix, OFFSET.as_matrix, atol=1e-12)
    assert EE_FRAME not in relabelled


def test_a_recording_that_states_its_frames_keeps_them():
    stated = ee_frames_entry(dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), EEFrame(site=EE_LINK)))
    for transform in (REAL_ROBOT_TRANSFORM, SIM_ROBOT_TRANSFORM):
        assert set(_sites(transform(_recording(**stated))).values()) == {EE_LINK}


def test_a_sim_recording_states_the_site_of_the_bundled_model():
    relabelled = SIM_ROBOT_TRANSFORM(_recording(**{CONTROL_FRAME: EE_LINK}))
    assert _sites(relabelled) == dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), DEFAULT_FRAME)


def test_training_refuses_to_move_a_recording_from_before_485():
    recording = _recording(**{CONTROL_FRAME: EE_LINK, roboarm_keys.URDF: bundled_franka_model()[roboarm_keys.URDF]})
    with pytest.raises(ValueError, match=EE_LINK):
        ChangeEEFrame(DEFAULT_FRAME, OFFSET)(REAL_ROBOT_TRANSFORM(recording))
