"""The robot transforms read every recording at the flange, from the model it was recorded on."""

from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.ds.internal import REAL_ROBOT_TRANSFORM, SIM_ROBOT_TRANSFORM
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import _prepare_spec
from positronic.drivers.roboarm.models import (
    DEFAULT_FRAME,
    EE_LINK,
    FLANGE_LINK,
    bundled_franka_model,
    bundled_panda_model,
)
from positronic.utils import package_assets_path

_QUAT = geom.Rotation.Representation.QUAT
_JOINT_SETS = [
    np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]),
    np.array([0.5, 0.3, -0.4, -1.2, 0.8, 1.0, -0.3]),
    np.array([-0.8, -1.0, 0.6, -2.5, -0.3, 2.5, 0.9]),
]
# rules-allow: hardcoded-keys — the plural name older recordings stored the arm command under is this test's input
_LEGACY_TARGET = 'robot_commands.pose'


def _pose(urdf: str, q: np.ndarray, frame: str) -> np.ndarray:
    model = _prepare_spec(urdf, frame).compile()
    data = mj.MjData(model)
    data.qpos[[model.joint(f'joint{i}').qposadr.item() for i in range(1, 8)]] = q
    mj.mj_forward(model, data)
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site(frame).xmat)
    return np.concatenate([data.site(frame).xpos.copy(), quat])


def _recording(poses_urdf: str, frame: str, target_key: str = keys.TARGET_EE_POSE, **statics) -> EpisodeContainer:
    ts = np.arange(len(_JOINT_SETS), dtype=np.int64) * 100_000_000
    poses = np.array([_pose(poses_urdf, q, frame) for q in _JOINT_SETS])
    return EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, np.array(_JOINT_SETS)),
            keys.EE_POSE: DummySignal(ts, poses),
            target_key: DummySignal(ts, poses),
            **statics,
        }
    )


def _model(model: dict) -> dict:
    return {key: model[key] for key in (roboarm_keys.URDF, roboarm_keys.JOINT_NAMES, roboarm_keys.CONTROL_FRAME)}


def _assert_at_the_flange(episode) -> None:
    flange_urdf = bundled_franka_model()[roboarm_keys.URDF]
    for signal in (keys.EE_POSE, keys.TARGET_EE_POSE):
        for i, q in enumerate(_JOINT_SETS):
            got = geom.Transform3D.from_vector(np.asarray(episode[signal][i][0]), _QUAT)
            want = geom.Transform3D.from_vector(_pose(flange_urdf, q, FLANGE_LINK), _QUAT)
            np.testing.assert_allclose(got.translation, want.translation, atol=1e-9)
            assert (want.rotation.inv * got.rotation).angle < 1e-9
    model_flange = _pose(episode[roboarm_keys.URDF], _JOINT_SETS[1], FLANGE_LINK)
    np.testing.assert_allclose(_pose(episode[roboarm_keys.URDF], _JOINT_SETS[1], DEFAULT_FRAME), model_flange)


def test_a_real_recording_at_the_old_default_reads_at_the_flange():
    legacy = bundled_franka_model(EE_LINK)
    _assert_at_the_flange(REAL_ROBOT_TRANSFORM(_recording(legacy[roboarm_keys.URDF], EE_LINK, **_model(legacy))))


def test_a_real_recording_that_names_its_frame_reads_at_the_flange():
    urdf = Path(package_assets_path('drivers/roboarm/fr3.urdf')).read_text()
    recording = _recording(
        urdf, EE_LINK, _LEGACY_TARGET, **{roboarm_keys.URDF: urdf, roboarm_keys.CONTROL_FRAME: EE_LINK}
    )
    _assert_at_the_flange(REAL_ROBOT_TRANSFORM(recording))


def test_a_real_recording_with_no_model_reads_at_the_flange():
    _assert_at_the_flange(REAL_ROBOT_TRANSFORM(_recording(bundled_franka_model(EE_LINK)[roboarm_keys.URDF], EE_LINK)))


@pytest.mark.parametrize(
    'transform, model',
    [(REAL_ROBOT_TRANSFORM, bundled_franka_model()), (SIM_ROBOT_TRANSFORM, bundled_panda_model())],
    ids=['real', 'sim'],
)
def test_a_recording_already_at_the_flange_keeps_its_poses(transform, model):
    recording = _recording(model[roboarm_keys.URDF], FLANGE_LINK, **_model(model))

    read = transform(recording)

    for signal in (keys.EE_POSE, keys.TARGET_EE_POSE):
        for i in range(len(_JOINT_SETS)):
            np.testing.assert_array_equal(read[signal][i][0], recording[signal][i][0])


def test_a_sim_recording_with_mujoco_xml_for_a_model_reads_at_the_flange():
    """Older sim recordings store the MuJoCo scene under ``urdf``; their poses sit at the old panda ``default``."""
    legacy = bundled_panda_model(EE_LINK)[roboarm_keys.URDF]
    scene = Path(package_assets_path('assets/mujoco/panda.xml')).read_text()
    recording = _recording(legacy, EE_LINK, **{roboarm_keys.URDF: scene, roboarm_keys.CONTROL_FRAME: EE_LINK})

    read = SIM_ROBOT_TRANSFORM(recording)

    panda = bundled_panda_model()[roboarm_keys.URDF]
    for i, q in enumerate(_JOINT_SETS):
        np.testing.assert_allclose(read[keys.EE_POSE][i][0], _pose(panda, q, FLANGE_LINK), atol=1e-9)


def test_a_sim_recording_carrying_the_old_panda_model_reads_at_the_flange():
    legacy = bundled_panda_model(EE_LINK)
    read = SIM_ROBOT_TRANSFORM(_recording(legacy[roboarm_keys.URDF], EE_LINK, **_model(legacy)))

    panda = bundled_panda_model()[roboarm_keys.URDF]
    for i, q in enumerate(_JOINT_SETS):
        np.testing.assert_allclose(read[keys.EE_POSE][i][0], _pose(panda, q, FLANGE_LINK), atol=1e-9)
