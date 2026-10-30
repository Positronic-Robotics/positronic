"""The ZED-X view of ABC's scene: a camera fit to the ZED-X stations of the ABC-130k dataset.

Runs in ABC's own interpreter, beside ``env.py``. The fit moves the ``top`` camera relative to the arm mounts and each
wrist camera relative to ABC's own wrist camera, and gives each camera a vertical field of view.
"""

import dataclasses
from collections.abc import Mapping

import mapping
import mujoco
import numpy as np

TOP_CAMERA = 'top'


@dataclasses.dataclass(frozen=True)
class TopCamera:
    """A camera that looks along the arms' forward axis from ``offset``, then turns by ``yaw``, ``pitch``, ``roll``."""

    offset: tuple[float, float, float]  # m from the midpoint of the arm mounts: forward, left, up, in the mount frame
    yaw_deg: float  # to the left
    pitch_deg: float  # below the horizontal
    roll_deg: float  # about the view axis
    fovy_deg: float


@dataclasses.dataclass(frozen=True)
class WristCamera:
    """ABC's own wrist camera, moved by ``offset`` and turned about its own x, y, z axes, in that order."""

    offset: tuple[float, float, float]  # m along ABC's wrist camera axes: right, up, back
    rotation_deg: tuple[float, float, float]
    fovy_deg: float


@dataclasses.dataclass(frozen=True)
class CameraFit:
    top: TopCamera
    wrists: Mapping[str, WristCamera]  # by arm


# The ZED-X stations of the ABC-130k dataset: vertical fields of view from the median factory calibration of 18
# episodes (top fy about 731 px, wrists fy about 719 px, at 1920x1200). The wrist pixels are not square (fx about
# 671 px), so a 16:10 render at this fovy sees about 106 deg across, against their 110 deg.
_ZED_X_TOP_FOVY_DEG = 78.8
_ZED_X_WRIST_FOVY_DEG = 79.7

# The top pose is the left eye of the stereo top camera: it overlaps the arms at their recorded first joints in nine
# first frames from six stations. The wrist poses overlap the open fingers common to 18 first frames. Render at 16:10
# to match their 1920x1200 frames.
_LEFT, _RIGHT = mapping.ARMS
ZED_X_FIT = CameraFit(
    top=TopCamera(
        offset=(-0.120, 0.053, 0.878), yaw_deg=0.75, pitch_deg=48.5, roll_deg=-2.06, fovy_deg=_ZED_X_TOP_FOVY_DEG
    ),
    wrists={
        _LEFT: WristCamera(
            offset=(0.0, 0.019, -0.061), rotation_deg=(1.375, 1.375, -0.125), fovy_deg=_ZED_X_WRIST_FOVY_DEG
        ),
        _RIGHT: WristCamera(
            offset=(-0.005, 0.019, -0.060), rotation_deg=(0.375, -3.5, 0.0), fovy_deg=_ZED_X_WRIST_FOVY_DEG
        ),
    },
)


def _rotation(axis: int, angle_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    rotation = np.eye(3)
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


def top_camera_axes(camera: TopCamera) -> np.ndarray:
    """The MuJoCo camera axes (columns x right, y up, z back) in the mount frame."""
    pitch = np.radians(camera.pitch_deg)
    back = -np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    axes = np.stack([right, np.cross(back, right), back], axis=1)
    return _rotation(2, camera.yaw_deg) @ axes @ _rotation(2, camera.roll_deg)


class CameraPlacer:
    """Places a fit's cameras in one compiled scene; ABC recompiles the scene at each reset, so place after each."""

    def __init__(self, fit: CameraFit, model):
        self._fit = fit
        self._top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TOP_CAMERA)
        # ABC names each wrist camera after its arm.
        self._wrists = {arm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, arm) for arm in fit.wrists}
        # ABC's own wrist poses, which a wrist fit is relative to; read before any placement moves them.
        self._abc_wrists = {arm: (model.cam_pos[c].copy(), model.cam_quat[c].copy()) for arm, c in self._wrists.items()}

    def place(self, model, data, mount_body_ids: list[int]) -> None:
        mujoco.mj_kinematics(model, data)
        mount_rot = data.xmat[mount_body_ids[0]].reshape(3, 3)
        mounts = np.mean([data.xpos[body] for body in mount_body_ids], axis=0)
        top = self._fit.top
        body = model.cam_bodyid[self._top]
        body_pos, body_rot = data.xpos[body], data.xmat[body].reshape(3, 3)
        model.cam_pos[self._top] = body_rot.T @ (mounts + mount_rot @ np.asarray(top.offset) - body_pos)
        mujoco.mju_mat2Quat(
            model.cam_quat[self._top], np.ascontiguousarray(body_rot.T @ mount_rot @ top_camera_axes(top)).reshape(9)
        )
        model.cam_fovy[self._top] = top.fovy_deg
        for arm, camera in self._wrists.items():
            wrist = self._fit.wrists[arm]
            abc_pos, abc_quat = self._abc_wrists[arm]
            abc_rot = np.empty(9)
            mujoco.mju_quat2Mat(abc_rot, abc_quat)
            abc_rot = abc_rot.reshape(3, 3)
            turn = np.linalg.multi_dot([_rotation(axis, angle) for axis, angle in enumerate(wrist.rotation_deg)])
            model.cam_pos[camera] = abc_pos + abc_rot @ np.asarray(wrist.offset)
            mujoco.mju_mat2Quat(model.cam_quat[camera], np.ascontiguousarray(abc_rot @ turn).reshape(9))
            model.cam_fovy[camera] = wrist.fovy_deg
        # The renderer reads camera poses from ``data``, which only mj_camlight writes.
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
