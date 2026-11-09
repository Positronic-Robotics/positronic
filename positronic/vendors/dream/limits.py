"""Motion limits on Dream-1 chunks, after the Default Limits of the Dream SDK station (`robots/control.py`).

DreamAPI enforces no motion limit, and the terms of its API make the client responsible for one (API Terms 4.7(d),
10.5). The station applies its limits at the motors; these apply to each chunk before the client plays it.
"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np

from positronic.drivers.roboarm.models import GRIPPER_SITE, YAM_JOINT_NAMES, YAM_JOINTS, YAM_MJCF_PATH
from positronic.utils import check_finite_nonnegative, package_assets_path


@dataclass(frozen=True)
class ChunkLimits:
    """The limits a chunk keeps, with the Default Limits of the Dream SDK station as defaults.

    The hand moves at most `max_ee_speed_m_s`, and each gripper opening changes by at most `max_grip_step` per row. For
    the first `ramp_s` of a session, and again after the client drops the rest of a chunk, no joint moves faster than
    `ramp_max_rad_s`. Each joint stays `joint_margin_rad` inside its range, as the station's tracker clips its reference
    (`robots/sys0/tracker.py`), so an arm that Dream sends past a stop holds short of it. A joint that starts inside the
    margin, as joints 2 and 3 parked at their lower stop do, moves out of it no faster than the other limits allow, and
    never toward the stop.
    """

    max_ee_speed_m_s: float = 0.6
    max_grip_step: float = 0.05
    ramp_s: float = 2.0
    ramp_max_rad_s: float = 0.5
    joint_margin_rad: float = 0.03

    def __post_init__(self):
        for name, value in vars(self).items():
            check_finite_nonnegative(value, name)


@dataclass(frozen=True)
class LimitReport:
    """How many rows of a chunk each limit changed."""

    joint_range: int
    ee_speed: int
    grip_step: int
    ramp: int


class YamChunkLimiter:
    """Applies `ChunkLimits` to chunks of YAM rows at `fps`. A row holds 6 joints and then the gripper opening.

    Each chunk starts from the measured state of the arm. The hand speed comes from forward kinematics at
    `GRIPPER_SITE` of the vendored YAM model, and the joint range from the same model.
    """

    # TODO(#517): take FK and the joint range from the shared driver kinematics once it exists.
    def __init__(self, limits: ChunkLimits, fps: float):
        self._limits = limits
        self._fps = fps
        self._model = mj.MjModel.from_xml_path(package_assets_path(YAM_MJCF_PATH))
        self._data = mj.MjData(self._model)
        self._site = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, GRIPPER_SITE)
        self._qpos = np.array([self._model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES])
        ranges = np.array([self._model.joint(name).range for name in YAM_JOINT_NAMES])
        margin = limits.joint_margin_rad
        if np.any(ranges[:, 1] - ranges[:, 0] <= 2 * margin):
            raise ValueError(f'joint_margin_rad {margin} leaves no room inside the range of some YAM joint')
        self._lower, self._upper = ranges[:, 0] + margin, ranges[:, 1] - margin

    def _hand(self, joints: np.ndarray) -> np.ndarray:
        self._data.qpos[self._qpos] = joints
        mj.mj_kinematics(self._model, self._data)
        return self._data.site_xpos[self._site].copy()

    def limit(self, rows: np.ndarray, measured: np.ndarray, elapsed_s: float) -> tuple[np.ndarray, LimitReport]:
        """`rows` of one arm, limited, as a `(count, 7)` array, and what each limit changed.

        `measured` is the arm's state as one row. `elapsed_s` is the session time at the first row.
        """
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != YAM_JOINTS + 1 or not np.isfinite(rows).all():
            raise ValueError(f'A chunk holds finite rows of {YAM_JOINTS + 1} values, got shape {rows.shape}')
        step_s = 1.0 / self._fps
        max_hand_step = self._limits.max_ee_speed_m_s * step_s
        out = np.empty_like(rows)
        previous = np.asarray(measured, dtype=np.float64)
        previous_hand = self._hand(previous[:YAM_JOINTS])
        joint_range = ee_speed = grip_steps = ramp = 0
        for index, row in enumerate(rows):
            joints = np.clip(row[:YAM_JOINTS], self._lower, self._upper)
            joint_range += int(not np.array_equal(joints, row[:YAM_JOINTS]))
            delta = joints - previous[:YAM_JOINTS]
            if elapsed_s + index * step_s < self._limits.ramp_s:
                cap = self._limits.ramp_max_rad_s * step_s
                if np.abs(delta).max() > cap:
                    delta *= cap / np.abs(delta).max()
                    ramp += 1
            hand = self._hand(previous[:YAM_JOINTS] + delta)
            # Over one row the hand path is close to straight, so the scaled joint step moves the hand about
            # `max_hand_step`.
            travel = np.linalg.norm(hand - previous_hand)
            if travel > max_hand_step:
                delta *= max_hand_step / travel
                hand = self._hand(previous[:YAM_JOINTS] + delta)
                ee_speed += 1
            grip = float(np.clip(row[YAM_JOINTS], 0.0, 1.0))
            step_cap = self._limits.max_grip_step
            grip_step = float(np.clip(grip - previous[YAM_JOINTS], -step_cap, step_cap))
            grip_steps += int(previous[YAM_JOINTS] + grip_step != grip)
            out[index, :YAM_JOINTS] = previous[:YAM_JOINTS] + delta
            out[index, YAM_JOINTS] = previous[YAM_JOINTS] + grip_step
            previous, previous_hand = out[index], hand
        return out, LimitReport(joint_range=joint_range, ee_speed=ee_speed, grip_step=grip_steps, ramp=ramp)
