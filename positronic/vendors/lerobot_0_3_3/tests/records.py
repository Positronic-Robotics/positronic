"""Records of a one-arm and a two-arm rig, for the tests of the LeRobot ACT codecs."""

from pathlib import Path

import numpy as np

from positronic import keys
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.time import Time
from positronic.drivers.recording import HARNESS_WORLD
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry

TASK = 'put the banana in the bowl'
SAMPLES = 4
IMAGE = np.full((32, 32, 3), 120, np.uint8)
ONE_ARM_CAMERAS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
TWO_ARM_CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE)
# The y of each arm, so that the poses of the two arms differ.
_ARM_Y = {None: 0.0, keys.LEFT_ARM: 0.3, keys.RIGHT_ARM: -0.3}


def arm_pose(arm: str | None, sample: int) -> np.ndarray:
    return np.array([0.3 + 0.01 * sample, _ARM_Y[arm], 0.4, 1.0, 0.0, 0.0, 0.0])


def arm_grip(arm: str | None, sample: int) -> float:
    return 1.0 - sample / 8 if arm == keys.RIGHT_ARM else sample / 8


def write_record(root: Path, arms: tuple[str | None, ...], cameras: tuple[str, ...]) -> LocalDataset:
    """A record of the pose, the grip and the command of each arm of ``arms``, and the images of ``cameras``. The
    command of an arm at a sample is the pose and the grip of the next sample. The poses are at ``GRIPPER_SITE``."""
    poses = [keys.arm_key(key, arm) for arm in arms for key in (keys.EE_POSE, keys.TARGET_EE_POSE)]
    frames = ee_frames_entry(dict.fromkeys(poses, EEFrame(site=GRIPPER_SITE)))
    with LocalDatasetWriter(root) as dataset, dataset.new_episode() as writer:
        writer.set_static(keys.TASK, TASK)
        writer.set_static(roboarm_keys.EE_FRAMES, frames[roboarm_keys.EE_FRAMES])
        for sample in range(SAMPLES):
            ts = Time(**{HARNESS_WORLD: sample * 100_000_000})
            for arm in arms:
                writer.append(keys.arm_key(keys.EE_POSE, arm), arm_pose(arm, sample), ts)
                writer.append(keys.arm_channel(keys.GRIP, arm), np.float32(arm_grip(arm, sample)), ts)
                writer.append(keys.arm_key(keys.TARGET_EE_POSE, arm), arm_pose(arm, sample + 1), ts)
                writer.append(keys.arm_channel(keys.TARGET_GRIP, arm), np.float32(arm_grip(arm, sample + 1)), ts)
            for camera in cameras:
                writer.append(camera, IMAGE, ts)
    return LocalDataset(root)
