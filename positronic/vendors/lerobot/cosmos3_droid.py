"""The Cosmos3-DROID training layout, which the Cosmos3 and FLUX 3 Action DROID recipes read.

`droid_rows` turns the 15 Hz samples of one recording into the rows of one training episode. The writer is
`positronic.vendors.lerobot.to_cosmos3_droid`.
"""

import itertools
import random
from collections.abc import Sequence

import numpy as np

from positronic.geom import Rotation

# Both recipes read the schema of a dataset from the name of its directory, and its episodes from `success/`.
DATASET_DIR = 'droid_plus_lerobot_640x360_20260412'
SUCCESS_DIR = 'success'
FPS = 15
VIEW_SIZE = (640, 360)

EXTERIOR_IMAGE_1 = 'observation.image.exterior_image_1_left'
EXTERIOR_IMAGE_2 = 'observation.image.exterior_image_2_left'
WRIST_IMAGE = 'observation.image.wrist_image_left'
JOINT_POSITIONS = 'observation.state.joint_positions'
GRIPPER_POSITION = 'observation.state.gripper_position'
CARTESIAN_POSITION = 'observation.state.cartesian_position'
ACTION_JOINT_POSITION = 'action.joint_position'
ACTION_GRIPPER_POSITION = 'action.gripper_position'
EPISODE_ID = 'episode_id'

# The keep ranges of each episode, in the JSON that `flux-action index-droid --filter` reads.
KEEP_RANGES_FILE = 'keep_ranges.json'
# The DROID loaders split a stored prompt into its alternatives at this separator.
PROMPT_SEPARATOR = ' | '

# A teleoperator often waits or adjusts the arm before the first reach. A kept episode starts where the arm leaves
# the start height on the first descent that gets this far below it.
MIN_START_DROP_M = 0.15
# The arm is at the start height while it stays within this distance of its height in the first frame.
START_HEIGHT_BAND_M = 0.01


def start_cut_frame(heights: np.ndarray) -> int | None:
    """The start of the first descent that reaches `MIN_START_DROP_M` below the height of frame 0: the last frame
    before it within `START_HEIGHT_BAND_M` of that height. None if the arm never gets that low."""
    below = np.flatnonzero(heights <= heights[0] - MIN_START_DROP_M)
    if not len(below):
        return None
    at_start = np.flatnonzero(np.abs(heights[: below[0]] - heights[0]) <= START_HEIGHT_BAND_M)
    return int(at_start[-1])


# `flux_action.data.droid.cosmos.filter_key` and the Cosmos3 DROID loader both build this key from an episode id.
_KEEP_RANGES_ROOT = 'gs://xembodiment_data/r2d2/r2d2-data-full/'


def keep_ranges_key(episode_id: str) -> str:
    base = _KEEP_RANGES_ROOT + episode_id.strip('/')
    return f'{base}/recordings/MP4--{base}/trajectory.h5'


def frame_timestamps(num_frames: int) -> np.ndarray:
    """The float32 15 Hz grid. Both recipes refuse an episode whose timestamps differ from it in any bit."""
    return np.arange(num_frames, dtype=np.float32) / np.float32(FPS)


def prompt(instructions: Sequence[str]) -> str:
    return PROMPT_SEPARATOR.join(dict.fromkeys(instructions))


def cartesian_position(ee_pose: np.ndarray) -> np.ndarray:
    """DROID's [x, y, z, roll, pitch, yaw] of each [x, y, z, qw, qx, qy, qz] pose. DROID reads the angles as
    extrinsic xyz Euler angles."""
    euler = [Rotation.from_quat(pose[3:]).as_euler for pose in ee_pose]
    return np.concatenate([ee_pose[:, :3], np.asarray(euler).reshape(-1, 3)], axis=1).astype(np.float32)


def droid_rows(joints: np.ndarray, grip: np.ndarray, target_grip: np.ndarray, ee_pose: np.ndarray, start: int):
    """The state and action rows of frames `start` to the second-to-last frame.

    The joint action of frame t is the joint state of frame t + 1, so the last frame has no row. The gripper action
    is the commanded closed fraction at t. Both grippers stay the raw closed fraction: each recipe flips it itself.
    """
    end = len(joints) - 1
    return {
        JOINT_POSITIONS: joints[start:end].astype(np.float32),
        GRIPPER_POSITION: grip[start:end].reshape(-1, 1).astype(np.float32),
        CARTESIAN_POSITION: cartesian_position(ee_pose[start:end]),
        ACTION_JOINT_POSITION: joints[start + 1 : end + 1].astype(np.float32),
        ACTION_GRIPPER_POSITION: target_grip[start:end].reshape(-1, 1).astype(np.float32),
    }


def heldout_sessions(sessions: dict[str, dict[str, list[str]]], share: float, seed: int) -> list[str]:
    """The episode ids to hold out: for each group, the set of whole sessions whose episode count is closest to
    `share` of the group's episodes. `sessions` maps each group to its sessions, and each session to its episode ids.
    A seeded draw breaks a tie."""
    rng = random.Random(seed)
    held = []
    for group in sorted(sessions):
        by_session = sessions[group]
        target = share * sum(len(ids) for ids in by_session.values())
        names = sorted(by_session)
        choices = [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)]
        rng.shuffle(choices)
        best = min(choices, key=lambda chosen: abs(sum(len(by_session[name]) for name in chosen) - target))
        held += [episode for name in best for episode in by_session[name]]
    return sorted(held)
