# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Regenerate the synthetic MolmoSpaces DROID observation fixture used by test_adapter.py.

MolmoSpaces renders real MuJoCo scenes that need the full asset stack and a GPU, so committing a real render is
impractical; the adapter under test only touches observation *structure* (keys, shapes, dtypes, gripper qpos
scaling, image resize), which a tiny synthetic frame exercises exactly. Images are kept small (36x64, the DROID
16:9 aspect) and orientation-marked so a resize regression or a wrist/exterior swap is visible.

Run: uv run --no-project positronic/simulator/molmo_spaces/tests/make_fixture.py
Output: droid_obs.npz next to this script  (well under 100 KB)
"""

from pathlib import Path

import numpy as np

RIG_HEIGHT, RIG_WIDTH = 36, 64  # (H, W); DROID exo/wrist cameras are 16:9.


def _marked_frame(base_rgb: tuple[int, int, int]) -> np.ndarray:
    """A solid-color frame with a white top-left block and a black bottom-right block — an orientation marker.

    resize_with_pad preserves orientation, so a vertical flip or a left/right swap moves these markers detectably.
    """
    frame = np.zeros((RIG_HEIGHT, RIG_WIDTH, 3), dtype=np.uint8)
    frame[:] = base_rgb
    frame[:8, :12] = 255  # top-left white
    frame[-8:, -12:] = 0  # bottom-right black
    return frame


def build_observation() -> dict[str, np.ndarray]:
    wrist = _marked_frame((200, 40, 40))  # reddish wrist view
    exterior = _marked_frame((40, 160, 40))  # greenish exterior view
    qpos_arm = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0], dtype=np.float32)  # FR3 init_qpos
    qpos_gripper = np.array([0.412016, 0.412016], dtype=np.float32)  # half of GRIPPER_QPOS_CLOSED -> grip 0.5
    return {'wrist_camera': wrist, 'exo_camera_1': exterior, 'qpos_arm': qpos_arm, 'qpos_gripper': qpos_gripper}


def main() -> None:
    out = Path(__file__).parent / 'droid_obs.npz'
    obs = build_observation()
    np.savez_compressed(out, **obs)
    print(f'Wrote {out} ({out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
