# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Regenerate the synthetic MolmoSpaces raw-observation fixture used by test_adapter.py.

``env.py`` reports the clean raw payload the adapter maps into the canonical contract — measured joints and
velocities, the eef world pose, the grip closure, and one frame per camera. MolmoSpaces renders real MuJoCo
scenes needing the full asset stack and a GPU, so committing a real payload is impractical; the adapter under
test only touches observation *structure* (keys, shapes, dtypes, the MujocoFrankaState assembly, camera key
mapping), which a tiny synthetic payload exercises exactly. Frames are small (36x64, the DROID 16:9 aspect) and
color-marked so a wrist/exterior swap is visible.

Run: uv run --no-project positronic/simulator/molmo_spaces/tests/make_fixture.py
Output: droid_obs.npz next to this script (well under 100 KB)
"""

from pathlib import Path

import numpy as np

RIG_HEIGHT, RIG_WIDTH = 36, 64  # (H, W); DROID exo/wrist cameras are 16:9.


def _marked_frame(base_rgb: tuple[int, int, int]) -> np.ndarray:
    """A solid-color frame with a white top-left block — an orientation marker a flip or swap would move."""
    frame = np.zeros((RIG_HEIGHT, RIG_WIDTH, 3), dtype=np.uint8)
    frame[:] = base_rgb
    frame[:8, :12] = 255
    return frame


def build_payload() -> dict[str, np.ndarray]:
    return {
        'joint_pos': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32),
        'joint_vel': np.linspace(-0.2, 0.2, 7, dtype=np.float32),
        'eef_pos': np.array([0.4, 0.0, 0.35], dtype=np.float32),
        # Identity orientation, scalar-first (wxyz) as env.py reports via mju_mat2Quat.
        'eef_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'grip': np.float32(0.5),
        'wrist_camera': _marked_frame((200, 40, 40)),  # reddish wrist view
        'exo_camera_1': _marked_frame((40, 160, 40)),  # greenish exterior view
    }


def main() -> None:
    out = Path(__file__).parent / 'droid_obs.npz'
    np.savez_compressed(out, **build_payload())
    print(f'Wrote {out} ({out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
