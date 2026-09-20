"""Generate synthetic observations for test_adapter.py, without simulator assets.

Camera images have distinct colours and corner markers to expose swaps and flips.

Run: uv run --locked python -m positronic.simulator.molmo_spaces.tests.make_fixture
Output: droid_obs.npz next to this script (well under 100 KB)
"""

from pathlib import Path
from typing import Any

import numpy as np

from positronic import keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import CAMERAS

# Legacy DROID benchmark camera names.
_WRIST, _EXTERIOR = (CAMERAS[k][-1] for k in (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE))

RIG_HEIGHT, RIG_WIDTH = 36, 64  # DROID camera aspect ratio.


def _marked_frame(base_rgb: tuple[int, int, int]) -> np.ndarray:
    """A solid-colour RGB image with a white top-left marker."""
    frame = np.zeros((RIG_HEIGHT, RIG_WIDTH, 3), dtype=np.uint8)
    frame[:] = base_rgb
    frame[:8, :12] = 255
    return frame


def build_payload() -> dict[str, Any]:
    return {
        mapping.OBS_JOINT_POS: np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32),
        mapping.OBS_JOINT_VEL: np.linspace(-0.2, 0.2, 7, dtype=np.float32),
        mapping.OBS_EEF_POS: np.array([0.4, 0.0, 0.35], dtype=np.float32),
        mapping.OBS_EEF_QUAT: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        mapping.OBS_GRIP: np.float32(0.5),
        _WRIST: _marked_frame((200, 40, 40)),
        _EXTERIOR: _marked_frame((40, 160, 40)),
    }


def main() -> None:
    out = Path(__file__).parent / 'droid_obs.npz'
    np.savez_compressed(out, **build_payload())  # pyright: ignore[reportArgumentType] -- numpy's savez **kwds stub
    print(f'Wrote {out} ({out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
