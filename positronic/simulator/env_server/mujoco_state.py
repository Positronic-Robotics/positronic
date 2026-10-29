"""The physics state of a MuJoCo scene, as an environment server reports it.

This module must work in an isolated interpreter without Positronic installed.
"""

import mujoco
import numpy as np


def integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """MuJoCo's ``mjSTATE_INTEGRATION`` vector: the state that restores the scene and replays it exactly."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state
