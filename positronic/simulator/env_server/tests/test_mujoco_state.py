"""A MuJoCo scene's integration state, as an env server reports it."""

import mujoco
import numpy as np

from positronic.simulator.env_server import mujoco_state

_SCENE = '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom size="0.05" mass="0.1"/></body></worldbody></mujoco>'


def test_the_integration_state_restores_the_scene_and_replays_it_exactly():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    data.qvel[:3] = [0.3, -0.2, 0.5]
    state = mujoco_state.integration_state(model, data)
    for _ in range(20):
        mujoco.mj_step(model, data)
    expected_qpos = data.qpos.copy()

    replay = mujoco.MjData(model)
    mujoco.mj_setState(model, replay, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    for _ in range(20):
        mujoco.mj_step(model, replay)

    assert state.dtype == np.float64
    assert state.size == mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(replay.qpos, expected_qpos)
