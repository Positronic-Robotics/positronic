"""Unit tests for ``MolmoAdapter``: the raw env-server payload -> canonical embodiment contract.

Runs without molmo_spaces (the env server lives in its own interpreter); it needs positronic, which is where
the adapter runs. Exercises the observation mapping against a synthetic raw payload (``droid_obs.npz``), the
terminal, and the reset token.

Run:  uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_adapter.py --no-cov
"""

from pathlib import Path

import numpy as np

from positronic import keys
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import DEFAULT_CAMERA_DICT as CAMERA_DICT
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter

FIXTURE = Path(__file__).parent / 'droid_obs.npz'


def _payload() -> dict:
    return dict(np.load(FIXTURE).items())


def test_observations_assemble_robot_state():
    payload = _payload()
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    state = obs[keys.ROBOT_STATE]
    assert np.allclose(state.q, payload[mapping.OBS_JOINT_POS])
    assert np.allclose(state.dq, payload[mapping.OBS_JOINT_VEL])
    assert np.allclose(state.ee_pose.translation, payload[mapping.OBS_EEF_POS])
    assert np.allclose(state.ee_pose.rotation.as_quat, payload[mapping.OBS_EEF_QUAT])  # wxyz round-trips
    assert obs[keys.GRIP] == 0.5


def test_observations_camera_passthrough_no_swap():
    payload = _payload()
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    # Frames pass through untouched (no resize/flip — the codec/client own preprocessing/transport).
    assert np.array_equal(obs[keys.WRIST_IMAGE].array, payload[mapping.MOLMO_WRIST_CAMERA])
    assert np.array_equal(obs[keys.EXTERIOR_IMAGE].array, payload[mapping.MOLMO_EXTERIOR_CAMERA])
    # Fixture marks wrist reddish, exterior greenish; a swap would flip the dominant channel.
    wrist_mean = obs[keys.WRIST_IMAGE].array.reshape(-1, 3).mean(axis=0)
    exterior_mean = obs[keys.EXTERIOR_IMAGE].array.reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]
    assert exterior_mean[1] > exterior_mean[0]


def test_observations_resolve_benchmark_variant_camera():
    # A Zed-wrist benchmark replaces the default key; the adapter must still land the reddish wrist view on
    # image.wrist.
    payload = _payload()
    payload[mapping.MOLMO_WRIST_CAMERA_VARIANTS[0]] = payload.pop(mapping.MOLMO_WRIST_CAMERA)
    obs = MolmoAdapter(CAMERA_DICT).observations(payload)
    wrist_mean = obs[keys.WRIST_IMAGE].array.reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]


def test_privileged_forwards_sim_state():
    # The full MuJoCo state is recorded as privileged ground truth (never fed to the policy), so success can be
    # recomputed offline.
    state = np.arange(10, dtype=np.float64)
    out = MolmoAdapter(CAMERA_DICT).privileged({mapping.OBS_SIM_STATE: state})
    assert list(out) == [mapping.OBS_SIM_STATE] and out[mapping.OBS_SIM_STATE] is state


def test_terminal_reports_success_only_when_done():
    adapter = MolmoAdapter(CAMERA_DICT)
    done_ok = {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: True}
    done_fail = {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False}
    running = {protocol.FRAME_DONE: False, protocol.FRAME_SUCCESS: False}
    assert adapter.terminal(done_ok) == {eval_keys.SUCCESS: True}
    assert adapter.terminal(done_fail) == {eval_keys.SUCCESS: False}
    assert adapter.terminal(running) is None


def test_task_params_name_an_episode_the_way_the_reset_token_reads_it():
    adapter = MolmoAdapter(CAMERA_DICT)
    params = adapter.task_params([{'name': 'put the banana in the bowl', 'episode_index': 3, 'task_horizon_sec': 30.0}])
    assert params == [
        {eval_keys.TASK: 'put the banana in the bowl', molmo_keys.EPISODE_INDEX: 3, molmo_keys.TASK_HORIZON: 30.0}
    ]


def test_reset_token_carries_episode_and_seed():
    adapter = MolmoAdapter(CAMERA_DICT)
    expected = {mapping.TOKEN_EPISODE_INDEX: 3, mapping.TOKEN_SEED: 7}
    assert adapter.reset_token({molmo_keys.EPISODE_INDEX: 3, eval_keys.SEED: 7}) == expected
    # An absent seed falls back to the spec's own (None here).
    assert adapter.reset_token({molmo_keys.EPISODE_INDEX: 2}) == {
        mapping.TOKEN_EPISODE_INDEX: 2,
        mapping.TOKEN_SEED: None,
    }
