"""MolmoAdapter checks using synthetic observations; no MolmoSpaces installation is required."""

from pathlib import Path

import numpy as np
import pytest

from positronic import keys
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import CAMERAS, MolmoAdapter

FIXTURE = Path(__file__).parent / 'droid_obs.npz'


def _payload() -> dict:
    return dict(np.load(FIXTURE).items())


def test_observations_assemble_robot_state():
    payload = _payload()
    obs = MolmoAdapter().observations(payload)
    state = obs[keys.ROBOT_STATE]
    assert np.allclose(state.q, payload[mapping.OBS_JOINT_POS])
    assert np.allclose(state.dq, payload[mapping.OBS_JOINT_VEL])
    assert np.allclose(state.ee_pose.translation, payload[mapping.OBS_EEF_POS])
    assert np.allclose(state.ee_pose.rotation.as_quat, payload[mapping.OBS_EEF_QUAT])
    assert obs[keys.GRIP] == 0.5


_WRIST_NEW, _WRIST_OLD = CAMERAS[keys.WRIST_IMAGE]
_EXTERIOR_OLD = CAMERAS[keys.EXTERIOR_IMAGE][-1]


def test_observations_camera_passthrough_no_swap():
    payload = _payload()
    obs = MolmoAdapter().observations(payload)
    assert np.array_equal(obs[keys.WRIST_IMAGE].array, payload[_WRIST_OLD])
    assert np.array_equal(obs[keys.EXTERIOR_IMAGE].array, payload[_EXTERIOR_OLD])
    # Fixture colours identify the cameras independently of their keys.
    wrist_mean = obs[keys.WRIST_IMAGE].array.reshape(-1, 3).mean(axis=0)
    exterior_mean = obs[keys.EXTERIOR_IMAGE].array.reshape(-1, 3).mean(axis=0)
    assert wrist_mean[0] > wrist_mean[1]
    assert exterior_mean[1] > exterior_mean[0]


def test_observations_prefer_the_first_camera_name_the_benchmark_carries():
    payload = _payload()
    payload[_WRIST_NEW] = payload.pop(_WRIST_OLD)
    obs = MolmoAdapter().observations(payload)
    assert np.array_equal(obs[keys.WRIST_IMAGE].array, payload[_WRIST_NEW])
    payload[_WRIST_OLD] = np.zeros_like(payload[_WRIST_NEW])
    obs = MolmoAdapter().observations(payload)
    assert np.array_equal(obs[keys.WRIST_IMAGE].array, payload[_WRIST_NEW])


def test_observations_fail_on_the_last_camera_name_when_none_is_present():
    payload = _payload()
    del payload[_WRIST_OLD]
    with pytest.raises(KeyError, match=_WRIST_OLD):
        MolmoAdapter().observations(payload)


def test_privileged_forwards_sim_state():
    state = np.arange(10, dtype=np.float64)
    out = MolmoAdapter().privileged({mapping.OBS_SIM_STATE: state})
    assert list(out) == [mapping.OBS_SIM_STATE] and out[mapping.OBS_SIM_STATE] is state


def test_terminal_reports_success_only_when_done():
    adapter = MolmoAdapter()
    done_ok = {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: True}
    done_fail = {protocol.FRAME_DONE: True, protocol.FRAME_SUCCESS: False}
    running = {protocol.FRAME_DONE: False, protocol.FRAME_SUCCESS: False}
    assert adapter.terminal(done_ok) == {eval_keys.SUCCESS: True}
    assert adapter.terminal(done_fail) == {eval_keys.SUCCESS: False}
    assert adapter.terminal(running) is None


_BENCH = mapping.BenchmarkPath('molmospaces-bench-v1', 'procthor-10k', 'FrankaPickDroidMiniBench', 'pick_20251231')
_BENCH_PARAMS = dict(zip(molmo_keys.BENCHMARK_DIMENSIONS, _BENCH, strict=True))


def test_task_params_name_an_episode_the_way_the_reset_token_reads_it():
    adapter = MolmoAdapter()
    record = {
        **_BENCH._asdict(),
        mapping.TASK_NAME: 'put the banana in the bowl',
        mapping.TOKEN_EPISODE_INDEX: 3,
        mapping.TASK_HORIZON_SEC: 30.0,
    }
    assert adapter.task_params([record]) == [
        {
            **_BENCH_PARAMS,
            eval_keys.TASK: 'put the banana in the bowl',
            molmo_keys.EPISODE_INDEX: 3,
            molmo_keys.TASK_HORIZON: 30.0,
        }
    ]


def test_reset_token_carries_benchmark_episode_and_seed():
    adapter = MolmoAdapter()
    expected = {**_BENCH._asdict(), mapping.TOKEN_EPISODE_INDEX: 3, mapping.TOKEN_SEED: 7}
    assert adapter.reset_token({**_BENCH_PARAMS, molmo_keys.EPISODE_INDEX: 3, eval_keys.SEED: 7}) == expected
    assert adapter.reset_token({**_BENCH_PARAMS, molmo_keys.EPISODE_INDEX: 2}) == {
        **_BENCH._asdict(),
        mapping.TOKEN_EPISODE_INDEX: 2,
        mapping.TOKEN_SEED: None,
    }
